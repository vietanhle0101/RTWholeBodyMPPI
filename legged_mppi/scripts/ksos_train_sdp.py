"""Train the learned KSOS solver (whole_body_mppi/ksos/sdp_l2o.py) for the planner's SDP with fixed per-knot
length scales (planner.FIXED_ELL), on replans recorded by run_push.py --ksos-kernel fixed --record, and
compare it with Newton on held-out episodes.

    .venv-ksos/bin/python legged_mppi/scripts/ksos_train_sdp.py --data logs/sdp_data_20261119 \
        --output logs/ksos_sdp/fixed1

--data: run directories with sdp_seed<N>.npz, or directories of a run queue (each <name>.txt there says
where its run logged). Episodes (seeds) are split into train / val / test.

Loss, label-free as in Global-MPPI's ksos_l2o.train_solver: the normalized dual objective
(f^T alpha - min f) / std f of every layer (deep supervision), on the knots divided by the length scales.
"""
import argparse
import glob
import json
import os
import re
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import newton_sos
import numpy as np
import optax

from whole_body_mppi.ksos.planner import FIXED_ELL
from whole_body_mppi.ksos.sdp_l2o import config, scale_of, solve, solve_path
from ksos_l2o.sdp_solver import init_solver
from ksos_l2o.checkpoint import load_pytree, save_pytree
from ksos_l2o.train_sampler import plateau_step

ACTIVE = np.array([0, 2])  # trunk controls: forward, lateral (pinned to 0), turn


def episode_files(paths):
    files = []
    for path in paths:
        files += sorted(glob.glob(os.path.join(path, "sdp_seed*.npz")))
        for txt in sorted(glob.glob(os.path.join(path, "*.txt"))):
            m = re.search(r"logging to (\S+)", open(txt).read())
            if m and not os.path.basename(txt).startswith(("smoke", "progress")):
                files += sorted(glob.glob(os.path.join(m.group(1), "sdp_seed*.npz")))
    return files


def load_split(files, val_frac, test_frac, seed):
    """{train, val, test}: x (problems, n, dim), f (problems, n), context, split by episode."""
    data = [np.load(f) for f in files]
    order = np.random.default_rng(seed).permutation(len(files))
    n_test, n_val = max(1, round(test_frac * len(files))), max(1, round(val_frac * len(files)))
    sets = {}
    for name, idx in (("test", order[:n_test]), ("val", order[n_test:n_test + n_val]), ("train", order[n_test + n_val:])):
        d = [data[i] for i in idx]
        sets[name] = dict(x=np.concatenate([e["knots"][..., ACTIVE].reshape(*e["lse_cost"].shape, -1) for e in d]),
                          f=np.concatenate([e["lse_cost"] for e in d]), context=np.concatenate([e["context"] for e in d]))
    return {name: {k: v.astype(np.float64) for k, v in s.items()} for name, s in sets.items()}


def train(args, cfg, sets, scale, lb, ub, out):
    tr, va = ([jnp.asarray(sets[s][k]) for k in ("x", "f", "context")] for s in ("train", "val"))
    k = jnp.arange(cfg.newton_steps + 1)
    w = args.decay ** (cfg.newton_steps - k)  # deep supervision, 1 on the last layer
    dual = jax.vmap(lambda prm, x, f, c: (solve_path(prm, x, f, c, scale, lb, ub, cfg) @ f - f.min()) / (f.std() + 1e-9),
                    in_axes=(None, 0, 0, 0))
    params = init_solver(jax.random.PRNGKey(args.seed), tr[0].shape[-1], tr[2].shape[-1], cfg)
    opt = optax.inject_hyperparams(lambda learning_rate: optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(learning_rate)))(learning_rate=args.lr)
    state = opt.init(params)

    @jax.jit
    def step(params, state, *batch):
        loss, g = jax.value_and_grad(lambda prm: jnp.mean(dual(prm, *batch) @ w))(params)
        upd, state = opt.update(g, state, params)
        return optax.apply_updates(params, upd), state, loss

    val_dual = jax.jit(lambda prm: jnp.mean(dual(prm, *va)[:, -1]))
    print(f"train {len(tr[0])} / val {len(va[0])} / test {len(sets['test']['x'])} problems, n = {tr[0].shape[1]}, "
          f"dim = {tr[0].shape[2]}, {cfg.newton_steps} Newton layers", flush=True)
    key, best, lr, plateau_best, plateau_stale, stale = jax.random.PRNGKey(args.seed + 1), np.inf, args.lr, np.inf, 0, 0
    for epoch in range(1, args.epochs + 1):
        losses, t0 = [], time.time()
        for _ in range(args.steps_per_epoch):
            key, sub = jax.random.split(key)
            idx = jax.random.randint(sub, (args.batch_size,), 0, len(tr[0]))
            params, state, loss = step(params, state, *(a[idx] for a in tr))
            losses.append(float(loss))
        vd = float(val_dual(params))
        if vd < best:
            best, stale = vd, 0
            save_pytree(params, out / "checkpoint.npz")
        else:
            stale += 1
        plateau_best, plateau_stale, lr, reduced = plateau_step(vd, plateau_best, plateau_stale, lr, patience=3,
                                                                factor=0.5, minimum=2e-5, min_delta=1e-5)
        state.hyperparams["learning_rate"] = lr
        print(f"epoch {epoch:3d} | train loss {np.mean(losses):.4f} | val dual {vd:.4f} | lr {lr:.1e}"
              f"{' (reduced)' if reduced else ''} | {time.time() - t0:.0f} s", flush=True)
        if stale >= 10:
            print("early stopping")
            break


def report(cfg, params, test, scale, lb, ub):
    x, f, c = (test[k] for k in ("x", "f", "context"))
    run = jax.jit(lambda x, f, c: solve(params, x, f, c, scale, lb, ub, cfg)[1])
    a = np.asarray(jax.vmap(run)(x, f, c))
    t0 = time.perf_counter()
    a_n = []
    for xi, fi in zip(x, f):  # the planner's Newton, on all candidates
        prob = newton_sos.Problem(cfg.lambd, 1e-2 / len(fi), xi / np.asarray(scale), fi)
        prob.initialize_native_kernel("Laplace", 1.0)
        a_n.append(np.ravel(newton_sos.solve(prob, max_iter=100, verbose=False, method="partial_piv_lu").alpha))
    newton_ms, a_n = 1e3 * (time.perf_counter() - t0) / len(x), np.stack(a_n)
    norm = lambda al: (np.sum(al * f, 1) - f.min(1)) / (f.std(1) + 1e-9)
    dz = np.linalg.norm(np.einsum("bn,bnd->bd", a - a_n, x), axis=1)
    best = np.argmin(f, 1)
    jax.block_until_ready(run(x[0], f[0], c[0]))
    ts = []
    for i in range(50):
        t0 = time.perf_counter(); jax.block_until_ready(run(x[i % len(x)], f[i % len(x)], c[i % len(x)])); ts.append(time.perf_counter() - t0)
    print(f"\n{len(x)} held-out problems, learned vs Newton: normalized dual gap median {np.median(norm(a) - norm(a_n)):+.4f}, "
          f"|z - z_Newton| median {np.median(dz):.3f} (p90 {np.percentile(dz, 90):.3f}; knot range 2), alpha on the best "
          f"candidate {np.median(a[np.arange(len(a)), best]):.2f} vs {np.median(a_n[np.arange(len(a)), best]):.2f}; "
          f"{1e3 * np.median(ts):.1f} ms ({jax.default_backend()}) vs {newton_ms:.1f} ms per problem")


def main(args):
    files = episode_files(args.data)
    assert files, f"no sdp_seed*.npz under {args.data}"
    sets = load_split(files, args.val_frac, args.test_frac, args.seed)
    cfg = config(newton_steps=args.newton_steps)
    x = np.concatenate([s["x"] for s in sets.values()])
    lb, ub = x.min((0, 1)), x.max((0, 1))  # the sampled box
    scale = jnp.asarray(scale_of(FIXED_ELL, x.shape[2]))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(solver=cfg.to_dict(), dim=int(x.shape[2]), context_dim=int(sets["train"]["context"].shape[1]),
                ell=list(FIXED_ELL), lb=lb.tolist(), ub=ub.tolist(), args=vars(args))
    lb, ub = jnp.asarray(lb), jnp.asarray(ub)
    if not args.eval_only:
        (out / "config.json").write_text(json.dumps(meta, indent=2))
        train(args, cfg, sets, scale, lb, ub, out)
    params = load_pytree(init_solver(jax.random.PRNGKey(0), meta["dim"], meta["context_dim"], cfg), out / "checkpoint.npz")
    report(cfg, params, sets["test"], scale, lb, ub)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", default=["logs/sdp_data_20261119"])
    p.add_argument("--output", default="logs/ksos_sdp/fixed1")
    p.add_argument("--newton-steps", type=int, default=3)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--steps-per-epoch", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--decay", type=float, default=0.5, help="deep supervision: weight decay^(K - k) of layer k")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-only", action="store_true")
    main(p.parse_args())
