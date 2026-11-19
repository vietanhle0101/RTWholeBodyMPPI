"""Train the first-pass KSOS sampler (a conditional RealNVP) for the trunk push.

Same method as Global-MPPI's ksos_l2o/train_state_sampler.py, with push states and our planner's
horizon and knots. Data is generated on the fly: random push states, candidates from a mix of uniform
and the current flow, their rollout costs in MJX, and the importance-weighted forward KL

    L = -E_state sum_j w_j log q(u_j | state),   w = softmax(-cost / T - log p(u_j | state)),

with T picked per state to give the effective sample size --target-ess.

Validation mimics a replan: each state is reached by running a plan for --replan s, and that plan
shifted by --replan s is the warm-start candidate, as in the planner. The improvement is how much
lower the best of N candidates' cost is than the warm start's (0 if it is not lower), averaged over
the states. The checkpoint kept is the one with the largest ratio of flow to uniform improvement.

    .venv-ksos/bin/python legged_mppi/scripts/ksos_train_flow.py --output logs/ksos_flow/plain
    .venv-ksos/bin/python legged_mppi/scripts/ksos_train_flow.py --cost lse --output logs/ksos_flow/lse

    .venv-ksos/bin/python legged_mppi/scripts/ksos_train_flow.py --init logs/ksos_flow/plain \
        --warmup-epochs 0 --output logs/ksos_flow/plain2      # continue training

Use it with run_push.py / ksos_reduced.py --flow <output>.
"""
import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import optax
from mujoco import mjx

from whole_body_mppi.ksos.flow_sampler import CONTEXT_DIM, ConditionalRealNVP, flow_dim, state_context, unit_to_knots
from whole_body_mppi.ksos.planner import KSOSPlanner
from whole_body_mppi.ksos.task import Go1PushTask
from ksos_l2o.checkpoint import load_pytree, save_pytree
from ksos_l2o.train_sampler import plateau_step
from ksos_l2o.train_state_sampler import ess_weights

BOX, TRUNK = (0.19, 0.19), (0.28, 0.15)  # half sizes in the trunk model


def support(phi, yaw, half):
    """Half width along direction phi of a rectangle at heading yaw."""
    return half[0] * jnp.abs(jnp.cos(phi - yaw)) + half[1] * jnp.abs(jnp.sin(phi - yaw))


def sample_state(key, base, model, args):
    """Box at the origin, goal in a random direction at a log-uniform distance (a third of the states
    within 0.3 m, so the end of the push is covered). The trunk is in a random direction
    from the box, mostly facing it, touching it or up to --clearance away (no overlap: the two are
    separated along that direction), moving forward at up to 0.2 m/s."""
    k = jax.random.split(key, 11)
    uni = lambda kk, lo, hi, shape=(): jax.random.uniform(kk, shape, minval=lo, maxval=hi)
    dist = jnp.exp(uni(k[0], jnp.log(args.goal_min), jnp.log(args.goal_max)))
    phi_g = uni(k[1], -jnp.pi, jnp.pi)
    goal = dist * jnp.array([jnp.cos(phi_g), jnp.sin(phi_g)])
    yaw_b = uni(k[2], -jnp.pi / 4, jnp.pi / 4)
    phi = uni(k[3], -jnp.pi, jnp.pi)
    yaw_t = jnp.where(jax.random.bernoulli(k[4], args.face_prob), phi + jnp.pi + uni(k[5], -0.5, 0.5),
                      uni(k[6], -jnp.pi, jnp.pi))
    gap = jnp.where(jax.random.bernoulli(k[7], args.contact_prob), 0.0, uni(k[8], 0.0, args.clearance))
    trunk = (support(phi, yaw_b, BOX) + support(phi, yaw_t, TRUNK) + gap) * jnp.array([jnp.cos(phi), jnp.sin(phi)])
    qpos = jnp.concatenate((jnp.array([0.0, 0.0, 0.19, jnp.cos(yaw_b / 2), 0.0, 0.0, jnp.sin(yaw_b / 2)]),
                            trunk, yaw_t[None]))
    v, speed = uni(k[9], -1.0, 1.0, (3,)), uni(k[10], 0.0, 0.2)
    qvel = jnp.zeros(9).at[0:2].set(0.05 * v[:2]).at[6:8].set(speed * jnp.array([jnp.cos(yaw_t), jnp.sin(yaw_t)]))
    qvel = qvel.at[8].set(0.5 * v[2])
    return forward_state(base, model, qpos, qvel, goal)


def forward_state(base, model, qpos, qvel, goal):
    return mjx.forward(model, base.replace(qpos=qpos, qvel=qvel, mocap_pos=base.mocap_pos.at[0, :2].set(goal)))


def make_batch_fn(ctrl, flow, args):
    """batch(key, sampler_params, uniform_fraction, rollout_fraction) -> contexts, unit candidates, costs,
    log proposal density and the reference cost, for --batch-size states. A rollout_fraction of the
    states are reached by running the best of --derive-samples uniform plans for --replan s; their
    reference is the warm start (that plan shifted by --replan s). The others' is standing still."""
    task, n, size = ctrl.task, args.num_samples, args.batch_size
    dim, nk = flow.dim, ctrl.num_knots
    tk = jnp.linspace(0.0, ctrl.plan_horizon, nk)
    base = mjx.put_data(task.mj_model, mujoco.MjData(task.mj_model))
    zeros, ones = jnp.zeros(dim), jnp.ones(dim)
    still = unit_to_knots(jnp.full((nk, dim // nk), 0.5).at[:, 0].set(0.0).ravel(), task, nk)  # speed 0, no turn
    tq = jnp.linspace(tk[0], tk[-1], ctrl.ctrl_steps)
    steps = int(round(args.replan / ctrl.dt))

    def plain_cost(state, knots):
        return jnp.sum(ctrl.rollout(state, tk, knots).costs, axis=1)

    def cost(state, knots, key):
        if args.cost == "plain":
            return plain_cost(state, knots)
        params = ctrl.init_params().replace(rng=key, iter=0)  # iter 0: the planner's first-pass sigma
        return ctrl.lse_smoothing_rollout(state, tk, knots, params).lse_cost.reshape(knots.shape[0])

    def replan_state(state, key):
        """Run the best of --derive-samples uniform plans for --replan s; returns the state then and the
        plan shifted by --replan s, clipped as in ksos_rollout."""
        knots = unit_to_knots(jax.random.uniform(key, (args.derive_samples, dim)), task, nk)
        best = knots[jnp.argmin(plain_cost(state, knots))][None]
        states, _ = ctrl.eval_rollouts(ctrl.model, state, ctrl.interp_func(tq, tk, best)[:, :steps], best)
        warm = jnp.clip(ctrl.interp_func(tk + tq[steps], tk, best)[0], task.u_min, task.u_max)
        return forward_state(base, task.model, states.qpos[0, -1], states.qvel[0, -1], state.mocap_pos[0, :2]), warm

    def batch(key, sampler_params, uniform_fraction, rollout_fraction):
        k_state, k_roll, k_use, k_u, k_z, k_mix, k_cost = jax.random.split(key, 7)
        states = jax.vmap(lambda kk: sample_state(kk, base, task.model, args))(jax.random.split(k_state, size))
        moved, warm = jax.vmap(replan_state)(states, jax.random.split(k_roll, size))
        use = jax.random.bernoulli(k_use, rollout_fraction, (size,))
        states = jax.tree_util.tree_map(lambda a, b: jnp.where(use.reshape(-1, *[1] * (a.ndim - 1)), a, b),
                                        moved, states)
        ref = jnp.where(use[:, None, None], warm, still)
        contexts = jax.vmap(state_context)(states)
        unit = jax.random.uniform(k_u, (size, n, dim))
        from_flow, _ = flow.sample_batch(sampler_params, jax.random.normal(k_z, unit.shape), contexts, zeros, ones)
        unit = jnp.where(jax.random.bernoulli(k_mix, uniform_fraction, (size, n, 1)), unit,
                         jnp.clip(from_flow, 1e-6, 1.0 - 1e-6))  # same clip as in log_prob
        log_flow = flow.log_prob_batch(sampler_params, unit, contexts, zeros, ones)
        log_prop = jnp.logaddexp(jnp.log(uniform_fraction), jnp.log1p(-uniform_fraction) + log_flow)
        keys = jax.random.split(k_cost, size)
        costs = jax.vmap(lambda s, u, kk: cost(s, unit_to_knots(u, task, nk), kk))(states, unit, keys)
        ref_cost = jax.vmap(lambda s, r, kk: cost(s, r[None], kk)[0])(states, ref, keys)
        return contexts, unit, costs, log_prop, ref_cost

    return jax.jit(batch)


def main(args):
    task = Go1PushTask()
    ctrl = KSOSPlanner(task, lse_sigma=args.lse_sigma).ctrl  # same horizon, knots and LSE settings as the planner
    dim = flow_dim(task, ctrl.num_knots)
    flow = ConditionalRealNVP(dim, CONTEXT_DIM, num_layers=args.layers, hidden_dim=args.hidden)
    batch_fn = make_batch_fn(ctrl, flow, args)
    params = flow.init(jax.random.PRNGKey(args.seed))
    if args.init:
        params = jax.tree_util.tree_map(jnp.asarray, load_pytree(params, Path(args.init) / "checkpoint.npz"))
    zeros, ones = jnp.zeros(dim), jnp.ones(dim)
    optimizer = optax.inject_hyperparams(
        lambda learning_rate: optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate)))(learning_rate=args.lr)
    opt_state = optimizer.init(params)

    def loss_fn(prm, c, u, w):
        return -jnp.mean(jnp.sum(w * flow.log_prob_batch(prm, u, c, zeros, ones), axis=-1))

    grad_fn, val_fn = jax.jit(jax.value_and_grad(loss_fn)), jax.jit(loss_fn)
    weigh = jax.jit(lambda cost, log_prop: ess_weights(cost, args.target_ess, log_prop)[0])

    # fixed validation states (all reached as after a replan), in --batch-size chunks. Their uniform
    # candidates and weights, drawn once, give the validation loss (the same data every epoch) and the
    # reference for the ratio.
    val_keys = jax.random.split(jax.random.PRNGKey(args.seed + 500), args.val_batches)
    val_batch = lambda prm, frac: [batch_fn(k, prm, frac, 1.0) for k in val_keys]
    improvement = lambda batches: float(np.mean(np.concatenate(
        [np.maximum(np.asarray(b[4] - b[2].min(-1)), 0.0) for b in batches])))
    t0 = time.time()
    val_uniform = val_batch(params, 1.0)
    val_data = [(b[0], b[1], weigh(b[2], b[3])) for b in val_uniform]
    val_loss = lambda prm: float(np.mean([val_fn(prm, *d) for d in val_data]))
    uniform_imp = improvement(val_uniform)
    print(f"validation: {args.val_batches * args.batch_size} states, best-of-{args.num_samples} uniform improvement "
          f"over the warm start {uniform_imp:.2f} ({time.time() - t0:.0f} s incl. jit)", flush=True)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**vars(args), "dim": dim, "context_dim": CONTEXT_DIM,
                                                 "num_knots": ctrl.num_knots}, indent=2))
    best_ratio, plateau_best, stale, early, lr = 0.0, np.inf, 0, 0, args.lr
    saved = bool(args.init)
    if args.init:  # the flow we start from is the one to beat
        best_ratio = improvement(val_batch(params, 0.0)) / uniform_imp
        plateau_best = -best_ratio
        save_pytree(params, out / "checkpoint.npz")
        print(f"start from {args.init}: ratio vs uniform {best_ratio:.3f}", flush=True)
    key = jax.random.PRNGKey(args.seed + 1)
    for epoch in range(1, args.epochs + 1):
        t0, losses, sampler = time.time(), [], params  # proposal for this epoch
        mix = 1.0 if epoch <= args.warmup_epochs else args.uniform_fraction
        for _ in range(args.steps_per_epoch):
            key, sub = jax.random.split(key)
            c, u, cost, log_prop, _ = batch_fn(sub, sampler, mix, args.rollout_fraction)
            loss, grads = grad_fn(params, c, u, weigh(cost, log_prop))
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            losses.append(float(loss))
        ratio = improvement(val_batch(params, 0.0)) / uniform_imp
        plateau_best, stale, lr, reduced = plateau_step(-ratio, plateau_best, stale, lr, patience=args.lr_patience,
                                                         factor=0.5, minimum=args.lr / 50, min_delta=0.01)
        opt_state.hyperparams["learning_rate"] = lr
        improved = ratio > best_ratio + 0.01
        if improved:
            best_ratio, early, saved = ratio, 0, True
            save_pytree(params, out / "checkpoint.npz")
        else:
            early += 1
        print(f"epoch {epoch:3d} train loss {np.mean(losses):.3f}  val loss {val_loss(params):.3f}  "
              f"ratio vs uniform {ratio:.3f}  lr {lr:.1e}{' (reduced)' if reduced else ''}"
              f"{' (best, saved)' if improved else ''}", flush=True)
        if early >= args.stop_patience:
            print("early stopping")
            break
    print(f"best ratio {best_ratio:.3f}; " + (f"saved {out / 'checkpoint.npz'}" if saved else "nothing saved"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="logs/ksos_flow/plain")
    p.add_argument("--cost", choices=["plain", "lse"], default="plain",
                   help="plain rollout cost, or the planner's first-pass LSE cost (16x the rollouts)")
    p.add_argument("--lse-sigma", type=float, default=0.1, help="first-pass LSE noise; use the same in the planner")
    p.add_argument("--goal-min", type=float, default=0.1, help="box-to-goal distance range [m], log-uniform")
    p.add_argument("--goal-max", type=float, default=2.5)
    p.add_argument("--face-prob", type=float, default=0.7, help="share of states with the trunk facing the box")
    p.add_argument("--contact-prob", type=float, default=0.4, help="share of states with the trunk touching the box")
    p.add_argument("--clearance", type=float, default=0.8, help="largest trunk-box gap [m]")
    p.add_argument("--rollout-fraction", type=float, default=0.3,
                   help="share of training states reached by running a plan for --replan s")
    p.add_argument("--derive-samples", type=int, default=16, help="uniform plans tried for that plan")
    p.add_argument("--replan", type=float, default=0.5, help="planner's replanning period [s]")
    p.add_argument("--target-ess", type=float, default=0.15)
    p.add_argument("--uniform-fraction", type=float, default=0.25, help="share of uniform candidates after warm-up")
    p.add_argument("--warmup-epochs", type=int, default=1, help="initial epochs with uniform candidates only")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--steps-per-epoch", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32, help="states per step")
    p.add_argument("--num-samples", type=int, default=32, help="candidates per state")
    p.add_argument("--val-batches", type=int, default=4, help="validation states = this x --batch-size")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--lr-patience", type=int, default=5)
    p.add_argument("--stop-patience", type=int, default=12)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--init", help="continue from this output dir (same --layers and --hidden)")
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
