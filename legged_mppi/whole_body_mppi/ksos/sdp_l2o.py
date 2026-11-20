"""Learned solver for the planner's per-knot KSOS step (KSOSPlanner with ksos_kernel="fixed").

The planner solves the KSOS SDP with Newton on the knots divided by fixed per-knot length scales, at
width 1. Here Global-MPPI's learned dual solver (ksos_l2o.sdp_solver: GNN warm start and unrolled Newton
layers) solves the same SDP. x holds the sampled controls only (the pinned lateral one is left out),
(n, num_knots * n_active); scale is the divisor of each column of x. Trained by
legged_mppi/scripts/ksos_train_sdp.py.
"""
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import global_mppi
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(global_mppi.__file__)))  # ksos_l2o is not installed
from ksos_l2o.checkpoint import load_pytree  # noqa: E402
from ksos_l2o.sdp_solver import (SolverConfig, init_solver, kernel_and_features, newton_step,  # noqa: E402
                                 predict_schedule, tau_schedule, warm_start)


def config(**kw):
    """The planner's SDP: width 1 on rescaled knots, lambda 0.1."""
    return replace(SolverConfig(sigma=1.0, lambd=0.1), **kw)


def scale_of(ell, dim):
    """(num_knots,) length scales -> (dim,) divisors of x (the active controls of a knot share one)."""
    return np.repeat(np.asarray(ell, np.float64), dim // len(ell))


def solve_path(params, x, f, context, scale, lb, ub, cfg):
    """alpha after the warm start and after every Newton layer, (newton_steps + 1, n), for the SDP on
    x / scale. The GNN sees the raw knots; the kernel and the Newton layers the rescaled ones."""
    kernel, phi = kernel_and_features(x / scale, cfg.sigma, cfg.kernel_jitter)
    alpha = warm_start(params["gnn"], x, f, kernel, phi, context, lb, ub, cfg)
    step_logit, gamma = predict_schedule(params["schedule"], context)
    taus = tau_schedule(gamma, cfg.tau, cfg.tau_final(x.shape[0]))

    def layer(alpha, step_tau):
        alpha = newton_step(alpha, f, phi, step_tau[0], cfg.lambd, step_tau[1], cfg.kkt_damping, cfg.psd_floor)
        return alpha, alpha

    _, path = jax.lax.scan(layer, alpha, (jax.nn.sigmoid(step_logit), taus))
    return jnp.concatenate((alpha[None], path))


def solve(params, x, f, context, scale, lb, ub, cfg):
    """(z, alpha). z = alpha @ x is in the raw coordinates (rescaling is linear)."""
    alpha = solve_path(params, x, f, context, scale, lb, ub, cfg)[-1]
    return alpha @ x, alpha


def load(model_dir):
    """A jitted solver(x, f, context) -> (z, alpha) in float64, and its config dict."""
    from jax.experimental import enable_x64
    meta = json.loads((Path(model_dir) / "config.json").read_text())
    cfg = config(**{k: v for k, v in meta["solver"].items() if k in SolverConfig.__dataclass_fields__})
    with enable_x64():
        params = load_pytree(init_solver(jax.random.PRNGKey(0), meta["dim"], meta["context_dim"], cfg),
                             Path(model_dir) / "checkpoint.npz")
        lb, ub, scale = (jnp.asarray(v, jnp.float64) for v in (meta["lb"], meta["ub"], scale_of(meta["ell"], meta["dim"])))
        run = jax.jit(lambda x, f, c: solve(params, x, f, c, scale, lb, ub, cfg))

    def solver(x, f, context):
        with enable_x64():
            return run(*(jnp.asarray(np.asarray(a), jnp.float64) for a in (x, f, context)))

    return solver, meta
