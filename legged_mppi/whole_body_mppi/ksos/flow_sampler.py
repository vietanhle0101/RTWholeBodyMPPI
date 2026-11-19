"""Learned first-pass KSOS sampler for the trunk push: the conditional RealNVP of ksos_l2o.

The flow lives in the unit box of the sampled controls: forward and turn at each knot (lateral is
pinned to 0 and left out). Its context is the push geometry in the frame whose x axis points from the
box to the goal, so it does not depend on where the box and goal are. Trained by
legged_mppi/scripts/ksos_train_flow.py.
"""
import json
import os
import sys
from pathlib import Path

import global_mppi
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(global_mppi.__file__)))  # ksos_l2o is not installed
from ksos_l2o.checkpoint import load_pytree  # noqa: E402
from ksos_l2o.flow import ConditionalRealNVP  # noqa: E402

CONTEXT_DIM = 13


def quat_yaw(q):
    w, x, y, z = q
    return jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def push_context(qpos, qvel, goal):
    """Box-to-goal distance; trunk position, heading and box yaw (mod 90 deg) relative to the box,
    and box and trunk velocities, all in the box-to-goal frame. (13,)"""
    box, trunk = qpos[0:2], qpos[7:9]
    to_goal = goal - box
    psi = jnp.arctan2(to_goal[1], to_goal[0])
    c, s = jnp.cos(psi), jnp.sin(psi)
    rot = jnp.array([[c, s], [-s, c]])  # world -> goal frame
    yaw_t = qpos[9] - psi
    yaw_b = 4 * (quat_yaw(qpos[3:7]) - psi)  # the box looks the same every 90 deg
    return jnp.concatenate((
        jnp.linalg.norm(to_goal)[None], rot @ (trunk - box),
        jnp.array([jnp.sin(yaw_t), jnp.cos(yaw_t), jnp.sin(yaw_b), jnp.cos(yaw_b)]),
        rot @ qvel[0:2], qvel[5:6],   # box xy velocity, yaw rate
        rot @ qvel[6:8], qvel[8:9],   # trunk xy velocity, yaw rate
    ))


def state_context(state):
    return push_context(state.qpos, state.qvel, state.mocap_pos[0, :2])


def active_controls(task):
    """Indices of the controls that are sampled (the others have a zero-width range)."""
    return np.flatnonzero(np.asarray(task.u_max) > np.asarray(task.u_min))


def unit_to_knots(unit, task, num_knots):
    """(..., num_knots * n_active) in the unit box -> knots (..., num_knots, nu); pinned controls keep
    their fixed value."""
    active = active_controls(task)
    shape = unit.shape[:-1] + (num_knots, task.model.nu)
    full = jnp.zeros(shape).at[..., active].set(unit.reshape(*shape[:-1], len(active)))
    return task.u_min + (task.u_max - task.u_min) * full


def flow_dim(task, num_knots):
    return num_knots * len(active_controls(task))


def load_flow(flow_dir):
    flow_dir = Path(flow_dir)
    config = json.loads((flow_dir / "config.json").read_text())
    flow = ConditionalRealNVP(config["dim"], config["context_dim"], num_layers=config["layers"],
                              hidden_dim=config["hidden"])
    params = load_pytree(flow.init(jax.random.PRNGKey(0)), flow_dir / "checkpoint.npz")
    return flow, jax.tree_util.tree_map(jnp.asarray, params), config


def make_flow_sampler(flow_dir, task, num_knots, lse_sigma):
    """sampler(state, key, n) -> knots (n, num_knots, nu), for GlobalMPPI's ksos_flow_sampler."""
    flow, params, config = load_flow(flow_dir)
    assert config["num_knots"] == num_knots, f"flow trained with {config['num_knots']} knots"
    if config.get("cost") == "lse":  # it learned the LSE cost at one sigma (flows from before --lse-sigma: 0.3)
        trained = config.get("lse_sigma") or 0.3
        assert trained == lse_sigma, f"flow trained with --lse-sigma {trained}, planner uses {lse_sigma}"
    zeros, ones = jnp.zeros(config["dim"]), jnp.ones(config["dim"])

    def sampler(state, key, n):
        z = jax.random.normal(key, (1, n, config["dim"]))
        unit, _ = flow.sample_batch(params, z, state_context(state)[None], zeros, ones)
        return unit_to_knots(unit[0], task, num_knots)

    return sampler
