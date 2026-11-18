"""Step 4b: is the reduced trunk push model trustworthy in MJX, and at which settings?

    .venv-ksos/bin/python legged_mppi/scripts/ksos_check_mjx.py

Open-loop scenarios with the box at rest at (1, 0). Reference: C MuJoCo at 1 ms with impratio 100
(as in the Go1 scene) and 100 solver iterations. Compared: C and MJX at each planner timestep and
contact setting. Errors are final box xy and trunk xy distances [m]. Then times batched MJX rollouts.
qpos layout: box free joint [0:7], trunk x, y, yaw [7:10]. ctrl: forward (0.1 + 0.1 * ctrl m/s), lateral (no effect), turn [rad/s].
"""
import os
import time

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

import whole_body_mppi

XML = os.path.join(os.path.dirname(whole_body_mppi.__file__), "ksos", "go1_push_trunk.xml")
DTS = [0.01, 0.02, 0.05]
VARIANTS = {"impratio 1": dict(impratio=1.0), "impratio 100": dict(impratio=100.0)}

d_diag = np.array([0.8, -0.6])
# name: (trunk x, y, yaw), ctrl (forward, lateral, turn), duration
SCENARIOS = {
    "rest":        ((-0.5, 0.0, 0.0), (-1.0, 0, 0.0), 5.0),
    "head-on":     ((0.45, 0.0, 0.0), (1.0, 0, 0.0), 6.0),
    "diagonal":    ((*(np.array([1.0, 0.0]) - 0.6 * d_diag), np.arctan2(d_diag[1], d_diag[0])), (1.0, 0, 0.0), 6.0),
    "slow push":   ((0.5, 0.0, 0.0), (-0.5, 0, 0.0), 8.0),
    "push + turn": ((0.45, 0.0, 0.0), (0.0, 0, -0.15), 8.0),
}


def model(dt=None, **opts):
    m = mujoco.MjModel.from_xml_path(XML)
    if dt is not None:
        m.opt.timestep = dt
    for k, v in opts.items():
        setattr(m.opt, k, v)
    return m


def start(m, trunk):
    d = mujoco.MjData(m)
    d.qpos[7:10] = trunk
    mujoco.mj_forward(m, d)
    return d


def run_c(m, trunk, u, T):
    d = start(m, trunk)
    d.ctrl[:] = u
    for _ in range(int(round(T / m.opt.timestep))):
        mujoco.mj_step(m, d)
    return d.qpos.copy()


def run_mjx(m, trunk, u, T):
    mx = mjx.put_model(m)
    dx = mjx.put_data(m, start(m, trunk)).replace(ctrl=jnp.asarray(u, dtype=jnp.float32))
    n = int(round(T / m.opt.timestep))
    q = np.asarray(jax.jit(lambda x: jax.lax.fori_loop(0, n, lambda i, y: mjx.step(mx, y), x))(dx).qpos)
    return q


def fmt(q, ref):
    if not np.all(np.isfinite(q)):
        return "    NaN     NaN"
    return f"{np.linalg.norm(q[:2] - ref[:2]):7.3f} {np.linalg.norm(q[7:9] - ref[7:9]):7.3f}"


def main():
    for name, (trunk, u, T) in SCENARIOS.items():
        ref = run_c(model(0.001, impratio=100.0, iterations=100, ls_iterations=50), trunk, u, T)
        yaw = np.degrees(2 * np.arctan2(ref[6], ref[3]))
        print(f"\n{name}: reference box moved {np.linalg.norm(ref[:2] - [1, 0]):.3f} m (yaw {yaw:.0f} deg), "
              f"trunk at ({ref[7]:.2f}, {ref[8]:.2f}, {np.degrees(ref[9]):.0f} deg)")
        print("  setting        dt   | C vs ref: box  trunk | MJX vs ref: box  trunk | MJX vs C: box")
        for vname, opts in VARIANTS.items():
            for dt in DTS:
                m = model(dt, **opts)
                qc, qj = run_c(m, trunk, u, T), run_mjx(m, trunk, u, T)
                mc = f"{np.linalg.norm(qj[:2] - qc[:2]):.3f}" if np.all(np.isfinite(qj)) else "NaN"
                print(f"  {vname:12s} {dt*1000:3.0f} ms |    {fmt(qc, ref)}  |      {fmt(qj, ref)}  | {mc:>8s}")

    print("\nbatched MJX rollout timing (4 s horizon, random controls, after jit):")
    for vname, opts in VARIANTS.items():
        for dt in [0.02, 0.05]:
            m = model(dt, **opts)
            mx = mjx.put_model(m)
            dx0 = mjx.put_data(m, start(m, (0.3, 0.3, 0.0)))
            H = int(round(4.0 / dt))

            def rollout(us):
                def f(x, u):
                    x = mjx.step(mx, x.replace(ctrl=u))
                    return x, x.qpos[:2]
                return jax.lax.scan(f, dx0, us)[1]

            roll = jax.jit(jax.vmap(rollout))
            lo, hi = jnp.asarray(m.actuator_ctrlrange[:, 0]), jnp.asarray(m.actuator_ctrlrange[:, 1])
            for N in [1024, 4096]:
                us = jax.random.uniform(jax.random.key(0), (N, H, 3), minval=lo, maxval=hi)
                roll(us).block_until_ready()
                t0 = time.time()
                roll(us).block_until_ready()
                print(f"  {vname:12s} dt {dt*1000:.0f} ms, {H} steps, {N} rollouts: {1000*(time.time()-t0):.0f} ms")


if __name__ == "__main__":
    main()
