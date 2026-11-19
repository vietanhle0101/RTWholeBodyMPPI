"""Step 4d: KSOS + MPPI pushing the box with the reduced trunk model alone (no Go1).

    .venv-ksos/bin/python legged_mppi/scripts/ksos_reduced.py --seeds 0-4
    .venv-ksos/bin/python legged_mppi/scripts/ksos_reduced.py --seeds 0 --view
    .venv-ksos/bin/python legged_mppi/scripts/ksos_reduced.py --replay logs/ksos_reduced/<run>/seed0.npz

The planner rolls out go1_push_trunk.xml in MJX (20 ms). The world it acts on is the same model in C
MuJoCo at 5 ms with 100 solver iterations. Replans every --replan s from the world state and applies
the planned controls until the next replan. Logs to logs/ksos_reduced/<time>_goal<x>_<y>/ with a plot.
"""
import argparse
import os
import time
from datetime import datetime

import mujoco
import numpy as np

from scripted_push import seed_list

SIM_DT = 0.005


def open_viewer(m, d):
    """MuJoCo's own passive viewer (mujoco_viewer 0.1.4 leaves new mjvGeom fields unset on MuJoCo 3.16
    and crashes at random when drawing markers on this model)."""
    import mujoco.viewer
    viewer = mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False)
    viewer.cam.distance, viewer.cam.elevation, viewer.cam.lookat[:] = 3.5, -60, [1.2, -0.4, 0.2]
    return viewer


def show(viewer, points=(), dt=0.0, t0=None):
    """Draw magenta dots at the given xy points, sync, and wait so frames come every dt s of wall time."""
    with viewer.lock():
        scn = viewer.user_scn
        scn.ngeom = 0
        for xy in points:
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.015, 0, 0]),
                                np.r_[xy, 0.3], np.eye(3).ravel(), np.array([1, 0, 1, 1], np.float32))
            scn.ngeom += 1
    viewer.sync()
    if t0 is not None:
        time.sleep(max(dt - (time.time() - t0), 0))


def run(seed, planner, args, out_dir):
    from whole_body_mppi.ksos.task import XML
    m = mujoco.MjModel.from_xml_path(XML)
    m.opt.timestep, m.opt.iterations = SIM_DT, 100
    d = mujoco.MjData(m)
    d.mocap_pos[0, :2] = args.goal
    mujoco.mj_forward(m, d)
    goal, box0 = np.array(args.goal), d.qpos[:2].copy()
    planner.reset(seed)

    viewer = open_viewer(m, d) if args.view else None

    n = int(round(args.replan / SIM_DT))
    qpos, cost, solve_t, t_goal = [], [], [], np.nan
    while d.time < args.time and (viewer is None or viewer.is_running()):
        planner.solve(d)
        cost.append(planner.last_cost); solve_t.append(planner.last_time)
        plan_xy = planner.rollout(m, d)[1][::40] if viewer is not None else None  # a dot every 0.2 s
        for i, u in enumerate(planner.controls(d.time + SIM_DT * np.arange(n))):
            t_frame = time.time()
            d.ctrl[:] = u
            mujoco.mj_step(m, d)
            qpos.append(d.qpos.copy())
            if viewer is not None and i % 4 == 0 and viewer.is_running():  # 20 ms frames, real time
                show(viewer, plan_xy, 4 * SIM_DT, t_frame)
        err = np.linalg.norm(d.qpos[:2] - goal)
        if np.isnan(t_goal) and err < args.tol:
            t_goal = d.time
        if len(cost) % 25 == 0:
            print(f"   seed {seed} t={d.time:5.1f}s box err {err:.3f} m cost {cost[-1]:8.2f} "
                  f"solve {solve_t[-1]:.2f}s", flush=True)
        if err < args.done_tol:
            break
    if viewer is not None:
        viewer.close()

    qpos = np.array(qpos)
    box, trunk = qpos[:, 0:2], qpos[:, 7:9]
    np.savez(os.path.join(out_dir, f"seed{seed}.npz"), qpos=qpos, cost=cost, solve_t=solve_t, goal=goal, box0=box0,
             t_goal=t_goal)
    st = np.array(solve_t[1:]) if len(solve_t) > 1 else np.array(solve_t)   # first solve includes jit
    print(f"seed {seed}: {'OK ' if np.isfinite(t_goal) else 'FAIL'} t_goal={t_goal:.1f}s  final box err "
          f"{np.linalg.norm(box[-1] - goal):.3f} m  min trunk-box dist "
          f"{np.linalg.norm(trunk - box, axis=1).min():.3f} m  solve median {np.median(st):.2f}s "
          f"max {st.max():.2f}s", flush=True)


def replay(path, speed=1.0):
    """Play a logged run in the viewer at `speed` x real time."""
    from whole_body_mppi.ksos.task import XML
    r = np.load(path)
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    d.mocap_pos[0, :2] = r["goal"]
    Q = r["qpos"]
    viewer = open_viewer(m, d)
    every = 4  # 20 ms per frame
    for k in range(0, len(Q), every):
        if not viewer.is_running():
            return
        t0 = time.time()
        d.qpos[:], d.time = Q[k], k * SIM_DT
        mujoco.mj_forward(m, d)
        show(viewer, (), every * SIM_DT / speed, t0)
    print(f"done at t = {d.time:.1f} s; close the window to exit")
    while viewer.is_running():  # hold the last frame until the window is closed
        time.sleep(0.1)


def plot(out_dir):
    import glob
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, f in enumerate(sorted(glob.glob(f"{out_dir}/seed*.npz"))):
        r = np.load(f)
        ax.plot(r["qpos"][:, 7], r["qpos"][:, 8], c=f"C{i}", lw=1, label=f"trunk {os.path.basename(f)[:-4]}")
        ax.plot(r["qpos"][:, 0], r["qpos"][:, 1], c=f"C{i}", lw=2, ls=":")
    ax.add_patch(plt.Circle(r["goal"], 0.3, fill=False, ls="--", color="tab:green", label="goal (0.3 m)"))
    ax.add_patch(plt.Rectangle(r["box0"] - 0.19, 0.38, 0.38, fill=False, color="gray", label="box start"))
    ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.legend(fontsize=7)
    ax.set_title("KSOS on the trunk model. solid: trunk, dotted: box center")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/ksos_reduced.png", dpi=150)
    print("saved", f"{out_dir}/ksos_reduced.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--goal", type=float, nargs=2, default=[2.0, -0.75])
    p.add_argument("--time", type=float, default=60.0)
    p.add_argument("--replan", type=float, default=0.5, help="replanning period [s]")
    p.add_argument("--tol", type=float, default=0.3, help="success radius [m]")
    p.add_argument("--done-tol", type=float, default=0.1, help="stop when the box is this close [m]")
    p.add_argument("--passes", type=int, default=1, help="KSOS passes per replan")
    p.add_argument("--flow", help="trained first-pass sampler (ksos_train_flow.py output dir)")
    p.add_argument("--lse-sigma", type=float, default=0.1, help="first-pass LSE smoothing noise")
    p.add_argument("--mppi-noise", type=float, default=0.3, help="first-pass MPPI sampling noise")
    p.add_argument("--ksos-samples", type=int, default=64, help="KSOS candidates per pass")
    p.add_argument("--impl", default="jax", choices=["jax", "warp"], help="MJX backend of the planner rollouts")
    p.add_argument("--seeds", default="0-4")
    p.add_argument("--view", action="store_true", help="live viewer; runs only the first seed")
    p.add_argument("--replay", help="play a logged seed*.npz in the viewer instead of running")
    p.add_argument("--speed", type=float, default=1.0, help="replay speed (x real time)")
    args = p.parse_args()
    if args.replay:
        replay(args.replay, args.speed)
        raise SystemExit

    from whole_body_mppi.ksos.planner import KSOSPlanner
    from whole_body_mppi.ksos.task import Go1PushTask

    planner = KSOSPlanner(Go1PushTask(), passes=args.passes, flow=args.flow, lse_sigma=args.lse_sigma,
                          mppi_noise=args.mppi_noise, ksos_samples=args.ksos_samples,
                          impl=args.impl)
    out = os.path.join("logs", "ksos_reduced", f"{datetime.now():%Y%m%d_%H%M%S}_goal{args.goal[0]:g}_{args.goal[1]:g}")
    os.makedirs(out)
    print("logging to", out)
    seeds = seed_list(args.seeds)[:1] if args.view else seed_list(args.seeds)
    for s in seeds:
        run(s, planner, args, out)
    plot(out)
