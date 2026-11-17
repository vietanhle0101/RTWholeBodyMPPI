"""Baseline: RTWholeBodyMPPI's own box-push controller on Go1 in MuJoCo (their notebook loop).

Runs in either env from the repo root (.venv = mujoco 3.1.6, .venv-ksos = mujoco 3.16 + JAX):

    .venv-ksos/bin/python legged_mppi/scripts/go1_push_baseline.py --goal 2 -0.75 --seeds 0-19 --jobs 4
    .venv-ksos/bin/python legged_mppi/scripts/go1_push_baseline.py --summary logs/go1_push_rtwb/<run>

Each run logs to logs/go1_push_rtwb/<timestamp>_mj<version>_goal<x>_<y>/seed<N>.npz.

Box starts at (1, 0) (keyframe "home"), robot at the origin, so --goal 2 0 is the paper's
"1 m forward" case and --goal 2 -0.75 the "1 m forward, 0.75 m right" case. A run succeeds and stops
when the box is within --tol of the goal at any time (paper's metric); otherwise it stops at --time.
"""
import argparse
import glob
import os
from datetime import datetime
from multiprocessing import Pool

import mujoco
import numpy as np

SCENE = "models/go1/go1_scene_mppi_pyr_push_box_14in.xml"  # 3.7 kg, 0.38 m box: matches the hardware box


def run_seed(seed, goal, tol, t_final, out_dir):
    import whole_body_mppi
    from whole_body_mppi.control.controllers.mppi_locomanipulation import MPPI_box_push

    ctrl = MPPI_box_push("push_box")
    ctrl.random_generator = np.random.default_rng(seed)
    ctrl.x_box_ref[:2] = goal

    model = mujoco.MjModel.from_xml_path(os.path.join(os.path.dirname(whole_body_mppi.__file__), SCENE))
    model.opt.timestep = 0.01
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    box, robot, phase = [], [], []
    t_goal = np.nan
    for k in range(int(t_final / 0.01)):
        data.ctrl[:] = ctrl.update(np.r_[data.qpos, data.qvel])
        mujoco.mj_step(model, data)
        if np.linalg.norm(ctrl.body_ref[:3] - data.qpos[7:10]) < 0.1:  # same gate as their notebook
            ctrl.next_goal()

        box.append(data.qpos[:7].copy())
        robot.append(data.qpos[7:14].copy())
        phase.append(ctrl.goal_index)
        if np.linalg.norm(data.qpos[:2] - goal) < tol:
            t_goal = (k + 1) * 0.01
            break

    box, robot = np.array(box), np.array(robot)
    np.savez(os.path.join(out_dir, f"seed{seed}.npz"), box=box, robot=robot, phase=np.array(phase),
             goal=goal, tol=tol, t_goal=t_goal, mujoco_version=mujoco.__version__)
    err = np.linalg.norm(box[-1, :2] - goal)
    print(f"seed {seed}: {'OK ' if np.isfinite(t_goal) else 'FAIL'} t_goal={t_goal:.2f} "
          f"final_err={err:.3f} min_trunk_z={robot[:, 2].min():.3f}", flush=True)
    ctrl.shutdown()


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return c - h, c + h


def summary(run_dir):
    runs = [np.load(f) for f in sorted(glob.glob(f"{run_dir}/seed*.npz"))]
    ok = np.array([np.isfinite(r["t_goal"]) for r in runs])
    err = np.array([np.linalg.norm(r["box"][-1, :2] - r["goal"]) for r in runs])
    best = np.array([np.linalg.norm(r["box"][:, :2] - r["goal"], axis=1).min() for r in runs])
    fell = np.array([r["robot"][100:, 2].min() < 0.15 for r in runs])  # skip the 1 s start-up dip
    t = np.array([float(r["t_goal"]) for r in runs])
    lo, hi = wilson(ok.sum(), len(ok))
    print(f"{run_dir}: mujoco {runs[0]['mujoco_version']}, goal {runs[0]['goal']}, n={len(ok)}")
    print(f"  success {ok.sum()}/{len(ok)} = {ok.mean():.2f}  (95% Wilson {lo:.2f}-{hi:.2f})")
    if ok.any():
        print(f"  time to goal (successes) median {np.median(t[ok]):.1f} s, range {t[ok].min():.1f}-{t[ok].max():.1f} s")
    print(f"  final box error median {np.median(err):.3f} m, closest approach median {np.median(best):.3f} m")
    print(f"  robot fell (trunk z < 0.15 m after 1 s) in {fell.sum()} runs")


def seed_list(spec):
    if "-" in spec:
        a, b = map(int, spec.split("-"))
        return list(range(a, b + 1))
    return [int(s) for s in spec.split(",")]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--goal", type=float, nargs=2, default=[2.0, -0.75])
    p.add_argument("--seeds", default="0-19")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--tol", type=float, default=0.3)
    p.add_argument("--time", type=float, default=60.0)
    p.add_argument("--summary", help="print stats for an existing run dir and exit")
    args = p.parse_args()

    if args.summary:
        summary(args.summary)
        raise SystemExit
    goal = np.array(args.goal)
    tag = f"{datetime.now():%Y%m%d_%H%M%S}_mj{mujoco.__version__}_goal{goal[0]:g}_{goal[1]:g}"
    out = os.path.join("logs", "go1_push_rtwb", tag)
    os.makedirs(out)
    print("logging to", out)
    with Pool(args.jobs) as pool:
        pool.starmap(run_seed, [(s, goal, args.tol, args.time, out) for s in seed_list(args.seeds)])
    summary(out)
