"""Closed-loop box push: a global planner replans the trunk path, their whole-body MPPI tracks it.

    .venv-ksos/bin/python legged_mppi/scripts/run_push.py --planner geometric --seeds 0-9
    .venv-ksos/bin/python legged_mppi/scripts/run_push.py --planner geometric --seeds 0 --view

Every --replan s the planner gets the current trunk and box xy and returns a time-indexed trunk plan;
BodyReference turns it into MPPI's per-step reference and picks the gait from the planned speed.
Logs to logs/run_push/<time>_<planner>_goal<x>_<y>/ with a plot.
"""
import argparse
import os
from datetime import datetime
from multiprocessing import Pool

import mujoco
import numpy as np

from scripted_push import plot, report, seed_list

DT = 0.01
SCENE = "models/go1/go1_scene_mppi_pyr_push_box_14in.xml"


def make_planner(args):
    if args.planner == "geometric":
        from whole_body_mppi.control.geometric_planner import GeometricPushPlanner
        return GeometricPushPlanner(args.goal)
    raise ValueError(args.planner)


def run(seed, args, out_dir):
    import whole_body_mppi
    from whole_body_mppi.control.controllers.mppi_locomanipulation import MPPI_box_push
    from whole_body_mppi.control.geometric_planner import polyline
    from whole_body_mppi.control.reference import BodyReference

    goal = np.array(args.goal)
    ctrl = MPPI_box_push("push_box")
    ctrl.random_generator = np.random.default_rng(seed)
    ctrl.x_box_ref[:2] = goal

    model = mujoco.MjModel.from_xml_path(os.path.join(os.path.dirname(whole_body_mppi.__file__), SCENE))
    model.opt.timestep = DT
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    box0 = data.qpos[:2].copy()

    planner = make_planner(args)
    ref = BodyReference(yaw0=0.0)
    ref.set_plan(*polyline(0.0, [data.qpos[7:9].copy()], []))  # stand still until the first replan
    next_replan = args.hold

    viewer = None
    if args.view:
        import mujoco_viewer
        viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=True)
        viewer.cam.distance, viewer.cam.elevation = 3.5, -35
        viewer.cam.lookat[:] = [1.0, 0.0, 0.2]

    box, trunk, refs, modes = [], [], [], []
    t_goal, t_done = np.nan, None
    for k in range(int(args.time / DT)):
        if data.time >= next_replan - 1e-9:
            ref.set_plan(*planner.plan(data.time, data.qpos[7:9].copy(), data.qpos[:2].copy()))
            next_replan += args.replan
        ctrl.ref_traj = ref.horizon(data.time)
        ctrl.gait_scheduler = ctrl.gaits[ref.gait(data.time)]
        data.ctrl[:] = ctrl.update(np.r_[data.qpos, data.qvel])
        mujoco.mj_step(model, data)

        box.append(data.qpos[:7].copy()); trunk.append(data.qpos[7:14].copy())
        refs.append(ctrl.ref_traj[0, :3].copy()); modes.append(planner.mode)
        if np.isnan(t_goal) and np.linalg.norm(data.qpos[:2] - goal) < args.tol:
            t_goal = data.time
        if planner.mode == "done" and t_done is None:
            t_done = data.time
        if t_done is not None and data.time > t_done + 2.0:
            break

        if viewer is not None:
            if not viewer.is_alive:
                break
            if k % 2 == 0:
                for pos, rgba in ((ctrl.ref_traj[0, :3], [1, 0, 1, 1]), (np.r_[goal, 0.2], [1, 1, 0, 1])):
                    viewer.add_marker(pos=pos * 1, size=[0.06] * 3, rgba=rgba,
                                      type=mujoco.mjtGeom.mjGEOM_SPHERE, label=planner.mode if pos is not None else "")
                viewer.render()
    ctrl.shutdown()
    if viewer is not None and viewer.is_alive:
        viewer.close()

    box, trunk, refs = np.array(box), np.array(trunk), np.array(refs)
    switches = sum(a != b for a, b in zip(modes[:-1], modes[1:]))
    np.savez(os.path.join(out_dir, f"seed{seed}.npz"), box=box, trunk=trunk, ref=refs, plan_P=refs[:, :2],
             goal=goal, t_goal=t_goal, modes=np.array(modes))
    d = (goal - box0) / np.linalg.norm(goal - box0)
    report(seed, box, trunk, refs, goal, box0, d, t_goal)
    print(f"   seed {seed}: mode switches {switches}, finished {'%.1f s' % t_done if t_done else 'no'}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--planner", default="geometric", choices=["geometric"])
    p.add_argument("--goal", type=float, nargs=2, default=[2.0, -0.75])
    p.add_argument("--replan", type=float, default=0.2, help="replanning period [s]")
    p.add_argument("--hold", type=float, default=1.0, help="stand still before the first plan [s]")
    p.add_argument("--time", type=float, default=60.0)
    p.add_argument("--tol", type=float, default=0.3)
    p.add_argument("--seeds", default="0-4")
    p.add_argument("--jobs", type=int, default=5)
    p.add_argument("--view", action="store_true", help="live viewer; runs only the first seed")
    args = p.parse_args()

    out = os.path.join("logs", "run_push",
                       f"{datetime.now():%Y%m%d_%H%M%S}_{args.planner}_goal{args.goal[0]:g}_{args.goal[1]:g}")
    os.makedirs(out)
    print("logging to", out)
    seeds = seed_list(args.seeds)
    if args.view:
        run(seeds[0], args, out)
    else:
        with Pool(args.jobs) as pool:
            pool.starmap(run, [(s, args, out) for s in seeds])
    plot(out)
