"""Push-box demo: the loop from notebooks/MPPI_tasks_push.ipynb as a script.

    .venv/bin/python legged_mppi/scripts/push_box_demo.py                    # notebook goal (2, 1)
    .venv/bin/python legged_mppi/scripts/push_box_demo.py --goal 2 -0.75     # paper's sideways case
    .venv/bin/python legged_mppi/scripts/push_box_demo.py --headless --plot push_box.png

The box starts at (1, 0) and the robot at the origin. Magenta marker: robot body reference,
yellow marker: box goal.
"""
import argparse
import os

import mujoco
import numpy as np

from whole_body_mppi.control.controllers.mppi_locomanipulation import MPPI_box_push
from whole_body_mppi.utils.tasks import get_task

PKG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../whole_body_mppi")


def add_markers(viewer, ctrl):
    for pos, rgba in ((ctrl.body_ref[:3], [1, 0, 1, 1]), (ctrl.x_box_ref[:3], [1, 1, 0, 1])):
        viewer.add_marker(pos=pos * 1, size=[0.15, 0.15, 0.15], rgba=rgba,
                          type=mujoco.mjtGeom.mjGEOM_SPHERE, label="")


def plot(path, box, robot, goal):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(box[:, 0], box[:, 1], label="box")
    ax.plot(robot[:, 0], robot[:, 1], label="robot trunk")
    ax.scatter(*box[0], c="k", zorder=3, label="box start")
    ax.add_patch(plt.Circle(goal, 0.3, fill=False, ls="--", color="tab:green", label="goal (0.3 m)"))
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("saved", path)


def main(args):
    ctrl = MPPI_box_push(task="push_box")
    ctrl.internal_ref = True
    ctrl.reset_planner()
    if args.seed is not None:
        ctrl.random_generator = np.random.default_rng(args.seed)
    ctrl.x_box_ref[:2] = args.goal
    goal = np.array(args.goal)

    model = mujoco.MjModel.from_xml_path(os.path.join(PKG_DIR, get_task("push_box")["sim_path"]))
    model.opt.timestep = 0.01
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    viewer = None
    if not args.headless:
        import mujoco_viewer
        viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=True)
        viewer.cam.distance = 3.5
        viewer.cam.lookat[:] = [1.5, 0, 0]

    box, robot = [], []
    t_goal = None
    for k in range(int(args.time / 0.01)):
        data.ctrl[:] = ctrl.update(np.concatenate([data.qpos, data.qvel]))
        mujoco.mj_step(model, data)
        if np.linalg.norm(ctrl.body_ref[:3] - data.qpos[7:10]) < 0.1:
            ctrl.next_goal()

        box.append(data.qpos[:2].copy())
        robot.append(data.qpos[7:9].copy())
        err = np.linalg.norm(data.qpos[:2] - goal)
        if t_goal is None and err < 0.3:
            t_goal = data.time
            print(f"t={data.time:5.2f}s  box within 0.3 m of goal")
        if k % 100 == 0:
            print(f"t={data.time:5.2f}s  box=({data.qpos[0]:.2f}, {data.qpos[1]:.2f})  goal err={err:.2f} m  "
                  f"trunk z={data.qpos[9]:.2f}  phase={ctrl.goal_index}")

        if viewer is not None:
            if not viewer.is_alive:
                break
            if k % 2 == 0:
                add_markers(viewer, ctrl)
                viewer.render()

    err = np.linalg.norm(data.qpos[:2] - goal)
    print(f"done: {'success at %.2f s' % t_goal if t_goal else 'goal not reached'}, final box error {err:.3f} m")
    if viewer is not None and viewer.is_alive:
        viewer.close()
    if args.plot:
        plot(args.plot, np.array(box), np.array(robot), goal)
    ctrl.shutdown()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--goal", type=float, nargs=2, default=[2.0, 1.0], help="box goal x y (notebook: 2 1)")
    p.add_argument("--time", type=float, default=26.0, help="simulated seconds (notebook: 26)")
    p.add_argument("--seed", type=int, default=None, help="MPPI sampling seed (default: config seed 42)")
    p.add_argument("--headless", action="store_true", help="no viewer window")
    p.add_argument("--plot", help="save box/robot xy trajectory plot to this path")
    main(p.parse_args())
