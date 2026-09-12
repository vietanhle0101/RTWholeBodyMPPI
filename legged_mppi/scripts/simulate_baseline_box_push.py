#!/usr/bin/env python3
"""Baseline box-push run: the original MPPI_box_push goal-following logic,
with no MINLP contact scheduler. Mirrors simulate_minlp_box_push.py's CLI so
the two can be compared on identical box-start/goal/duration settings.

The baseline has no notion of contact faces: once past its initial in-place
stance it just walks the robot's body reference to the box's own position
every tick (MPPI_box_push.next_goal()'s "follow_box" branch), which only
produces useful pushing when the robot's approach direction already happens
to line up with the direction the box needs to move.
"""

import argparse
import os

import mujoco
import numpy as np

from whole_body_mppi.control.controllers.mppi_locomanipulation import MPPI_box_push
from whole_body_mppi.utils.tasks import get_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--task", default="push_box")
    parser.add_argument("--goal-x", type=float, default=None, help="Override the box goal x position.")
    parser.add_argument("--goal-y", type=float, default=None, help="Override the box goal y position.")
    parser.add_argument("--box-x", type=float, default=0.6, help="Override the box's initial x position.")
    parser.add_argument("--box-y", type=float, default=0.0, help="Override the box's initial y position.")
    parser.add_argument("--goal-thresh", type=float, default=0.2, help="Box-to-goal distance counted as success.")
    parser.add_argument("--save-data", default=None,
                         help="Log qpos/goal/body_ref trajectory to this .npz, in the same schema "
                              "simulate_minlp_box_push.py uses, so render_box_push.py can render "
                              "either one for a like-for-like video comparison.")
    args = parser.parse_args()

    task = get_task(args.task)
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sim_path = os.path.join(package_dir, "whole_body_mppi", task["sim_path"])
    model = mujoco.MjModel.from_xml_path(sim_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[0] = args.box_x
    data.qpos[1] = args.box_y
    mujoco.mj_forward(model, data)

    log = None
    if args.save_data is not None:
        log = {"time": [], "qpos": [], "goal_xy": None, "body_ref_xy": [], "sim_path": sim_path}

    mppi = MPPI_box_push(args.task)
    mppi.internal_ref = True
    if args.goal_x is not None:
        mppi.x_box_ref[0] = args.goal_x
    if args.goal_y is not None:
        mppi.x_box_ref[1] = args.goal_y
    print(f"Goal: {mppi.x_box_ref[:2]}")
    if log is not None:
        log["goal_xy"] = mppi.x_box_ref[:2].copy()

    steps = round(args.duration / model.opt.timestep)
    reached_at = None

    for step in range(steps):
        error = np.linalg.norm(np.array(mppi.body_ref[:3]) - np.array(data.qpos[7:10]))
        if error < 0.1:
            mppi.next_goal()

        action = mppi.update(np.concatenate((data.qpos, data.qvel)))
        data.ctrl[:] = action
        mujoco.mj_step(model, data)

        box_to_goal = np.linalg.norm(data.qpos[:2] - mppi.x_box_ref[:2])
        if reached_at is None and box_to_goal < args.goal_thresh:
            reached_at = data.time
            print(f"t={data.time:.2f}s: box reached goal (distance {box_to_goal:.3f}m)")

        if step % 100 == 0:
            print(f"t={data.time:.2f}s box={data.qpos[:2]} robot={data.qpos[7:9]} "
                  f"body_ref={mppi.body_ref[:2]} follow_box={mppi.follow_box}")

        if log is not None:
            log["time"].append(data.time)
            log["qpos"].append(data.qpos.copy())
            log["body_ref_xy"].append(mppi.body_ref[:2].copy())

        if reached_at is not None:
            break

    final_box = data.qpos[:2].copy()
    final_distance = float(np.linalg.norm(final_box - mppi.x_box_ref[:2]))
    print(f"final box position: {final_box}, distance to goal: {final_distance:.3f}m")
    print(f"reached goal: {reached_at is not None}" + (f" at t={reached_at:.2f}s" if reached_at else ""))

    if log is not None:
        np.savez(args.save_data, time=np.array(log["time"]), qpos=np.array(log["qpos"]),
                 goal_xy=log["goal_xy"], body_ref_xy=np.array(log["body_ref_xy"]),
                 sim_path=log["sim_path"])
        print(f"Saved trajectory data to {args.save_data}")


if __name__ == "__main__":
    main()
