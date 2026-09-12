#!/usr/bin/env python3
"""Run the high-level MINLP contact scheduler above the unchanged MPPI in MuJoCo."""

import argparse
import os

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from whole_body_mppi.control.contact_scheduler import BoxPlanarState, MinlpContactScheduler
from whole_body_mppi.control.contact_scheduler.reference_adapter import pushing_reference
from whole_body_mppi.control.controllers.mppi_locomanipulation import MPPI_box_push
from whole_body_mppi.utils.tasks import get_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--task", default="push_box")
    parser.add_argument("--goal-x", type=float, default=None, help="Override the box goal x position.")
    parser.add_argument("--goal-y", type=float, default=None, help="Override the box goal y position.")
    # The "home" keyframe starts the box 1m ahead of the robot, which the
    # scheduler's short horizon can't reach in one step (reachable_distance =
    # horizon * scheduler_dt * robot_speed_max = 0.25m), so most runs burn
    # several seconds in the free-mode approach heuristic before any real
    # contact planning starts. Starting closer cuts that dead time.
    parser.add_argument("--box-x", type=float, default=0.6, help="Override the box's initial x position.")
    parser.add_argument("--box-y", type=float, default=0.0, help="Override the box's initial y position.")
    parser.add_argument("--save-data", default=None,
                         help="Log qpos/goal/body_ref trajectory to this .npz for later offline "
                              "rendering with render_box_push.py, instead of rendering live here "
                              "(live rendering competes with the solver for CPU/memory and has "
                              "caused OOM and multiprocessing-pipe crashes during long runs).")
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
    if args.goal_x is not None:
        mppi.x_box_ref[0] = args.goal_x
    if args.goal_y is not None:
        mppi.x_box_ref[1] = args.goal_y
    print(f"Goal: {mppi.x_box_ref[:2]}")
    if log is not None:
        log["goal_xy"] = mppi.x_box_ref[:2].copy()
    scheduler = MinlpContactScheduler()
    scheduler_steps = max(1, round(1.0 / (model.opt.timestep * scheduler.config["replan_rate_hz"])))
    steps = round(args.duration / model.opt.timestep)
    action = np.zeros(model.nu)

    for step in range(steps):
        if step % scheduler_steps == 0:
            yaw = Rotation.from_quat(data.qpos[3:7][[1, 2, 3, 0]]).as_euler("xyz")[2]
            schedule = scheduler.plan(
                BoxPlanarState(data.qpos[0], data.qpos[1], yaw, data.qvel[0], data.qvel[1], data.qvel[5]),
                data.qpos[7:9], mppi.x_box_ref[:2])
            reference = pushing_reference(schedule, scheduler.config)
            mppi.body_ref[:2] = reference.position
            mppi.body_ref[2] = 0.27
            mppi.body_ref[3:7] = [np.cos(reference.yaw / 2), 0.0, 0.0, np.sin(reference.yaw / 2)]
            mppi.goal_ori = mppi.body_ref[3:7].copy()
            # Pin this yaw so update()'s walk-toward-waypoint heuristic doesn't
            # overwrite it before the next replan (see MPPI_box_push.update()).
            mppi.external_ori = mppi.body_ref[3:7].copy()
            mppi.body_ref[7:9] = reference.velocity
            # The contact scheduler replaces the old task-level phase change
            # that selected walking after the initial in-place stance.
            mppi.gait_scheduler = mppi.gaits["walk"]
            print(f"t={data.time:.2f}s robot={data.qpos[7:9]} ref={reference.position} "
                  f"face={reference.face} status={schedule.status}")

        action = mppi.update(np.concatenate((data.qpos, data.qvel)))
        data.ctrl[:] = action
        mujoco.mj_step(model, data)

        if log is not None:
            log["time"].append(data.time)
            log["qpos"].append(data.qpos.copy())
            log["body_ref_xy"].append(mppi.body_ref[:2].copy())

    print("final box position:", data.qpos[:2])

    if log is not None:
        np.savez(args.save_data, time=np.array(log["time"]), qpos=np.array(log["qpos"]),
                 goal_xy=log["goal_xy"], body_ref_xy=np.array(log["body_ref_xy"]),
                 sim_path=log["sim_path"])
        print(f"Saved trajectory data to {args.save_data}")


if __name__ == "__main__":
    main()
