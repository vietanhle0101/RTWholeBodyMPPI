#!/usr/bin/env python3
"""Run the high-level MINLP contact scheduler above MPPI (with its scheduler-reference interface, e.g. external_ori) in MuJoCo."""

import argparse
import json
import os
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from whole_body_mppi.control.contact_scheduler import BoxPlanarState, MinlpContactScheduler
from whole_body_mppi.control.contact_scheduler.reference_adapter import pushing_reference
from whole_body_mppi.control.controllers.mppi_locomanipulation import MPPI_box_push
from whole_body_mppi.utils.tasks import get_task


def _yaw_from_quat(qpos_quat: np.ndarray) -> float:
    """MuJoCo quaternion order is [w,x,y,z]; scipy expects [x,y,z,w]."""
    return Rotation.from_quat(qpos_quat[[1, 2, 3, 0]]).as_euler("xyz")[2]


def _angle_delta(target: float, source: float) -> float:
    """Signed shortest yaw change from `source` to `target`."""
    return float(np.arctan2(np.sin(target - source), np.cos(target - source)))


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
    parser.add_argument("--goal-thresh", type=float, default=0.2, help="Box-to-goal distance counted as success.")
    parser.add_argument("--save-data", default=None,
                         help="Log qpos/goal/body_ref trajectory plus per-replan scheduler decisions and "
                              "run metadata to this .npz, for later offline rendering with "
                              "render_box_push.py and post-hoc analysis, instead of rendering live here "
                              "(live rendering competes with the solver for CPU/memory and has "
                              "caused OOM and multiprocessing-pipe crashes during long runs).")
    parser.add_argument("--scheduler-config", default=None,
                         help="Path to a MinlpContactScheduler config yml (default: "
                              "contact_scheduler/configs/push_box_minlp.yml).")
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
    initial_box_pose = data.qpos[:7].copy()

    log = None
    if args.save_data is not None:
        log = {"time": [], "qpos": [], "goal_xy": None, "body_ref_xy": [], "sim_path": sim_path,
               "replan_time": [], "replan_status": [], "replan_face": [], "replan_face_index": [],
               "replan_force": [], "replan_contact_location": [], "replan_yaw": [],
               "replan_target_yaw": [], "replan_yaw_step": [], "replan_robot_yaw": [],
               "replan_phase": [], "replan_committed_face_index": [], "replan_standoff_error": [],
               "replan_contact_hold_replans": [], "replan_release_violations": [],
               "replan_velocity": [], "replan_position": [], "replan_solve_time": []}

    mppi = MPPI_box_push(args.task)
    if args.goal_x is not None:
        mppi.x_box_ref[0] = args.goal_x
    if args.goal_y is not None:
        mppi.x_box_ref[1] = args.goal_y
    print(f"Goal: {mppi.x_box_ref[:2]}")
    if log is not None:
        log["goal_xy"] = mppi.x_box_ref[:2].copy()
    scheduler = MinlpContactScheduler(config_path=args.scheduler_config)
    scheduler_steps = max(1, round(1.0 / (model.opt.timestep * scheduler.config["replan_rate_hz"])))
    # MPPI's own rollout model is built at its configured dt (mppi.h, 0.01s /
    # 100Hz) and update() is called every physics step (0.002s / 500Hz) --
    # nominally a mismatch (the gait tables are named "*_100hz.tsv"). Tried
    # throttling update() to match; measured results were consistently worse
    # (e.g. fixed_diag45 0.118m->0.992m), likely because it also cuts the
    # robot's effective speed within a fixed sim duration. Left uncorrected
    # since it empirically performs better; worth revisiting later.
    steps = round(args.duration / model.opt.timestep)
    action = np.zeros(model.nu)

    status_counts = {"exact_success": 0, "incumbent": 0, "fallback": 0,
                     "free_mode_approach": 0, "align": 0}
    mode_switches = 0  # any change in face_index, including to/from free (-1)
    face_switches = 0  # only transitions between two distinct *active* faces
    prev_face_index = None
    cumulative_solver_time = 0.0
    # Begin the slew limiter at the actual robot yaw, not an arbitrary world
    # heading.  Robot free-joint quaternion follows the box free joint.
    last_commanded_yaw = _yaw_from_quat(data.qpos[10:14])
    yaw_steps = []

    for step in range(steps):
        if step % scheduler_steps == 0:
            yaw = _yaw_from_quat(data.qpos[3:7])
            robot_yaw = _yaw_from_quat(data.qpos[10:14])
            solve_start = time.perf_counter()
            schedule = scheduler.plan(
                BoxPlanarState(data.qpos[0], data.qpos[1], yaw, data.qvel[0], data.qvel[1], data.qvel[5]),
                data.qpos[7:9], mppi.x_box_ref[:2], robot_yaw=robot_yaw)
            diagnostics = scheduler.diagnostics
            solve_time = time.perf_counter() - solve_start
            cumulative_solver_time += solve_time
            reference = pushing_reference(schedule, scheduler.config, last_commanded_yaw)
            yaw_step = _angle_delta(reference.yaw, last_commanded_yaw)
            last_commanded_yaw = reference.yaw
            yaw_steps.append(yaw_step)
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
                  f"face={reference.face} yaw={reference.yaw:.2f}/{reference.target_yaw:.2f} "
                  f"phase={diagnostics['phase']} standoff={diagnostics['standoff_error']:.3f} "
                  f"status={schedule.status}")

            face_index = int(schedule.active_faces[0]) if len(schedule.active_faces) else -1
            if face_index >= 0:
                contact_location = float(schedule.contact_locations[0, face_index])
            else:
                contact_location = 0.0
            if schedule.status == "SUCCESS":
                status_counts["exact_success"] += 1
            elif schedule.status.startswith("incumbent"):
                status_counts["incumbent"] += 1
            elif schedule.status.startswith("fallback"):
                status_counts["fallback"] += 1
            elif schedule.status == "free-mode approach":
                status_counts["free_mode_approach"] += 1
            elif schedule.status == "align":
                status_counts["align"] += 1
            if prev_face_index is not None and face_index != prev_face_index:
                mode_switches += 1
                if prev_face_index >= 0 and face_index >= 0:
                    face_switches += 1
            prev_face_index = face_index

            if log is not None:
                log["replan_time"].append(data.time)
                log["replan_status"].append(schedule.status)
                log["replan_face"].append(reference.face)
                log["replan_face_index"].append(face_index)
                log["replan_force"].append(reference.force)
                log["replan_contact_location"].append(contact_location)
                log["replan_yaw"].append(reference.yaw)
                log["replan_target_yaw"].append(reference.target_yaw)
                log["replan_yaw_step"].append(yaw_step)
                log["replan_robot_yaw"].append(robot_yaw)
                log["replan_phase"].append(diagnostics["phase"])
                log["replan_committed_face_index"].append(diagnostics["committed_face_index"])
                log["replan_standoff_error"].append(diagnostics["standoff_error"])
                log["replan_contact_hold_replans"].append(diagnostics["contact_hold_replans"])
                log["replan_release_violations"].append(diagnostics["release_violations"])
                log["replan_velocity"].append(np.asarray(reference.velocity, dtype=float).copy())
                log["replan_position"].append(np.asarray(reference.position, dtype=float).copy())
                log["replan_solve_time"].append(solve_time)

        action = mppi.update(np.concatenate((data.qpos, data.qvel)))
        data.ctrl[:] = action
        mujoco.mj_step(model, data)

        if log is not None:
            log["time"].append(data.time)
            log["qpos"].append(data.qpos.copy())
            log["body_ref_xy"].append(mppi.body_ref[:2].copy())

        if np.linalg.norm(data.qpos[:2] - mppi.x_box_ref[:2]) < args.goal_thresh:
            print(f"t={data.time:.2f}s: box reached goal, stopping early")
            break

    final_box_xy = data.qpos[:2].copy()
    final_yaw = _yaw_from_quat(data.qpos[3:7])
    initial_distance = float(np.linalg.norm(initial_box_pose[:2] - mppi.x_box_ref[:2]))
    final_distance = float(np.linalg.norm(final_box_xy - mppi.x_box_ref[:2]))
    summary = {
        "initial_distance": initial_distance,
        "final_distance": final_distance,
        "distance_reduction": initial_distance - final_distance,
        "success": final_distance < args.goal_thresh,
        "final_box_xy": final_box_xy.tolist(),
        "final_yaw": final_yaw,
        "mode_switches": mode_switches,
        "face_switches": face_switches,
        "num_replans": sum(status_counts.values()),
        "status_counts": status_counts,
        "cumulative_solver_time_s": cumulative_solver_time,
        "max_commanded_yaw_step_rad": float(np.max(np.abs(yaw_steps))) if yaw_steps else 0.0,
    }

    print("final box position:", final_box_xy)
    print(f"distance to goal: {final_distance:.3f}m (from {initial_distance:.3f}m, "
          f"reduced {summary['distance_reduction']:.3f}m), success={summary['success']}")
    print(f"mode switches: {mode_switches} (face switches: {face_switches}), "
          f"replans: {summary['num_replans']} ({status_counts}), "
          f"cumulative solver time: {cumulative_solver_time:.1f}s")
    print(f"max commanded yaw step: {summary['max_commanded_yaw_step_rad']:.3f} rad")

    if log is not None:
        np.savez(
            args.save_data,
            time=np.array(log["time"]), qpos=np.array(log["qpos"]),
            goal_xy=log["goal_xy"], body_ref_xy=np.array(log["body_ref_xy"]), sim_path=log["sim_path"],
            # Per-replan scheduler decisions.
            replan_time=np.array(log["replan_time"]),
            replan_status=np.array(log["replan_status"], dtype=object),
            replan_face=np.array(log["replan_face"], dtype=object),
            replan_face_index=np.array(log["replan_face_index"]),
            replan_force=np.array(log["replan_force"]),
            replan_contact_location=np.array(log["replan_contact_location"]),
            replan_yaw=np.array(log["replan_yaw"]),
            replan_target_yaw=np.array(log["replan_target_yaw"]),
            replan_yaw_step=np.array(log["replan_yaw_step"]),
            replan_robot_yaw=np.array(log["replan_robot_yaw"]),
            replan_phase=np.array(log["replan_phase"], dtype=object),
            replan_committed_face_index=np.array(log["replan_committed_face_index"]),
            replan_standoff_error=np.array(log["replan_standoff_error"]),
            replan_contact_hold_replans=np.array(log["replan_contact_hold_replans"]),
            replan_release_violations=np.array(log["replan_release_violations"]),
            replan_velocity=np.array(log["replan_velocity"]),
            replan_position=np.array(log["replan_position"]),
            replan_solve_time=np.array(log["replan_solve_time"]),
            # Reproducibility metadata.
            scheduler_config_json=json.dumps(scheduler.config),
            task=args.task,
            cli_args_json=json.dumps(vars(args)),
            mujoco_version=mujoco.__version__,
            mppi_seed=mppi.seed,
            initial_box_pose=initial_box_pose,
            final_summary_json=json.dumps(summary),
        )
        print(f"Saved trajectory data to {args.save_data}")


if __name__ == "__main__":
    main()
