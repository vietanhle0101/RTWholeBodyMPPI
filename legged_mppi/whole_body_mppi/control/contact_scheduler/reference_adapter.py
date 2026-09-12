"""Conversion from a scheduled box contact into an MPPI body reference."""

import math
from typing import Mapping

import numpy as np

from .interfaces import ContactSchedule, Go1PushReference


FACE_NAMES = ("rear", "left", "right")


def face_geometry(half_length: float, half_width: float) -> Mapping[str, tuple]:
    """Return box-frame (midpoint, outward normal, tangent, half face length)."""
    return {
        "rear": (np.array([-half_length, 0.0]), np.array([-1.0, 0.0]), np.array([0.0, 1.0]), half_width),
        "left": (np.array([0.0, half_width]), np.array([0.0, 1.0]), np.array([-1.0, 0.0]), half_length),
        "right": (np.array([0.0, -half_width]), np.array([0.0, -1.0]), np.array([1.0, 0.0]), half_length),
    }


def pushing_reference(schedule: ContactSchedule, config: Mapping[str, float], last_yaw: float = 0.0) -> Go1PushReference:
    """Return the first future MPPI reference from a contact schedule.

    A free stage has no contact pose.  In that case the planned robot position is
    used and its yaw is inferred from its planned velocity, or held at `last_yaw`
    when that velocity is too small to give a meaningful direction (e.g. the
    free-mode approach heuristic has just reached its intermediate waypoint).
    Defaulting to a fixed 0.0 there instead of holding the last commanded yaw
    caused a real bug: whenever the planned velocity dipped near zero right
    after commanding a real (possibly very different) heading the step before,
    the reference would snap back to "face world +x", commanding a near-180
    degree spin for one replan cycle before flipping back -- visible in
    recorded runs as the robot suddenly spinning/stumbling mid-approach.
    """
    stage = 0
    face_index = int(schedule.face_indices[stage])
    robot_position = np.asarray(schedule.robot_positions[stage + 1], dtype=float)
    robot_velocity = np.asarray(schedule.robot_velocities[stage], dtype=float)
    if face_index < 0:
        yaw = math.atan2(robot_velocity[1], robot_velocity[0]) if np.linalg.norm(robot_velocity) > 1e-5 else last_yaw
        return Go1PushReference(robot_position, yaw, robot_velocity, "free", 0.0)

    face = FACE_NAMES[face_index]
    # The planned contact pose is already q_{k+1}; derive heading from the
    # corresponding box orientation and the selected outward normal.
    yaw_box = float(schedule.box_states[stage + 1, 2])
    _, outward_normal, _, _ = face_geometry(config["box_half_length"], config["box_half_width"])[face]
    direction = -np.array([
        math.cos(yaw_box) * outward_normal[0] - math.sin(yaw_box) * outward_normal[1],
        math.sin(yaw_box) * outward_normal[0] + math.cos(yaw_box) * outward_normal[1],
    ])
    yaw = math.atan2(direction[1], direction[0])
    return Go1PushReference(robot_position, yaw, robot_velocity, face, float(schedule.forces[stage, face_index]))
