"""Solver-independent data exchanged by high-level contact schedulers."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class BoxPlanarState:
    """Measured planar state of the pushed box in the world frame."""

    x: float
    y: float
    yaw: float
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0

    def vector(self) -> np.ndarray:
        return np.array([self.x, self.y, self.yaw, self.vx, self.vy, self.yaw_rate], dtype=float)


@dataclass(frozen=True)
class Go1PushReference:
    """The planar reference handed from the scheduler to whole-body MPPI."""

    position: np.ndarray
    yaw: float
    velocity: np.ndarray
    face: str
    force: float
    # Face geometry supplies this unsmoothed heading.  `yaw` is the bounded
    # command actually handed to MPPI, so logging both distinguishes a
    # deliberate rate limit from a scheduler mode change.
    target_yaw: float = 0.0


@dataclass(frozen=True)
class ContactSchedule:
    """A solved high-level schedule; index zero is the measured state."""

    box_states: np.ndarray
    robot_positions: np.ndarray
    robot_velocities: np.ndarray
    forces: np.ndarray
    contact_locations: np.ndarray
    face_indices: np.ndarray
    success: bool
    status: str

    @property
    def active_faces(self) -> Tuple[int, ...]:
        return tuple(int(face) for face in self.face_indices)
