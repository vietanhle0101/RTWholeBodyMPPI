"""Unit tests for the non-MINLP approach/alignment staging policy."""

import math
import unittest

import numpy as np

from whole_body_mppi.control.contact_scheduler import BoxPlanarState, ContactSchedule, MinlpContactScheduler
from whole_body_mppi.control.contact_scheduler.reference_adapter import pushing_reference


def _scheduler() -> MinlpContactScheduler:
    """Build only the lightweight state required by staging-policy methods."""
    scheduler = MinlpContactScheduler.__new__(MinlpContactScheduler)
    scheduler.config = {
        "box_half_length": 0.19,
        "box_half_width": 0.19,
        "robot_standoff": 0.291,
        "robot_speed_max": 0.25,
    }
    scheduler.horizon = 5
    scheduler.dt = 0.2
    scheduler.replan_period = 0.2
    scheduler._phase = "approach"
    scheduler._committed_face_index = -1
    scheduler._previous_face_index = -1
    scheduler._last_solution = None
    scheduler._contact_hold_replans = 0
    scheduler._release_violations = 0
    scheduler._last_standoff_error = np.inf
    scheduler._min_contact_hold_replans = 0
    scheduler._contact_release_margin = 0.10
    scheduler._contact_release_replans = 1
    scheduler._bumper_forward_offset = 0.276
    scheduler._bumper_clearance = 0.015
    scheduler._align_yaw_tolerance = math.radians(12.5)
    scheduler._align_creep_speed = 0.05
    scheduler._approach_face_score_margin = 0.10
    scheduler._approach_reselect_cooldown_replans = 0
    scheduler._approach_reselect_cooldown = 0
    return scheduler


class ApproachPolicyTest(unittest.TestCase):
    def test_counterproductive_precontact_face_is_reselected(self):
        scheduler = _scheduler()
        scheduler._committed_face_index = 0  # rear pushes +x for a yaw-zero box
        schedule = scheduler._approach_if_needed(
            BoxPlanarState(0.0, 0.0, 0.0), np.array([0.0, -1.0]), np.array([-1.0, 0.0]), 0.0)

        self.assertIsNotNone(schedule)
        self.assertEqual(schedule.status, "free-mode approach")
        self.assertEqual(scheduler._committed_face_index, 2)  # right is the nearer neutral face

    def test_hold_release_clears_stale_face_before_selecting(self):
        scheduler = _scheduler()
        scheduler._phase = "hold"
        scheduler._committed_face_index = 0
        scheduler._contact_hold_replans = 1
        schedule = scheduler._approach_if_needed(
            BoxPlanarState(0.0, 0.0, 0.0), np.array([0.0, -1.0]), np.array([-1.0, 0.0]), 0.0)

        self.assertIsNotNone(schedule)
        self.assertEqual(schedule.status, "free-mode approach")
        self.assertEqual(scheduler._phase, "approach")
        self.assertEqual(scheduler._committed_face_index, 2)

    def test_aligned_position_but_wrong_yaw_returns_align_schedule(self):
        scheduler = _scheduler()
        box = BoxPlanarState(0.0, 0.0, 0.0)
        rear_target, rear_heading = scheduler._standoff_pose(box, 0)
        schedule = scheduler._approach_if_needed(box, rear_target, np.array([1.0, 0.0]), math.pi)

        self.assertIsNotNone(schedule)
        self.assertEqual(schedule.status, "align")
        self.assertAlmostEqual(schedule.desired_yaw, rear_heading)
        np.testing.assert_allclose(schedule.robot_velocities, 0.0)

    def test_motion_schedule_velocity_matches_position_steps(self):
        scheduler = _scheduler()
        schedule = scheduler._motion_schedule(
            BoxPlanarState(0.0, 0.0, 0.0), np.zeros(2), np.array([0.06, 0.0]), 0.25, "free-mode approach")
        np.testing.assert_allclose(schedule.robot_velocities,
                                   np.diff(schedule.robot_positions, axis=0) / scheduler.dt)
        np.testing.assert_allclose(schedule.robot_velocities[2:], 0.0)

    def test_align_schedule_preserves_heading_when_stationary(self):
        schedule = ContactSchedule(
            np.zeros((2, 6)), np.zeros((2, 2)), np.zeros((1, 2)), np.zeros((1, 3)),
            np.zeros((1, 3)), -np.ones(1, dtype=int), True, "align", desired_yaw=1.0)
        reference = pushing_reference(schedule, {"replan_rate_hz": 5.0, "reference_yaw_rate_max": 1.25})

        self.assertEqual(reference.face, "align")
        self.assertAlmostEqual(reference.target_yaw, 1.0)
        self.assertAlmostEqual(reference.yaw, 0.25)


if __name__ == "__main__":
    unittest.main()
