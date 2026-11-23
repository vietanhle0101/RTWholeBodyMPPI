"""Global-MPPI task for the reduced Go1 box push (go1_push_trunk.xml).

State: qpos = [box free joint 0:7, trunk x, y, yaw 7:10]. Controls: forward, lateral (pinned to 0), turn.
The goal is the `goal` mocap body (position, and yaw with w_yaw), read from the state, so it can change
without re-jitting.
"""
import os

import jax.numpy as jnp
import mujoco
from global_mppi.task_base import Task
from mujoco import mjx

XML = os.path.join(os.path.dirname(__file__), "go1_push_trunk.xml")
CONTACT = 0.47  # trunk center -> box center while pushing (measured on the Go1)


MAX_OFFSET = 0.12  # m, push point off the box's centre line when turning it
NEAR = 0.2  # m: with a yaw goal, closer than this the push direction blends into the trunk's own


def quat_yaw(q):
    w, x, y, z = q
    return jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def cube_yaw_error(yaw, goal_yaw):
    """Box yaw minus the goal's, in (-pi/4, pi/4]: the cube looks the same every quarter turn."""
    return 0.25 * jnp.arctan2(jnp.sin(4 * (yaw - goal_yaw)), jnp.cos(4 * (yaw - goal_yaw)))


class Go1PushTask(Task):
    def __init__(self, w_box=10.0, w_push=2.0, w_rate=1e-3, w_terminal=5.0, w_yaw=0.0, k_turn=0.0):
        """w_yaw: weight on the squared box yaw error to the goal mocap's yaw (0: position goal only).
        k_turn [m/rad]: the push point moves this far sideways per radian of yaw error (up to MAX_OFFSET),
        so the push turns the box towards the goal's yaw."""
        super().__init__(mujoco.MjModel.from_xml_path(XML), trace_sites=["trunk"])
        # the lateral actuator is a damper with no command: do not sample it
        self.u_min = self.u_min.at[1].set(0.0)
        self.u_max = self.u_max.at[1].set(0.0)
        self.w_box, self.w_push, self.w_rate, self.w_terminal = w_box, w_push, w_rate, w_terminal
        self.w_yaw, self.k_turn = w_yaw, k_turn

    def running_cost(self, state: mjx.Data, control):
        box, trunk, goal = state.qpos[0:2], state.qpos[7:9], state.mocap_pos[0, :2]
        to_goal = goal - box
        dist = jnp.linalg.norm(to_goal)
        d = to_goal / (dist + 1e-6)
        if self.w_yaw:  # at the goal the direction to it is undefined: push from where the trunk is instead
            w = jnp.minimum(dist / NEAR, 1.0)
            d = w * d + (1 - w) * (box - trunk) / (jnp.linalg.norm(box - trunk) + 1e-6)
            d = d / (jnp.linalg.norm(d) + 1e-6)
        yaw_err = cube_yaw_error(quat_yaw(state.qpos[3:7]), quat_yaw(state.mocap_quat[0]))
        # pull the trunk to the pushing point behind the box. Only touching it (from any side) left the
        # trunk stuck at the side of the box near the goal: walking around it does not fit the horizon.
        # Pushing at s to the left of the centre line turns the box clockwise, so s = k_turn * yaw error.
        s = jnp.clip(self.k_turn * yaw_err, -MAX_OFFSET, MAX_OFFSET)
        push_point = box - CONTACT * d + s * jnp.array([-d[1], d[0]])
        return (self.w_box * jnp.sum(jnp.square(to_goal)) + self.w_push * jnp.sum(jnp.square(trunk - push_point))
                + self.w_yaw * jnp.square(yaw_err))

    def control_rate_cost(self, du):
        return self.w_rate / self.dt * jnp.sum(jnp.square(du))

    def terminal_cost(self, state: mjx.Data):
        return self.w_terminal * self.running_cost(state, jnp.zeros(self.model.nu))
