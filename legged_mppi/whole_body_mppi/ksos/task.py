"""Global-MPPI task for the reduced Go1 box push (go1_push_trunk.xml).

State: qpos = [box free joint 0:7, trunk x, y, yaw 7:10]. Controls: forward, lateral (pinned to 0), turn.
The goal is the `goal` mocap body, read from the state, so it can change without re-jitting.
"""
import os

import jax.numpy as jnp
import mujoco
from global_mppi.task_base import Task
from mujoco import mjx

XML = os.path.join(os.path.dirname(__file__), "go1_push_trunk.xml")
CONTACT = 0.47  # trunk center -> box center while pushing (measured on the Go1)


class Go1PushTask(Task):
    def __init__(self, w_box=10.0, w_near=2.0, w_rate=1e-3, w_terminal=5.0):
        super().__init__(mujoco.MjModel.from_xml_path(XML), trace_sites=["trunk"])
        # the lateral actuator is a damper with no command: do not sample it
        self.u_min = self.u_min.at[1].set(0.0)
        self.u_max = self.u_max.at[1].set(0.0)
        self.w_box, self.w_near, self.w_rate, self.w_terminal = w_box, w_near, w_rate, w_terminal

    def running_cost(self, state: mjx.Data, control):
        box, trunk, goal = state.qpos[0:2], state.qpos[7:9], state.mocap_pos[0, :2]
        box_cost = jnp.sum(jnp.square(box - goal))
        # pull the trunk towards the box until it touches it (shaping only, no push direction)
        gap = jnp.maximum(jnp.linalg.norm(trunk - box) - CONTACT, 0.0)
        return self.w_box * box_cost + self.w_near * gap ** 2

    def control_rate_cost(self, du):
        return self.w_rate / self.dt * jnp.sum(jnp.square(du))

    def terminal_cost(self, state: mjx.Data):
        return self.w_terminal * self.running_cost(state, jnp.zeros(self.model.nu))
