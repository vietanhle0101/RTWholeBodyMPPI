import os
import yaml
import numpy as np
import mujoco
import jax
import jax.numpy as jnp

from whole_body_mppi.utils.tasks import get_task
from whole_body_mppi.control.controllers.base_controller import BaseMPPI
from whole_body_mppi.control.gait_scheduler.scheduler import GaitScheduler, Timer
from whole_body_mppi.utils.transforms import calculate_orientation_quaternion


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GAIT_DIR = os.path.join(BASE_DIR, "../gait_scheduler/gaits/")

GAIT_INPLACE_PATH = os.path.join(
    GAIT_DIR, "FAST/walking_gait_raibert_FAST_0_0_10cm_100hz.tsv"
)
GAIT_TROT_PATH = os.path.join(
    GAIT_DIR, "MED/walking_gait_raibert_MED_0_5_15cm_100hz.tsv"
)
GAIT_WALK_PATH = os.path.join(
    GAIT_DIR, "MED/walking_gait_raibert_MED_0_1_10cm_100hz.tsv"
)
GAIT_WALK_FAST_PATH = os.path.join(
    GAIT_DIR, "FAST/walking_gait_raibert_FAST_0_1_10cm_100hz.tsv"
)

GRAVITY = jnp.array([0.0, 0.0, -9.81], dtype=jnp.float32)


def quat_normalize_jax(q):
    return q / jnp.maximum(jnp.linalg.norm(q), 1e-6)


def quat_distance_jax(q1, q2):
    return 1.0 - jnp.abs(jnp.sum(q1 * q2, axis=-1))


def quat_multiply_jax(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return jnp.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=q1.dtype,
    )


def quat_to_rotmat_jax(q):
    q = quat_normalize_jax(q)
    w, x, y, z = q
    return jnp.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=q.dtype,
    )


def integrate_quat_world_jax(q, omega_world, dt):
    omega_quat = jnp.array([0.0, omega_world[0], omega_world[1], omega_world[2]], dtype=q.dtype)
    q_dot = 0.5 * quat_multiply_jax(omega_quat, q)
    return quat_normalize_jax(q + dt * q_dot)


def srbm_step_jax(state, forces, contacts, foot_positions_body, mass, inertia_body, dt):
    pos = state[0:3]
    quat = quat_normalize_jax(state[3:7])
    lin_vel = state[7:10]
    ang_vel = state[10:13]

    rot = quat_to_rotmat_jax(quat)
    foot_world = jnp.einsum("ij,lj->li", rot, foot_positions_body)
    masked_forces = forces * contacts[:, None]

    total_force = jnp.sum(masked_forces, axis=0)
    torque_world = jnp.sum(jnp.cross(foot_world, masked_forces), axis=0)

    inertia_world = rot @ jnp.diag(inertia_body) @ rot.T
    ang_momentum = inertia_world @ ang_vel
    ang_acc = jnp.linalg.solve(
        inertia_world,
        torque_world - jnp.cross(ang_vel, ang_momentum),
    )
    lin_acc = total_force / mass + GRAVITY

    next_pos = pos + dt * lin_vel
    next_lin_vel = lin_vel + dt * lin_acc
    next_ang_vel = ang_vel + dt * ang_acc
    next_quat = integrate_quat_world_jax(quat, next_ang_vel, dt)

    return jnp.concatenate([next_pos, next_quat, next_lin_vel, next_ang_vel], axis=0)


def rollout_srbm_jax(initial_state, force_traj, contact_schedule, foot_positions_body, mass, inertia_body, dt):
    def scan_step(carry, inputs):
        force_k, contact_k = inputs
        next_state = srbm_step_jax(
            carry,
            force_k.reshape(4, 3),
            contact_k,
            foot_positions_body,
            mass,
            inertia_body,
            dt,
        )
        return next_state, next_state

    _, states = jax.lax.scan(scan_step, initial_state, (force_traj, contact_schedule))
    return states


def rollout_cost_jax(
    initial_state,
    force_traj,
    contact_schedule,
    body_ref,
    nominal_force_traj,
    foot_positions_body,
    mass,
    inertia_body,
    dt,
):
    states = rollout_srbm_jax(
        initial_state,
        force_traj,
        contact_schedule,
        foot_positions_body,
        mass,
        inertia_body,
        dt,
    )

    pos = states[:, 0:3]
    quat = states[:, 3:7]
    lin_vel = states[:, 7:10]
    ang_vel = states[:, 10:13]

    ref_pos = body_ref[0:3]
    ref_quat = body_ref[3:7]
    ref_lin_vel = body_ref[7:10]

    pos_err = pos - ref_pos[None, :]
    vel_err = lin_vel - ref_lin_vel[None, :]
    quat_err = quat_distance_jax(quat, jnp.repeat(ref_quat[None, :], quat.shape[0], axis=0))

    stance_counts = jnp.maximum(jnp.sum(contact_schedule, axis=1, keepdims=True), 1.0)
    forces_reshaped = force_traj.reshape(force_traj.shape[0], 4, 3)
    swing_forces = forces_reshaped * (1.0 - contact_schedule[:, :, None])

    running_cost = (
        4.0 * jnp.sum(pos_err ** 2, axis=1)
        + 2.0 * jnp.sum(vel_err ** 2, axis=1)
        + 0.3 * jnp.sum(ang_vel ** 2, axis=1)
        + 6.0 * quat_err
        + 2.0e-4 * jnp.sum((force_traj - nominal_force_traj) ** 2, axis=1)
    )
    smoothness_cost = 5.0e-4 * jnp.sum((force_traj[1:] - force_traj[:-1]) ** 2)
    swing_cost = 5.0e-3 * jnp.sum(swing_forces ** 2)
    negative_fz_cost = 1.0e-2 * jnp.sum((forces_reshaped[:, :, 2] < -1e-3).astype(jnp.float32))
    force_balance_cost = 5.0e-2 * jnp.sum((forces_reshaped[:, :, 2] / stance_counts) ** 2)
    terminal_cost = (
        10.0 * jnp.sum((states[-1, 0:3] - ref_pos) ** 2)
        + 4.0 * jnp.sum((states[-1, 7:10] - ref_lin_vel) ** 2)
    )

    return running_cost.sum() + smoothness_cost + swing_cost + negative_fz_cost + force_balance_cost + terminal_cost, states


batched_rollout_cost_jax = jax.jit(
    jax.vmap(
        rollout_cost_jax,
        in_axes=(None, 0, None, None, None, None, None, None, None),
        out_axes=(0, 0),
    )
)


class SRBM_MPPI(BaseMPPI):
    """
    Reduced-order MPPI controller:
    - internal optimizer: JAX SRBM rollouts over per-foot GRFs
    - simulator interface: unchanged (`update`, `next_goal`, `eval_best_trajectory`)
    - low-level output: joint targets obtained from gait reference plus SRBM force residuals
    """

    def __init__(self, task="walk_straight"):
        self.task = task
        self.task_data = get_task(task)

        self.goal_pos = self.task_data["goal_pos"]
        self.goal_ori = self.task_data["default_orientation"]
        self.cmd_vel = self.task_data["cmd_vel"]
        self.goal_thresh = self.task_data["goal_thresh"]
        self.desired_gait = self.task_data["desired_gait"]
        self.waiting_times = self.task_data["waiting_times"]

        config_path = os.path.join(BASE_DIR, self.task_data["config_path"])
        model_path = os.path.join(BASE_DIR, "../..", self.task_data["model_path"])

        super().__init__(model_path, config_path)

        with open(config_path, "r") as f:
            params = yaml.safe_load(f)

        self.Q = np.diag(np.array(params["Q_diag"]))
        self.R = np.diag(np.array(params["R_diag"]))

        self.gaits = {
            "in_place": GaitScheduler(gait_path=GAIT_INPLACE_PATH, name="in_place"),
            "trot": GaitScheduler(gait_path=GAIT_TROT_PATH, name="trot"),
            "walk": GaitScheduler(gait_path=GAIT_WALK_PATH, name="walk"),
            "walk_fast": GaitScheduler(gait_path=GAIT_WALK_FAST_PATH, name="walk_fast"),
        }

        self.goal_index = 0
        self.timer = Timer(end_time=self.waiting_times[0])
        self.gait_scheduler = self.gaits[self.desired_gait[self.goal_index]]

        self.body_ref = np.concatenate(
            (
                self.goal_pos[self.goal_index],
                self.goal_ori[self.goal_index],
                self.cmd_vel[self.goal_index],
                np.zeros(4),
            )
        )

        self.obs = None
        self.internal_ref = True
        self.task_success = False
        self.exp_weights = np.ones(self.n_samples) / self.n_samples
        self.best_srbm_cost = None
        self.best_srbm_rollout = None
        self.selected_srbm_trajectory = None

        trunk_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.robot_mass = float(np.sum(self.model.body_mass[1:]))
        # Approximate reduced-order inertia around the trunk/CoM.
        self.srbm_inertia = np.array([0.24, 0.80, 0.90], dtype=np.float32)
        if trunk_body_id >= 0:
            trunk_inertia = np.asarray(self.model.body_inertia[trunk_body_id], dtype=np.float32)
            self.srbm_inertia = np.maximum(self.srbm_inertia, 2.5 * trunk_inertia)

        self.foot_positions_body = np.array(
            [
                [0.19, -0.11, -0.28],  # FR
                [0.19, 0.11, -0.28],   # FL
                [-0.19, -0.11, -0.28], # RR
                [-0.19, 0.11, -0.28],  # RL
            ],
            dtype=np.float32,
        )

        self.force_sigma = np.array([25.0, 20.0, 35.0] * 4, dtype=np.float32)
        self.force_bounds_min = np.array([-80.0, -80.0, 0.0] * 4, dtype=np.float32)
        self.force_bounds_max = np.array([80.0, 80.0, 180.0] * 4, dtype=np.float32)
        self.joint_force_gains = np.array(
            [
                [0.0015, -0.0020, -0.0025],
                [0.0015, -0.0020, -0.0025],
                [0.0015, -0.0020, -0.0025],
                [0.0015, -0.0020, -0.0025],
            ],
            dtype=np.float32,
        )

        self.reset_planner()
        self._initialized_from_gait = False

        print(f"[SRBM_MPPI] task={task}")
        print(
            f"[SRBM_MPPI] horizon={self.horizon}, n_samples={self.n_samples}, "
            f"force_dim=12, joint_dim={self.act_dim}"
        )

    def reset_planner(self):
        super().reset_planner()
        self.srbm_trajectory = np.zeros((self.horizon, 12), dtype=np.float32)

    def obs_to_srbm_state(self, obs):
        pos = obs[0:3]
        quat = obs[3:7]
        lin_vel = obs[19:22]
        ang_vel = obs[22:25]
        quat = quat / max(np.linalg.norm(quat), 1e-8)
        return np.concatenate([pos, quat, lin_vel, ang_vel]).astype(np.float32)

    def _update_goal_orientation(self, obs):
        direction = self.body_ref[:3] - obs[:3]
        goal_delta = np.linalg.norm(direction)

        if goal_delta > 0.1 and not self.timer.waiting:
            goal_ori = calculate_orientation_quaternion(obs[:3], self.body_ref[:3])
        else:
            goal_ori = np.array([1, 0, 0, 0])

        self.body_ref[3:7] = goal_ori
        return direction

    def _maybe_initialize_trajectory_from_gait(self):
        if self.internal_ref:
            self.joints_ref = self.gait_scheduler.gait[:, self.gait_scheduler.indices[: self.horizon]]

        if not self._initialized_from_gait:
            jr = np.transpose(self.joints_ref, (1, 0))[:, :12]
            self.trajectory = jr.copy()
            self.selected_trajectory = jr.copy()
            self._initialized_from_gait = True

    def _build_contact_schedule(self):
        gait_name = self.desired_gait[self.goal_index]

        if gait_name == "trot":
            phase_pattern = np.array(
                [
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            )
        elif gait_name in ("walk", "walk_fast"):
            phase_pattern = np.array(
                [
                    [1.0, 1.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0, 1.0],
                ],
                dtype=np.float32,
            )
        else:
            phase_pattern = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)

        phase_ids = self.gait_scheduler.indices[: self.horizon] % phase_pattern.shape[0]
        return phase_pattern[phase_ids]

    def _build_nominal_force_trajectory(self, contact_schedule):
        nominal = np.zeros((self.horizon, 4, 3), dtype=np.float32)
        for k in range(self.horizon):
            stance = contact_schedule[k]
            n_stance = max(int(np.sum(stance)), 1)
            nominal_fz = self.robot_mass * 9.81 / n_stance
            nominal[k, :, 2] = stance * nominal_fz
        return nominal.reshape(self.horizon, 12)

    def _sample_force_trajectories(self, nominal_force_traj):
        noise = self.random_generator.normal(
            size=(self.n_samples, self.horizon, 12)
        ).astype(np.float32) * self.force_sigma
        actions = self.srbm_trajectory[None, :, :] + noise
        actions = 0.6 * actions + 0.4 * nominal_force_traj[None, :, :]
        actions = np.clip(actions, self.force_bounds_min, self.force_bounds_max)
        return actions

    def _joint_targets_from_forces(self, joint_ref_traj, force_traj, nominal_force_traj, contact_schedule):
        force_delta = (force_traj - nominal_force_traj).reshape(self.horizon, 4, 3)
        force_delta = np.clip(force_delta, -60.0, 60.0)

        joint_ref = joint_ref_traj.copy().reshape(self.horizon, 4, 3)
        joint_delta = force_delta * self.joint_force_gains[None, :, :]
        joint_delta *= contact_schedule[:, :, None]

        joint_plan = joint_ref + joint_delta
        joint_plan = joint_plan.reshape(self.horizon, 12)
        return np.clip(joint_plan, self.act_min, self.act_max)

    def update(self, obs):
        self.obs = obs
        direction = self._update_goal_orientation(obs)
        self._maybe_initialize_trajectory_from_gait()

        contact_schedule = self._build_contact_schedule()
        nominal_force_traj = self._build_nominal_force_trajectory(contact_schedule)
        force_trajs = self._sample_force_trajectories(nominal_force_traj)

        initial_state = self.obs_to_srbm_state(obs)

        costs_sum, state_rollouts = batched_rollout_cost_jax(
            jnp.asarray(initial_state),
            jnp.asarray(force_trajs),
            jnp.asarray(contact_schedule),
            jnp.asarray(self.body_ref[:10], dtype=jnp.float32),
            jnp.asarray(nominal_force_traj),
            jnp.asarray(self.foot_positions_body),
            jnp.asarray(self.robot_mass, dtype=jnp.float32),
            jnp.asarray(self.srbm_inertia),
            jnp.asarray(self.h, dtype=jnp.float32),
        )

        costs_sum = np.asarray(costs_sum)
        state_rollouts = np.asarray(state_rollouts)

        self.gait_scheduler.roll()

        min_cost = np.min(costs_sum)
        max_cost = np.max(costs_sum)
        denom = max(max_cost - min_cost, 1e-8)

        self.exp_weights = np.exp(
            -(costs_sum - min_cost) / (self.temperature * denom)
        )

        weighted_forces = self.exp_weights[:, None, None] * force_trajs
        updated_force_traj = np.sum(weighted_forces, axis=0) / (
            np.sum(self.exp_weights) + 1e-10
        )
        updated_force_traj = np.clip(updated_force_traj, self.force_bounds_min, self.force_bounds_max)

        joint_ref_traj = np.transpose(self.joints_ref, (1, 0))[:, :12]
        updated_joint_traj = self._joint_targets_from_forces(
            joint_ref_traj,
            updated_force_traj,
            nominal_force_traj,
            contact_schedule,
        )

        best_idx = int(np.argmin(costs_sum))
        self.best_srbm_cost = float(costs_sum[best_idx])
        self.best_srbm_rollout = state_rollouts[best_idx]

        self.selected_srbm_trajectory = updated_force_traj
        self.srbm_trajectory = np.roll(updated_force_traj, shift=-1, axis=0)
        self.srbm_trajectory[-1] = nominal_force_traj[-1]

        self.selected_trajectory = updated_joint_traj
        self.trajectory = np.roll(updated_joint_traj, shift=-1, axis=0)
        self.trajectory[-1] = updated_joint_traj[-1]

        if np.linalg.norm(direction) >= self.goal_thresh[self.goal_index]:
            self.timer.waiting = False

        action = updated_joint_traj[0]

        if not hasattr(self, "_printed_debug_once"):
            print("[SRBM_MPPI] srbm_state:", initial_state)
            print("[SRBM_MPPI] contact_schedule shape:", contact_schedule.shape)
            print("[SRBM_MPPI] sampled_force_trajs shape:", force_trajs.shape)
            print("[SRBM_MPPI] updated_joint_traj shape:", updated_joint_traj.shape)
            self._printed_debug_once = True

        return action

    def next_goal(self):
        self.timer.increment()

        if self.goal_index < len(self.goal_pos) - 1 and self.timer.done:
            self.goal_index += 1
            self.body_ref[:3] = self.goal_pos[self.goal_index]
            self.body_ref[7:9] = self.cmd_vel[self.goal_index]
            self.gait_scheduler = self.gaits[self.desired_gait[self.goal_index]]

            self.timer.reset()
            self.timer.end_time = self.waiting_times[self.goal_index]
            self.timer.waiting = False

            print(f"Moved to next goal {self.goal_index}: {self.goal_pos[self.goal_index]}")
            print(f"Gait: {self.desired_gait[self.goal_index]}")

        elif self.goal_index == len(self.goal_pos) - 1 and not self.task_success and self.timer.done:
            print("Task succeeded.")
            self.task_success = True

        else:
            self.timer.waiting = True

    def eval_best_trajectory(self):
        if self.obs is None:
            return None
        return self.best_srbm_cost

    def __del__(self):
        if hasattr(self, "executor"):
            self.shutdown()
