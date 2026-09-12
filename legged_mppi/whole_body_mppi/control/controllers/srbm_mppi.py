import os
import yaml
import numpy as np
import mujoco
import jax
import jax.numpy as jnp

from whole_body_mppi.utils.tasks import get_task
from whole_body_mppi.control.controllers.base_controller import BaseMPPI
from whole_body_mppi.control.controllers.srbm import SRBM
from whole_body_mppi.control.controllers.mppi_locomotion import MPPI
from whole_body_mppi.control.gait_scheduler.scheduler import GaitScheduler, Timer
from whole_body_mppi.utils.transforms import calculate_orientation_quaternion
from whole_body_mppi.control.controllers.jax_utils import quat_distance_jax

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

class SRBM_MPPI(BaseMPPI):
    """
    Reduced-order MPPI controller:
    - internal optimizer: JAX SRBM rollouts over per-foot GRFs
    - simulator interface: update, next_goal, eval_best_trajectory
    - low-level output: joint targets obtained from gait reference plus SRBM force residuals
    """

    def __init__(self, task="walk_straight"):
        # Task metadata and waypoint schedule from task registry.
        self.task = task
        self.task_data = get_task(task)

        self.goal_pos = self.task_data["goal_pos"]
        self.goal_ori = self.task_data["default_orientation"]
        self.cmd_vel = self.task_data["cmd_vel"]
        self.goal_thresh = self.task_data["goal_thresh"]
        self.desired_gait = self.task_data["desired_gait"]
        self.waiting_times = self.task_data["waiting_times"]

        # Resolve model/config paths for BaseMPPI initialization.
        config_path = os.path.join(BASE_DIR, self.task_data["config_path"])
        model_path = os.path.join(BASE_DIR, "../..", self.task_data["model_path"])

        # BaseMPPI sets MuJoCo model, sampling params, buffers, and thread pool.
        super().__init__(model_path, config_path)

        # load the configuration file
        with open(config_path, "r") as f:
            params = yaml.safe_load(f)

        # Kept for compatibility with existing config structure (not used directly by SRBM cost).
        self.Q = np.diag(np.array(params["Q_diag"]))
        self.R = np.diag(np.array(params["R_diag"]))

        # Gait objects provide phase indices and joint reference trajectories.
        self.gaits = {
            "in_place": GaitScheduler(gait_path=GAIT_INPLACE_PATH, name="in_place"),
            "trot": GaitScheduler(gait_path=GAIT_TROT_PATH, name="trot"),
            "walk": GaitScheduler(gait_path=GAIT_WALK_PATH, name="walk"),
            "walk_fast": GaitScheduler(gait_path=GAIT_WALK_FAST_PATH, name="walk_fast"),
        }

        self.goal_index = 0
        self.timer = Timer(end_time=self.waiting_times[0])
        self.gait_scheduler = self.gaits[self.desired_gait[self.goal_index]]

        # Body reference format matches existing controllers:
        # [pos(3), quat(4), vel_cmd/padding(6)].
        self.body_ref = np.concatenate(
            (
                self.goal_pos[self.goal_index],
                self.goal_ori[self.goal_index],
                self.cmd_vel[self.goal_index],
                np.zeros(4),
            )
        )

        # Runtime state cache.
        self.obs = None
        self.internal_ref = True
        self.task_success = False
        self.exp_weights = np.ones(self.n_samples) / self.n_samples
        self.best_srbm_cost = None
        self.best_srbm_rollout = None
        self.selected_srbm_trajectory = None
        self.fallback_blend = 0.9
        self.fallback_blend_min = 0.60
        self.fallback_blend_max = 0.95
        self.action_smoothing_alpha = 0.35
        self.max_action_step = 0.14
        self.fallback_controller = MPPI(task=task)

        # Reduced-order physical model extracted from MuJoCo.
        self.srbm = SRBM.from_mujoco_model(self.model, dt=self.h)
        # Build batched cost evaluator once (JIT compile on first call).
        self._batched_rollout_cost_jax = jax.jit(
            jax.vmap(
                SRBM_MPPI.rollout_cost_jax,
                in_axes=(None, 0, None, None, None, None, None, None, None),
                out_axes=(0, 0),
            )
        )

        # Force-space MPPI settings (controls are GRFs, not joint angles).
        self.force_sigma = np.array([25.0, 20.0, 35.0] * 4, dtype=np.float32)
        self.force_bounds_min = np.array([-80.0, -80.0, 0.0] * 4, dtype=np.float32)
        self.force_bounds_max = np.array([80.0, 80.0, 180.0] * 4, dtype=np.float32)
        self.force_mu = 0.5
        self.force_fz_min = 0.0
        self.force_fz_max = float(self.srbm.mass * 9.81)
        self.force_delta_clip = 60.0
        self.force_blend_sample = 0.6
        self.force_blend_nominal = 0.4
        self.weight_eps = 1e-8
        self.joint_torque_to_pos_gain = np.array(
            [0.0020, 0.0012, 0.0022] * 4, dtype=np.float32
        )
        self.joint_vel_damping = np.array([0.0030, 0.0025, 0.0030] * 4, dtype=np.float32)
        self.max_joint_delta = 0.25
        # Actuator-space sign correction [FR, FL, RR, RL] x [thigh, hip, knee].
        # For this MuJoCo setup, knee commands from gait tables need inversion.
        self.joint_sign_correction = np.array(
            [
                [1.0, 1.0, -1.0],
                [1.0, 1.0, -1.0],
                [1.0, 1.0, -1.0],
                [1.0, 1.0, -1.0],
            ],
            dtype=np.float32,
        )
        self.foot_site_names = ("FR", "FL", "RR", "RL")
        self.foot_site_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in self.foot_site_names
        ]

        # Initialize nominal joint-space plan and SRBM force plan.
        self.reset_planner()
        self._initialized_from_gait = False
        self.prev_action = self.sampling_init.copy()

        # print(f"[SRBM_MPPI] task={task}")
        # print(
        #     f"[SRBM_MPPI] horizon={self.horizon}, n_samples={self.n_samples}, "
        #     f"force_dim=12, joint_dim={self.act_dim}"
        # )

    @staticmethod
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
        # Predict SRBM state rollout for one candidate force trajectory.
        # force_traj: (H, 12), states: (H, 13)
        states = SRBM.rollout_srbm_jax(
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

        # Tracking errors against task reference.
        pos_err = pos - ref_pos[None, :]
        vel_err = lin_vel - ref_lin_vel[None, :]
        quat_err = quat_distance_jax(quat, jnp.repeat(ref_quat[None, :], quat.shape[0], axis=0))

        # Contact-dependent helper terms.
        stance_counts = jnp.maximum(jnp.sum(contact_schedule, axis=1, keepdims=True), 1.0)
        forces_reshaped = force_traj.reshape(force_traj.shape[0], 4, 3)
        swing_forces = forces_reshaped * (1.0 - contact_schedule[:, :, None])

        # Running stage cost over horizon.
        running_cost = (
            4.0 * jnp.sum(pos_err ** 2, axis=1)
            + 2.0 * jnp.sum(vel_err ** 2, axis=1)
            + 0.3 * jnp.sum(ang_vel ** 2, axis=1)
            + 6.0 * quat_err
            + 2.0e-4 * jnp.sum((force_traj - nominal_force_traj) ** 2, axis=1)
        )
        # Regularization and feasibility penalties.
        smoothness_cost = 5.0e-4 * jnp.sum((force_traj[1:] - force_traj[:-1]) ** 2)
        swing_cost = 5.0e-3 * jnp.sum(swing_forces ** 2)
        negative_fz_cost = 1.0e-2 * jnp.sum((forces_reshaped[:, :, 2] < -1e-3).astype(jnp.float32))
        force_balance_cost = 5.0e-2 * jnp.sum((forces_reshaped[:, :, 2] / stance_counts) ** 2)
        # Terminal state tracking.
        terminal_cost = (
            10.0 * jnp.sum((states[-1, 0:3] - ref_pos) ** 2)
            + 4.0 * jnp.sum((states[-1, 7:10] - ref_lin_vel) ** 2)
        )
        # Encourage moving toward the current waypoint direction over the horizon.
        start_pos = initial_state[0:3]
        goal_vec = ref_pos - start_pos
        goal_norm = jnp.maximum(jnp.linalg.norm(goal_vec), 1e-3)
        goal_dir = goal_vec / goal_norm
        net_progress = jnp.dot(states[-1, 0:3] - start_pos, goal_dir)
        progress_reward = -2.0 * net_progress

        # Return scalar sample cost and rollout for diagnostics/best-sample logging.
        total_cost = (
            running_cost.sum()
            + smoothness_cost
            + swing_cost
            + negative_fz_cost
            + force_balance_cost
            + terminal_cost
            + progress_reward
        )
        return total_cost, states

    @staticmethod
    def _quat_to_roll_pitch(q):
        w, x, y, z = q
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)
        return float(roll), float(pitch)

    def _compute_adaptive_blend(self, obs, costs_sum):
        # Increase fallback reliance when the robot is unstable or SRBM sample quality is poor.
        roll, pitch = self._quat_to_roll_pitch(obs[3:7] / max(np.linalg.norm(obs[3:7]), 1e-8))
        tilt_mag = np.sqrt(roll * roll + pitch * pitch)
        height_err = abs(obs[2] - self.body_ref[2])

        costs = np.asarray(costs_sum, dtype=np.float64)
        valid = np.isfinite(costs)
        if not np.any(valid):
            return self.fallback_blend_max
        costs = costs[valid]
        spread = (np.percentile(costs, 80) - np.percentile(costs, 20)) / max(abs(np.median(costs)), 1.0)

        stability_score = 0.0
        stability_score += np.clip((tilt_mag - 0.25) / 0.35, 0.0, 1.0)
        stability_score += np.clip((height_err - 0.05) / 0.12, 0.0, 1.0)
        stability_score += np.clip((spread - 0.10) / 0.40, 0.0, 1.0)
        stability_score = np.clip(stability_score / 3.0, 0.0, 1.0)

        return float(
            self.fallback_blend_min
            + (self.fallback_blend_max - self.fallback_blend_min) * stability_score
        )

    def reset_planner(self):
        # Base plan in joint space + separate SRBM force plan.
        super().reset_planner()
        self.srbm_trajectory = np.zeros((self.horizon, 12), dtype=np.float32)

    def update_goal_orientation(self, obs):
        # Heading target points toward current position goal unless in waiting mode.
        direction = self.body_ref[:3] - obs[:3]
        goal_delta = np.linalg.norm(direction)

        if goal_delta > 0.1 and not self.timer.waiting:
            goal_ori = calculate_orientation_quaternion(obs[:3], self.body_ref[:3])
        else:
            goal_ori = np.array([1, 0, 0, 0])

        self.body_ref[3:7] = goal_ori
        return direction

    def initialize_trajectory_from_gait(self):
        # Pull gait joint references for current horizon.
        if self.internal_ref:
            self.joints_ref = self.gait_scheduler.gait[:, self.gait_scheduler.indices[: self.horizon]]

        # One-time initialization of open-loop joint trajectory from gait table.
        if not self._initialized_from_gait:
            jr = np.transpose(self.joints_ref, (1, 0))[:, :12]
            self.trajectory = jr.copy()
            self.selected_trajectory = jr.copy()
            self._initialized_from_gait = True

    def build_contact_schedule(self):
        # Build per-step stance mask (H,4) from gait phase pattern.
        gait_name = self.desired_gait[self.goal_index]
        if gait_name == "trot":
            phase_pattern = np.array(
                [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]], dtype=np.float32
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

    def build_nominal_force_trajectory(self, contact_schedule):
        # Nominal vertical load sharing over stance feet (quasi-static support).
        nominal = np.zeros((self.horizon, 4, 3), dtype=np.float32)
        for k in range(self.horizon):
            stance = contact_schedule[k]
            n_stance = max(int(np.sum(stance)), 1)
            nominal_fz = self.srbm.mass * 9.81 / n_stance
            nominal[k, :, 2] = stance * nominal_fz
        return nominal.reshape(self.horizon, 12)

    def sample_force_trajectories(self, nominal_force_traj, contact_schedule):
        # Sample MPPI candidates around previous force plan and blend with nominal support.
        noise = self.random_generator.normal(
            size=(self.n_samples, self.horizon, 12)
        ).astype(np.float32) * self.force_sigma
        actions = self.srbm_trajectory[None, :, :] + noise
        actions = (
            self.force_blend_sample * actions
            + self.force_blend_nominal * nominal_force_traj[None, :, :]
        )
        return self._enforce_force_constraints_batched(actions, contact_schedule)

    def _enforce_force_constraints_single(self, force_traj, contact_schedule):
        forces = force_traj.reshape(self.horizon, 4, 3).copy()
        contact = contact_schedule.astype(np.float32)

        # Zero forces on swing feet.
        forces *= contact[:, :, None]

        # Push-only and per-leg vertical force bounds.
        fz = np.clip(forces[:, :, 2], self.force_fz_min, self.force_fz_max)
        forces[:, :, 2] = fz

        # Friction cone: |fx|,|fy| <= mu * fz.
        fxy_lim = self.force_mu * fz
        forces[:, :, 0] = np.clip(forces[:, :, 0], -fxy_lim, fxy_lim)
        forces[:, :, 1] = np.clip(forces[:, :, 1], -fxy_lim, fxy_lim)
        return forces.reshape(self.horizon, 12)

    def _enforce_force_constraints_batched(self, force_trajs, contact_schedule):
        out = np.empty_like(force_trajs)
        for i in range(force_trajs.shape[0]):
            out[i] = self._enforce_force_constraints_single(force_trajs[i], contact_schedule)
        return out

    def _compute_leg_jacobians(self, obs):
        data = mujoco.MjData(self.model)
        data.qpos[:] = obs[: self.model.nq]
        data.qvel[:] = obs[self.model.nq : self.model.nq + self.model.nv]
        mujoco.mj_forward(self.model, data)

        jacobians = []
        for site_id in self.foot_site_ids:
            if site_id < 0:
                jacobians.append(np.zeros((3, self.model.nv), dtype=np.float32))
                continue
            jacp = np.zeros((3, self.model.nv), dtype=np.float64)
            jacr = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacSite(self.model, data, jacp, jacr, site_id)
            jacobians.append(jacp.astype(np.float32))
        return jacobians

    def _joint_targets_from_forces(self, obs, joint_ref_traj, force_traj, contact_schedule):
        # Jacobian-transpose stance force mapping: tau = -J^T f.
        jacobians = self._compute_leg_jacobians(obs)

        joint_ref = joint_ref_traj.copy().reshape(self.horizon, 4, 3)
        # Convert gait-table convention into actuator convention.
        joint_ref *= self.joint_sign_correction[None, :, :]
        joint_plan = joint_ref.reshape(self.horizon, 12)

        qvel = obs[self.model.nq : self.model.nq + self.model.nv]
        leg_qvel = np.stack(
            [
                qvel[6:9],
                qvel[9:12],
                qvel[12:15],
                qvel[15:18],
            ],
            axis=0,
        ).reshape(12)

        forces = force_traj.reshape(self.horizon, 4, 3)
        first_contact = contact_schedule[0]
        tau_cmd = np.zeros((4, 3), dtype=np.float32)
        for leg in range(4):
            if first_contact[leg] < 0.5:
                continue
            J_leg = jacobians[leg][:, 6 + 3 * leg : 6 + 3 * (leg + 1)]
            tau_cmd[leg] = -J_leg.T @ forces[0, leg]

        tau_cmd = tau_cmd.reshape(12)
        dq = (
            self.joint_torque_to_pos_gain * tau_cmd
            - self.joint_vel_damping * leg_qvel
        )
        dq = np.clip(dq, -self.max_joint_delta, self.max_joint_delta)
        joint_plan[0] = np.clip(joint_plan[0] + dq, self.act_min, self.act_max)

        for k in range(1, self.horizon):
            joint_plan[k] = np.clip(joint_plan[k], self.act_min, self.act_max)
        return joint_plan

    def compute_sample_weights(self, costs_sum):
        # MPPI importance weighting with normalized cost spread.
        finite_costs = costs_sum[np.isfinite(costs_sum)]
        if finite_costs.size == 0:
            return np.ones_like(costs_sum) / max(len(costs_sum), 1)
        min_cost = np.min(finite_costs)
        max_cost = np.max(finite_costs)
        denom = max(max_cost - min_cost, self.weight_eps)
        logits = -(costs_sum - min_cost) / (self.temperature * denom)
        logits = np.where(np.isfinite(logits), logits, -1e6)
        logits = logits - np.max(logits)
        weights = np.exp(logits)
        weight_sum = np.sum(weights)
        if not np.isfinite(weight_sum) or weight_sum <= 1e-12:
            return np.ones_like(costs_sum) / max(len(costs_sum), 1)
        return weights / weight_sum

    def next_goal(self):
        # Goal progression follows the same timer-based logic as MPPI locomotion baseline.
        self.fallback_controller.next_goal()
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
        # Exposed for simulator logging/plotting hooks.
        if self.obs is None:
            return None
        return self.best_srbm_cost

    def calculate_total_cost(self,
        initial_state, force_trajs, contact_schedule, body_ref, nominal_force_traj,return_states=False,
    ):
        """
        Evaluate all MPPI force samples with SRBM rollout-cost.
        initial_state: (13,)
        force_trajs: (N, H, 12)
        contact_schedule: (H, 4)
        body_ref: (10,)
        nominal_force_traj: (H, 12)
        """
        # Keep this thin wrapper so update() can stay high-level/readable.
        costs_sum, state_rollouts = self.rollout_cost_batched(
            initial_state=initial_state,
            force_trajs=force_trajs,
            contact_schedule=contact_schedule,
            body_ref=body_ref,
            nominal_force_traj=nominal_force_traj,
            return_states=True,
        )
        if return_states:
            return costs_sum, state_rollouts
        return costs_sum

    def rollout_cost_batched(
        self, initial_state, force_trajs, contact_schedule, body_ref, nominal_force_traj, return_states=False,
    ):
        # JIT + VMAP evaluation across sampled force trajectories.
        # Output shapes: costs_sum -> (N,), state_rollouts -> (N, H, 13)
        costs_sum, state_rollouts = self._batched_rollout_cost_jax(
            jnp.asarray(initial_state, dtype=jnp.float32),
            jnp.asarray(force_trajs, dtype=jnp.float32),
            jnp.asarray(contact_schedule, dtype=jnp.float32),
            jnp.asarray(body_ref, dtype=jnp.float32),
            jnp.asarray(nominal_force_traj, dtype=jnp.float32),
            jnp.asarray(self.srbm.foot_positions_body, dtype=jnp.float32),
            jnp.asarray(self.srbm.mass, dtype=jnp.float32),
            jnp.asarray(self.srbm.inertia_body, dtype=jnp.float32),
            jnp.asarray(self.srbm.dt, dtype=jnp.float32),
        )
        costs_sum = np.asarray(costs_sum)
        if return_states:
            return costs_sum, np.asarray(state_rollouts)
        return costs_sum

    def update(self, obs):
        fallback_action = self.fallback_controller.update(obs)

        # 1) Update references from current observation.
        self.obs = obs
        direction = self.update_goal_orientation(obs)
        self.initialize_trajectory_from_gait()

        # 2) Build force-sampling primitives for this MPC step.
        contact_schedule = self.build_contact_schedule()
        nominal_force_traj = self.build_nominal_force_trajectory(contact_schedule)
        force_trajs = self.sample_force_trajectories(nominal_force_traj, contact_schedule)

        # 3) Evaluate sampled force trajectories on SRBM.
        initial_state = self.srbm.obs_to_state(obs)

        costs_sum, state_rollouts = self.calculate_total_cost(
            initial_state=initial_state,
            force_trajs=force_trajs,
            contact_schedule=contact_schedule,
            body_ref=self.body_ref[:10],
            nominal_force_traj=nominal_force_traj,
            return_states=True,
        )

        # 4) MPPI weighting from sample costs.
        self.gait_scheduler.roll()
        self.exp_weights = self.compute_sample_weights(costs_sum)

        # 5) Weighted force update and bounds projection.
        weighted_forces = self.exp_weights[:, None, None] * force_trajs
        updated_force_traj = np.sum(weighted_forces, axis=0) / (np.sum(self.exp_weights) + 1e-10)
        updated_force_traj = self._enforce_force_constraints_single(updated_force_traj, contact_schedule)

        # 6) Map optimized forces to joint-space targets for MuJoCo low-level actuation.
        joint_ref_traj = np.transpose(self.joints_ref, (1, 0))[:, :12]
        updated_joint_traj = self._joint_targets_from_forces(
            obs,
            joint_ref_traj,
            updated_force_traj,
            contact_schedule,
        )

        # 7) Store best sample diagnostics.
        best_idx = int(np.argmin(costs_sum))
        self.best_srbm_cost = float(costs_sum[best_idx])
        self.best_srbm_rollout = state_rollouts[best_idx]

        # 8) Shift receding-horizon plans.
        self.selected_srbm_trajectory = updated_force_traj
        self.srbm_trajectory = np.roll(updated_force_traj, shift=-1, axis=0)
        self.srbm_trajectory[-1] = nominal_force_traj[-1]

        self.selected_trajectory = updated_joint_traj
        self.trajectory = np.roll(updated_joint_traj, shift=-1, axis=0)
        self.trajectory[-1] = updated_joint_traj[-1]

        # 9) Progress task gating if still far from current goal.
        if np.linalg.norm(direction) >= self.goal_thresh[self.goal_index]:
            self.timer.waiting = False

        # Controller output: blend SRBM joint target with robust baseline MPPI command.
        blend = self._compute_adaptive_blend(obs, costs_sum)
        action = blend * fallback_action + (1.0 - blend) * updated_joint_traj[0]

        # Smooth and rate-limit commands to reduce jerks and contact loss.
        action = self.action_smoothing_alpha * action + (1.0 - self.action_smoothing_alpha) * self.prev_action
        step = np.clip(action - self.prev_action, -self.max_action_step, self.max_action_step)
        action = self.prev_action + step
        action = np.clip(action, self.act_min, self.act_max)
        self.prev_action = action.copy()
        return action

    def __del__(self):
        if hasattr(self, "executor"):
            self.shutdown()
