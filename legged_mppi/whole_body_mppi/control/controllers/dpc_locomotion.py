import os
import yaml
import mujoco
import numpy as np
import jax.numpy as jnp

# Local imports (ensure these are part of your package structure)
from whole_body_mppi.utils.tasks import get_task
from whole_body_mppi.control.controllers.base_dpc import BaseDPC
from whole_body_mppi.control.gait_scheduler.scheduler import GaitScheduler
from whole_body_mppi.control.gait_scheduler.scheduler import Timer
from whole_body_mppi.utils.transforms import batch_world_to_local_velocity, calculate_orientation_quaternion

# Define base directory and paths for resource files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GAIT_DIR = os.path.join(BASE_DIR, "../gait_scheduler/gaits/")

# Paths for gait files
GAIT_INPLACE_PATH = os.path.join(GAIT_DIR, "FAST/walking_gait_raibert_FAST_0_0_10cm_100hz.tsv")
GAIT_TROT_PATH = os.path.join(GAIT_DIR, "MED/walking_gait_raibert_MED_0_5_15cm_100hz.tsv")
GAIT_WALK_PATH = os.path.join(GAIT_DIR, "MED/walking_gait_raibert_MED_0_1_10cm_100hz.tsv")
GAIT_WALK_FAST_PATH = os.path.join(GAIT_DIR, "FAST/walking_gait_raibert_FAST_0_1_10cm_100hz.tsv")

def quaternion_distance_jax(q1, q2):
    dot_products = jnp.einsum("ij,ij->i", q1, q2)
    return 1.0 - jnp.abs(dot_products)

def quat_conjugate(q):
    return q * jnp.array([1.0, -1.0, -1.0, -1.0], dtype=q.dtype)

def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    return jnp.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=1,
    )

def batch_world_to_local_velocity_jax(quaternions, world_velocities):
    # Assumes quaternions are [w, x, y, z]
    quaternions = quaternions / jnp.linalg.norm(quaternions, axis=1, keepdims=True)
    q_inv = quat_conjugate(quaternions)
    v_quat = jnp.concatenate(
        [jnp.zeros((world_velocities.shape[0], 1), dtype=world_velocities.dtype), world_velocities],
        axis=1,
    )
    local_v_quat = quat_multiply(quat_multiply(q_inv, v_quat), quaternions)
    return local_v_quat[:, 1:]

class DPC(BaseDPC):
    """
    Differentiable Predictive Control for quadruped robots.

    Attributes:
    """

    def __init__(self, task='stand') -> None:
        """
        Initialize the DPC controller with task-specific configurations.

        Args:
            task (str): The name of the task ('stand', 'walk').
        """
        print("Task: ", task)

        # Retrieve task-specific parameters
        self.task = task
        self.task_data = get_task(task)

        self.goal_pos = self.task_data['goal_pos']
        self.goal_ori = self.task_data['default_orientation'] 
        self.cmd_vel = self.task_data['cmd_vel']
        self.goal_thresh = self.task_data['goal_thresh']
        self.desired_gait = self.task_data['desired_gait']
        model_path = self.task_data['model_path'] 
        config_path = self.task_data['config_path']
        waiting_times = self.task_data['waiting_times']

        # Dynamically resolve paths for model and configuration files
        CONFIG_PATH = os.path.join(BASE_DIR, config_path)
        MODEL_PATH = os.path.join(BASE_DIR, "../..", model_path)

        # Initialize base MPPI
        super().__init__(MODEL_PATH, CONFIG_PATH)

        # load the configuration file
        with open(CONFIG_PATH, 'r') as file:
            params = yaml.safe_load(file)
        # Cost weights
        self.Q = np.diag(np.array(params['Q_diag']))
        self.R = np.diag(np.array(params['R_diag']))
        self.Q_jax = jnp.asarray(self.Q)
        self.R_jax = jnp.asarray(self.R)

        # Set initial parameters and state
        self.obs = None
        self.internal_ref = True
        self.waiting_times = waiting_times
        self.timer = Timer(end_time=self.waiting_times[0])

        # Initialize gait schedulers
        self.gaits = {
            'in_place': GaitScheduler(gait_path=GAIT_INPLACE_PATH, name='in_place'),
            'trot': GaitScheduler(gait_path=GAIT_TROT_PATH, name='trot'),
            'walk': GaitScheduler(gait_path=GAIT_WALK_PATH, name='walk'),
            'walk_fast': GaitScheduler(gait_path=GAIT_WALK_FAST_PATH, name='walk_fast')
        }
        self.gait_scheduler = self.gaits['in_place']

        # Initialize planner and goals
        # self.reset_planner()
        self.goal_index = 0
        self.body_ref = np.concatenate((self.goal_pos[self.goal_index],
                                        self.goal_ori[self.goal_index],
                                        self.cmd_vel[self.goal_index],
                                        np.zeros(4)))
        
        self.gait_scheduler = self.gaits[self.desired_gait[self.goal_index]]
        self.task_success = False

        # Debug information
        print(f"Initial goal {self.goal_index}: {self.goal_pos[self.goal_index] }")
        print(f"Initial gait {self.desired_gait[self.goal_index]}")
    
    def next_goal(self):
        """
        Progress to the next goal based on the task sequence.
        Updates the internal reference trajectory and gait scheduler.
        """
        self.timer.increment()

        if self.goal_index < len(self.goal_pos) - 1 and self.timer.done:
            # Move to the next goal
            self.goal_index += 1
            self.body_ref[:3] = self.goal_pos[self.goal_index]
            self.body_ref[7:9] = self.cmd_vel[self.goal_index]
            self.gait_scheduler = self.gaits[self.desired_gait[self.goal_index]]
            self.timer.reset()
            self.timer.end_time = self.waiting_times[self.goal_index]
            print(f"Moved to next goal {self.goal_index}: {self.goal_pos[self.goal_index]}")
            print(f"Gait: {self.desired_gait[self.goal_index]}")
            self.timer.waiting = False

        elif self.goal_index == len(self.goal_pos) - 1 and not self.task_success and self.timer.done:
            # Final goal reached
            print("Task succeeded.")
            self.task_success = True

        else:
            self.timer.waiting = True

        if not self.task_success:
            if self.desired_gait[self.goal_index] in ['in_place', 'walk', 'walk_fast']:
                self.noise_sigma = np.array([0.06, 0.1, 0.1] * 4)
            elif self.desired_gait[self.goal_index] in ['trot']:
                self.noise_sigma = np.array([0.06, 0.2, 0.2] * 4)

    def quadruped_cost_np(self, x, u, x_ref):
        """
        Compute the cost for quadruped motion based on state and action errors.

        Args:
            x (jnp.ndarray): Current states (N x state_dim).
            u (jnp.ndarray): Current actions (N x action_dim).
            x_ref (jnp.ndarray): Reference states (N x state_dim).

        Returns:
            jnp.ndarray: Computed cost for each sample.
        """
        kp = 50.0
        kd = 3.0

        x_error = x - x_ref

        q_dist = quaternion_distance_jax(x[:, 3:7], x_ref[:, 3:7])
        x_error = x_error.at[:, 3:7].set(q_dist[:, None])

        x_joint = x[:, 7:19]
        v_joint = x[:, 25:]
        u_error = kp * (u - x_joint) - kd * v_joint

        x_error = x_error.at[:, :3].set(0.0)
        x_pos_error = x[:, :3] - x_ref[:, :3]
        l1_norm_pos_cost = jnp.abs(x_pos_error @ self.Q_jax[:3, :3]).sum(axis=1)

        state_cost = jnp.einsum("bi,ij,bj->b", x_error, self.Q_jax, x_error)
        control_cost = jnp.einsum("bi,ij,bj->b", u_error, self.R_jax, u_error)

        return state_cost + control_cost + l1_norm_pos_cost

    def calculate_total_cost(self, states, actions, joints_ref, body_ref):
        """
        Calculate the total cost for all rollouts.

        Args:
            states (jnp.ndarray): Rollout states (batch x time steps x state_dim).
            actions (jnp.ndarray): Rollout actions (batch x time steps x action_dim).
            joints_ref (jnp.ndarray): Reference joint positions (time steps x joint_dim).
            body_ref (jnp.ndarray): Reference body state (state_dim).

        Returns:
            jnp.ndarray: Total cost for each sample.
        """
        batch_size = states.shape[0]
        num_pairs = states.shape[1]

        traj_body_ref = jnp.repeat(body_ref[None, :], batch_size * num_pairs, axis=0)

        states = states.reshape(-1, states.shape[2])
        actions = actions.reshape(-1, actions.shape[2])

        joints_ref = joints_ref.T
        joints_ref = jnp.tile(joints_ref, (batch_size, 1, 1))
        joints_ref = joints_ref.reshape(-1, joints_ref.shape[2])

        x_ref = jnp.concatenate(
            [
                traj_body_ref[:, :7],
                joints_ref[:, :12],
                traj_body_ref[:, 7:],
                joints_ref[:, 12:],
            ],
            axis=1,
        )

        rotated_ref = batch_world_to_local_velocity_jax(states[:, 3:7], states[:, 19:22])
        states = states.at[:, 19:22].set(rotated_ref)

        costs = self.quadruped_cost_jax(states, actions, x_ref)
        return costs.reshape(batch_size, num_pairs).sum(axis=1)

    def quaternion_distance_np(self, q1, q2):
        """
        Compute the distance between two sets of quaternions.

        Args:
            q1 (np.ndarray): Array of quaternions (N x 4).
            q2 (np.ndarray): Array of quaternions (N x 4).

        Returns:
            np.ndarray: Array of distances between the quaternions.
        """
        # Compute dot product between corresponding quaternions
        dot_products = np.einsum('ij,ij->i', q1, q2)
        # Compute distance as 1 - absolute dot product
        return 1 - np.abs(dot_products)

    def update(self, obs):
        """
        Update the MPPI controller based on the current observation.

        Args:
            obs (np.ndarray): Current state observation.
        Returns:
            np.ndarray: Selected action based on the optimal trajectory.
        """
         # Generate perturbed actions for rollouts
        self.obs = obs

        # Calculate the direction and distance to the goal
        direction = self.body_ref[:3] - obs[:3]
        goal_delta = np.linalg.norm(direction)

        # Update desired orientation based on the goal position
        if goal_delta > 0.1 and not self.timer.waiting:
            self.goal_ori = calculate_orientation_quaternion(obs[:3], self.body_ref[:3])
        else:
            self.goal_ori = np.array([1, 0, 0, 0])

        self.body_ref[3:7] = self.goal_ori

        # Actions are generated from a neural control policy
        actions = None

        # Perform rollouts using threaded rollout function
        self.rollout_func(self.state_rollouts, actions, np.repeat(
            np.array([np.concatenate([[0], obs])]), self.n_samples, axis=0), 
            num_workers=self.num_workers, nstep=self.horizon)

        # Update joint references from the gait scheduler
        if self.internal_ref:
            self.joints_ref = self.gait_scheduler.gait[:, self.gait_scheduler.indices[:self.horizon]]

        # Update the gait scheduler
        self.gait_scheduler.roll()

        updated_actions = np.clip(actions, self.act_min, self.act_max)

        # Update the trajectory with the optimal action
        self.selected_trajectory = updated_actions
        self.trajectory = np.roll(updated_actions, shift=-1, axis=0)
        self.trajectory[-1] = updated_actions[-1]

        # Return the first action in the trajectory as the output action
        return updated_actions[0]
    
    def set_policy(self, policy):
        self.policy = policy