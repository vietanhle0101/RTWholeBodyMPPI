import numpy as np
import mujoco
import jax
import jax.numpy as jnp

from whole_body_mppi.control.controllers.jax_utils import (
    quat_multiply_jax,
    quat_normalize_jax,
    quat_to_rotmat_jax,
)

GRAVITY = jnp.array([0.0, 0.0, -9.81], dtype=jnp.float32)

def step_rk4_jax(state, control, dt, ode_fn):
    """
    Generic RK4 discretization for x_{k+1} = x_k + integral(f(x,u) dt).

    Args:
        state: (13,) SRBM state [p(3), q(4), v(3), w(3)].
        control: input payload for `ode_fn`.
        dt: integration step [s].
        ode_fn: callable f(x, u) -> x_dot, shape (13,).

    Returns:
        next_state: (13,) state advanced by one RK4 step.
    """
    # Keep quaternion normalized before derivative evaluations.
    state = state.at[3:7].set(quat_normalize_jax(state[3:7]))

    k1 = ode_fn(state, control)

    s2 = state + 0.5 * dt * k1
    s2 = s2.at[3:7].set(quat_normalize_jax(s2[3:7]))
    k2 = ode_fn(s2, control)

    s3 = state + 0.5 * dt * k2
    s3 = s3.at[3:7].set(quat_normalize_jax(s3[3:7]))
    k3 = ode_fn(s3, control)

    s4 = state + dt * k3
    s4 = s4.at[3:7].set(quat_normalize_jax(s4[3:7]))
    k4 = ode_fn(s4, control)

    next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    next_state = next_state.at[3:7].set(quat_normalize_jax(next_state[3:7]))
    return next_state

class SRBM:
    """
    Single rigid body model (SRBM) wrapper.

    State convention:
    - x = [p, q, v, w], size 13
      p: position (3)
      q: orientation quaternion [w, x, y, z] (4)
      v: linear velocity (3)
      w: angular velocity (3)
    """

    def __init__(self, mass, inertia_body, foot_positions_body, dt):
        self.mass = float(mass)
        self.inertia_body = np.asarray(inertia_body, dtype=np.float32)
        self.foot_positions_body = np.asarray(foot_positions_body, dtype=np.float32)
        self.dt = float(dt)

    @staticmethod
    def srbm_ode_jax(state, forces, contacts, foot_positions_body, mass, inertia_body):
        """
        Continuous SRBM dynamics x_dot = f(x, u).

        Args:
            state: (13,) [p, q, v, w].
            forces: (4,3) foot forces for [FR, FL, RR, RL] in world frame.
            contacts: (4,) binary contact mask (1 stance, 0 swing).
            foot_positions_body: (4,3) nominal feet positions in trunk/body frame.
            mass: scalar trunk mass approximation.
            inertia_body: (3,) principal inertia values in body frame.

        Returns:
            state_dot: (13,) [p_dot, q_dot, v_dot, w_dot].
        """
        # Unpack state blocks.
        pos = state[0:3]
        quat = quat_normalize_jax(state[3:7])
        lin_vel = state[7:10]
        ang_vel = state[10:13]

        # Body->world rotation and nominal foot positions in world frame.
        rot = quat_to_rotmat_jax(quat)
        foot_world = jnp.einsum("ij,lj->li", rot, foot_positions_body)
        # Only stance legs can apply forces.
        masked_forces = forces * contacts[:, None]

        # Net wrench at COM.
        total_force = jnp.sum(masked_forces, axis=0)
        torque_world = jnp.sum(jnp.cross(foot_world, masked_forces), axis=0)

        # Rotational rigid-body dynamics in world frame.
        inertia_world = rot @ jnp.diag(inertia_body) @ rot.T
        ang_momentum = inertia_world @ ang_vel
        ang_acc = jnp.linalg.solve(
            inertia_world,
            torque_world - jnp.cross(ang_vel, ang_momentum),
        )
        # Translational dynamics.
        lin_acc = total_force / mass + GRAVITY

        # Quaternion kinematics: q_dot = 0.5 * [0,w] \otimes q.
        omega_quat = jnp.array([0.0, ang_vel[0], ang_vel[1], ang_vel[2]], dtype=quat.dtype)
        quat_dot = 0.5 * quat_multiply_jax(omega_quat, quat)

        return jnp.concatenate([lin_vel, quat_dot, lin_acc, ang_acc], axis=0)

    @staticmethod
    def rollout_srbm_jax(initial_state, force_traj, contact_schedule, foot_positions_body, mass, inertia_body, dt):
        """
        Horizon rollout with JAX scan + RK4 integration.

        Args:
            initial_state: (13,) initial SRBM state.
            force_traj: (H,12) flattened force sequence (4 legs x 3 axes).
            contact_schedule: (H,4) stance/swing mask over horizon.
            foot_positions_body: (4,3) nominal feet in body frame.
            mass, inertia_body, dt: SRBM parameters.

        Returns:
            states: (H,13) predicted states.
        """
        def scan_step(carry, inputs):
            force_k, contact_k = inputs
            # Convert flattened (12,) force vector to per-leg (4,3).
            control_k = (force_k.reshape(4, 3), contact_k)

            def ode_fn(x, u):
                f_k, c_k = u
                return SRBM.srbm_ode_jax(x, f_k, c_k, foot_positions_body, mass, inertia_body)

            next_state = step_rk4_jax(carry, control_k, dt, ode_fn)
            return next_state, next_state

        _, states = jax.lax.scan(scan_step, initial_state, (force_traj, contact_schedule))
        return states

    @classmethod
    def from_mujoco_model(cls, model, dt, trunk_name="trunk", foot_site_names=("FR", "FL", "RR", "RL")):
        """
        Construct SRBM parameters directly from MuJoCo model data.
        """
        trunk_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, trunk_name)
        if trunk_body_id < 0:
            raise ValueError(f"Body '{trunk_name}' not found in model.")

        mass = float(np.sum(model.body_mass[1:]))
        inertia_body = np.asarray(model.body_inertia[trunk_body_id], dtype=np.float32)
        foot_positions_body = cls._extract_foot_positions_body_from_model(
            model=model,
            trunk_name=trunk_name,
            foot_site_names=foot_site_names,
        )
        return cls(mass=mass, inertia_body=inertia_body, foot_positions_body=foot_positions_body, dt=dt)

    @staticmethod
    def _extract_foot_positions_body_from_model(model, trunk_name="trunk", foot_site_names=("FR", "FL", "RR", "RL")):
        """
        Extract nominal foot points in trunk frame from MuJoCo sites.
        Falls back to defaults if names are missing.
        """
        default_foot_positions = np.array(
            [
                [0.19, -0.11, -0.28],  # FR
                [0.19, 0.11, -0.28],   # FL
                [-0.19, -0.11, -0.28], # RR
                [-0.19, 0.11, -0.28],  # RL
            ],
            dtype=np.float32,
        )

        trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, trunk_name)
        if trunk_id < 0:
            return default_foot_positions

        site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in foot_site_names]
        if any(sid < 0 for sid in site_ids):
            return default_foot_positions

        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        if model.nkey > 1:
            data.qpos[:] = model.key_qpos[1]
            data.qvel[:] = model.key_qvel[1]
            data.ctrl[:] = model.key_ctrl[1]
        elif model.nkey > 0:
            data.qpos[:] = model.key_qpos[0]
            data.qvel[:] = model.key_qvel[0]
            data.ctrl[:] = model.key_ctrl[0]
        mujoco.mj_forward(model, data)

        trunk_pos = data.xpos[trunk_id].copy()
        trunk_rot = data.xmat[trunk_id].reshape(3, 3).copy()

        foot_positions_body = []
        for sid in site_ids:
            p_world = data.site_xpos[sid]
            p_body = trunk_rot.T @ (p_world - trunk_pos)
            foot_positions_body.append(p_body)
        return np.asarray(foot_positions_body, dtype=np.float32)

    @staticmethod
    def obs_to_state(obs):
        """
        Convert RTWholeBodyMPPI observation to SRBM state layout.

        Expected fields:
        - obs[0:3]   : base position
        - obs[3:7]   : base quaternion
        - obs[19:22] : base linear velocity
        - obs[22:25] : base angular velocity
        """
        pos = obs[0:3]
        quat = obs[3:7]
        lin_vel = obs[19:22]
        ang_vel = obs[22:25]
        quat = quat / max(np.linalg.norm(quat), 1e-8)
        return np.concatenate([pos, quat, lin_vel, ang_vel]).astype(np.float32)
