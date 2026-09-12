import jax
import jax.numpy as jnp

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