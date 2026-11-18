"""Turn a planned trunk path p(t), v(t) (world xy) into the per-step body reference MPPI_box_push uses.

Row layout matches body_ref: [x, y, z, qw, qx, qy, qz, vx_body, vy_body, vz, wx, wy, wz].
Heading comes from the plan when it gives one (e.g. KSOS on the trunk model), otherwise it follows the
planned velocity and is held while the plan is (nearly) still. Either way it is rate limited, so a
sharp turn or a replan does not ask the trunk for a yaw jump.
"""
import numpy as np


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class BodyReference:
    def __init__(self, yaw0=0.0, z=0.27, v_min=0.03, yaw_rate=1.0):  # Go1 tracks 1 rad/s (tested)
        self.z, self.v_min, self.yaw_rate = z, v_min, yaw_rate
        self.yaw, self.t_last = yaw0, None

    def set_plan(self, t, P, V, Y=None):
        """t: (N,) times, P: (N, 2) positions, V: (N, 2) velocities, Y: (N,) optional yaw, world frame."""
        self.t, self.P, self.V = np.asarray(t), np.asarray(P), np.asarray(V)
        self.Y = None if Y is None else np.unwrap(np.asarray(Y))

    def sample(self, tq):
        tq = np.clip(tq, self.t[0], self.t[-1])  # hold the ends of the plan
        p = np.stack([np.interp(tq, self.t, self.P[:, i]) for i in range(2)], 1)
        v = np.stack([np.interp(tq, self.t, self.V[:, i]) for i in range(2)], 1)
        return p, v

    def horizon(self, t0, H=40, dt=0.01):
        tq = t0 + dt * np.arange(H)
        p, v = self.sample(tq)

        yaw = np.empty(H)
        y = self.yaw
        step0 = dt if self.t_last is None else max(t0 - self.t_last, 0.0)
        target = None if self.Y is None else np.interp(np.clip(tq, self.t[0], self.t[-1]), self.t, self.Y)
        for k in range(H):
            if target is not None or np.linalg.norm(v[k]) > self.v_min:
                goal = target[k] if target is not None else np.arctan2(v[k, 1], v[k, 0])
                max_step = self.yaw_rate * (step0 if k == 0 else dt)
                y += np.clip(wrap(goal - y), -max_step, max_step)
            yaw[k] = y
        self.yaw, self.t_last = yaw[0], t0  # commit only the current step

        c, s = np.cos(yaw), np.sin(yaw)
        ref = np.zeros((H, 13))
        ref[:, 0:2] = p
        ref[:, 2] = self.z
        ref[:, 3] = np.cos(yaw / 2)
        ref[:, 6] = np.sin(yaw / 2)
        ref[:, 7] = c * v[:, 0] + s * v[:, 1]   # world -> body frame
        ref[:, 8] = -s * v[:, 0] + c * v[:, 1]
        return ref

    # The joint reference sets the walking speed (it outweighs the position cost), so pick the gait
    # whose speed is closest to the planned speed. Names are keys of MPPI_box_push.gaits.
    GAITS = {"walk": 0.1, "walk_0.2": 0.2}

    def speed(self, t):
        return np.linalg.norm(self.sample(np.array([t]))[1][0])

    def moving(self, t):
        return self.speed(t) > self.v_min

    def gait(self, t):
        v = self.speed(t)
        if v <= self.v_min:
            return "in_place"
        return min(self.GAITS, key=lambda g: abs(self.GAITS[g] - v))
