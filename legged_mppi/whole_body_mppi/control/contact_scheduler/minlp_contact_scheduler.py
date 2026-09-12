"""Small-horizon exact MINLP contact scheduler for one Go1 and one box.

BONMIN solves the binary face schedule and the nonlinear yaw-dependent box
dynamics.  It is intentionally isolated behind ``plan`` so a learned schedule
predictor plus unrolled SQP can replace this class without changing MPPI.
"""

import multiprocessing as mp
import os
from pathlib import Path
from typing import Mapping, Optional, Sequence

import casadi as ca
import numpy as np
import yaml

from .interfaces import BoxPlanarState, ContactSchedule


def _solve_in_quiet_worker(solver, kwargs, connection):
    """Run BONMIN in a child whose native output is permanently discarded."""
    null_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        solution = solver(**kwargs)
        connection.send((True, np.asarray(solution["x"]).reshape(-1), solver.stats()))
    except Exception as error:  # pragma: no cover - solver-specific failures
        connection.send((False, str(error), {}))
    finally:
        os.close(null_fd)
        connection.close()


class MinlpContactScheduler:
    """Receding-horizon planar MINLP with rear/left/right/free modes."""

    FACE_COUNT = 3

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).with_name("configs") / "push_box_minlp.yml"
        with open(config_path, "r", encoding="utf-8") as config_file:
            self.config = yaml.safe_load(config_file)
        self.horizon = int(self.config["horizon"])
        self.dt = float(self.config["scheduler_dt"])
        self._build_solver()
        # Raw decision vector from the last successful exact solve, used to
        # warm-start the next call instead of cold-starting from an all-free
        # guess every replan (BONMIN was otherwise re-discovering the same
        # face schedule from scratch every 0.2s, which both slowed it down
        # enough to occasionally hit solver_time_limit_s and gave it no
        # incentive to keep pursuing a left/right face switch across calls).
        self._last_solution: Optional[np.ndarray] = None

    def _build_solver(self) -> None:
        c, h, dt = self.config, self.horizon, self.dt
        # Rows are time; x_box = [x, y, yaw, vx, vy, yaw_rate].
        box = ca.MX.sym("box", 6, h + 1)
        robot = ca.MX.sym("robot", 2, h + 1)
        velocity = ca.MX.sym("velocity", 2, h)
        force = ca.MX.sym("force", self.FACE_COUNT, h)
        location = ca.MX.sym("location", self.FACE_COUNT, h)
        face = ca.MX.sym("face", self.FACE_COUNT, h)
        free = ca.MX.sym("free", 1, h)
        switching = ca.MX.sym("switching", self.FACE_COUNT, h)
        decision = ca.vertcat(ca.vec(box), ca.vec(robot), ca.vec(velocity), ca.vec(force),
                              ca.vec(location), ca.vec(face), ca.vec(free), ca.vec(switching))
        self._decision_size = int(decision.numel())
        self._face_slice_start = int(ca.vertcat(ca.vec(box), ca.vec(robot), ca.vec(velocity), ca.vec(force), ca.vec(location)).numel())
        self._free_slice_start = self._face_slice_start + self.FACE_COUNT * h

        initial = ca.MX.sym("initial", 8)  # box state followed by measured robot xy
        goal = ca.MX.sym("goal", 2)
        g, lower, upper = [], [], []

        def constrain(expression, lb=0.0, ub=0.0):
            g.append(expression)
            count = int(expression.numel())
            lower.extend([lb] * count)
            upper.extend([ub] * count)

        constrain(box[:, 0] - initial[:6])
        constrain(robot[:, 0] - initial[6:8])

        half_length, half_width = float(c["box_half_length"]), float(c["box_half_width"])
        midpoints = ((-half_length, 0.0), (0.0, half_width), (0.0, -half_width))
        normals = ((-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
        tangents = ((0.0, 1.0), (-1.0, 0.0), (1.0, 0.0))
        half_face_lengths = (half_width, half_length, half_length)
        weight = c["weights"]
        objective = 0

        for k in range(h):
            psi = box[2, k]
            rotation = ca.vertcat(ca.horzcat(ca.cos(psi), -ca.sin(psi)), ca.horzcat(ca.sin(psi), ca.cos(psi)))
            net_force = ca.MX.zeros(2, 1)
            net_torque = 0
            for s in range(self.FACE_COUNT):
                midpoint = ca.DM(midpoints[s])
                normal = ca.DM(normals[s])
                tangent = ca.DM(tangents[s])
                contact = midpoint + location[s, k] * tangent
                applied_force = -force[s, k] * rotation @ normal
                net_force += applied_force
                net_torque += contact[0] * (-force[s, k] * normal[1]) - contact[1] * (-force[s, k] * normal[0])
                constrain(force[s, k], 0.0, np.inf)
                constrain(force[s, k] - float(c["force_max"]) * face[s, k], -np.inf, 0.0)
                constrain(location[s, k] - half_face_lengths[s] * face[s, k], -np.inf, 0.0)
                constrain(location[s, k] + half_face_lengths[s] * face[s, k], 0.0, np.inf)
                push_pose = box[:2, k] + rotation @ (contact + float(c["robot_standoff"]) * normal)
                # Componentwise big-M contact-pose condition.
                delta = robot[:, k] - push_pose
                constrain(delta - float(c["contact_tolerance"]) - float(c["big_m"]) * (1 - face[s, k]), -np.inf, 0.0)
                constrain(delta + float(c["contact_tolerance"]) + float(c["big_m"]) * (1 - face[s, k]), 0.0, np.inf)
                if k:
                    constrain(switching[s, k] - face[s, k] + face[s, k - 1], 0.0, np.inf)
                    constrain(switching[s, k] + face[s, k] - face[s, k - 1], 0.0, np.inf)

            constrain(ca.sum1(face[:, k]) + free[0, k], 1.0, 1.0)
            constrain(switching[:, k], 0.0, np.inf)
            constrain(box[:2, k + 1] - box[:2, k] - dt * box[3:5, k])
            constrain(box[2, k + 1] - box[2, k] - dt * box[5, k])
            constrain(box[3:5, k + 1] - box[3:5, k] - dt / float(c["box_mass"]) * (net_force - float(c["linear_damping"]) * box[3:5, k]))
            constrain(box[5, k + 1] - box[5, k] - dt / float(c["box_yaw_inertia"]) * (net_torque - float(c["angular_damping"]) * box[5, k]))
            constrain(robot[:, k + 1] - robot[:, k] - dt * velocity[:, k])
            constrain(ca.sumsqr(velocity[:, k]), -np.inf, float(c["robot_speed_max"]) ** 2)
            constrain(box[3:5, k], -float(c["box_speed_max"]), float(c["box_speed_max"]))
            constrain(box[5, k], -float(c["box_yaw_rate_max"]), float(c["box_yaw_rate_max"]))
            objective += weight["box_position"] * ca.sumsqr(box[:2, k] - goal)
            objective += weight["box_velocity"] * ca.sumsqr(box[3:5, k]) + weight["box_yaw_rate"] * box[5, k] ** 2
            objective += weight["robot_velocity"] * ca.sumsqr(velocity[:, k])
            objective += weight["force"] * ca.sumsqr(force[:, k]) + weight["contact_switch"] * ca.sum1(switching[:, k])

        objective += weight["terminal_box_position"] * ca.sumsqr(box[:2, h] - goal)
        discrete = [False] * self._decision_size
        for index in range(self._face_slice_start, self._free_slice_start + h):
            discrete[index] = True
        self._solver = ca.nlpsol("contact_scheduler", "bonmin", {"x": decision, "f": objective,
                                  "g": ca.vertcat(*g), "p": ca.vertcat(initial, goal)},
                                 {"discrete": discrete, "print_time": False,
                                  "bonmin": {"time_limit": float(c["solver_time_limit_s"]),
                                             "bb_log_level": 0, "nlp_log_level": 0,
                                             "print_level": 0}})
        # Used to sanity-check incumbents accepted from a time-limited solve
        # (see plan()) -- BONMIN reports "success" only for a proven-optimal
        # termination, but on LIMIT_EXCEEDED it still returns its best
        # feasible incumbent when the search found one at all.
        self._g_func = ca.Function("g", [decision, ca.vertcat(initial, goal)], [ca.vertcat(*g)])
        self._lbg, self._ubg = np.asarray(lower), np.asarray(upper)
        self._lbx = np.full(self._decision_size, -np.inf)
        self._ubx = np.full(self._decision_size, np.inf)
        # Bounds make the binary nature explicit and prevent unbounded damping states.
        self._lbx[self._face_slice_start:self._free_slice_start + h] = 0.0
        self._ubx[self._face_slice_start:self._free_slice_start + h] = 1.0

    def plan(self, box_state: BoxPlanarState, robot_xy: Sequence[float], goal_xy: Sequence[float]) -> ContactSchedule:
        """Solve one short-horizon plan from the current measured state."""
        approach = self._approach_if_needed(box_state, robot_xy, goal_xy)
        if approach is not None:
            return approach
        parameters = np.r_[box_state.vector(), np.asarray(robot_xy, dtype=float), np.asarray(goal_xy, dtype=float)]
        if self._last_solution is not None:
            x0 = self._warm_start(box_state, robot_xy)
        else:
            x0 = self._initial_guess(box_state, robot_xy)
        try:
            context = mp.get_context("fork")
            parent, child = context.Pipe(duplex=False)
            worker = context.Process(target=_solve_in_quiet_worker, args=(self._solver, {
                "x0": x0, "lbx": self._lbx, "ubx": self._ubx,
                "lbg": self._lbg, "ubg": self._ubg, "p": parameters}, child))
            worker.start()
            child.close()
            try:
                solved, payload, stats = parent.recv()
            except EOFError as error:
                # The worker died before sending a result (killed, crashed,
                # OOM-killed under memory pressure) -- treat it the same as a
                # reported solver failure rather than crashing the caller.
                raise RuntimeError(f"worker died without a result: {error}") from error
            worker.join()
            if not solved:
                raise RuntimeError(payload)
            values = payload
            success, status = bool(stats.get("success", False)), stats.get("return_status", "unknown")
        except RuntimeError as error:
            return self._fallback(box_state, robot_xy, goal_xy, str(error))
        if not success:
            if status == "LIMIT_EXCEEDED" and self._is_feasible_incumbent(values, parameters):
                self._last_solution = values.copy()
                return self._unpack(values, True, f"incumbent: {status}")
            return self._fallback(box_state, robot_xy, goal_xy, status)
        self._last_solution = values.copy()
        return self._unpack(values, success, status)

    def _is_feasible_incumbent(self, values: np.ndarray, parameters: np.ndarray, tol: float = 1e-4) -> bool:
        """Whether a non-optimal BONMIN return still carries a usable solution.

        A LIMIT_EXCEEDED termination means the time budget ran out before
        optimality was proven, but the returned point is only actually usable
        if branch-and-bound had found a feasible incumbent by then -- reported
        via integral face/free variables and satisfied constraints. Without a
        feasible incumbent BONMIN returns the last (generally infeasible or
        fractional) relaxation it was exploring, which must not be treated as
        a real contact schedule.
        """
        h = self.horizon
        discrete = values[self._face_slice_start:self._free_slice_start + h]
        if not np.all((np.abs(discrete) < tol) | (np.abs(discrete - 1) < tol)):
            return False
        constraints = np.asarray(self._g_func(values, parameters)).flatten()
        return bool(np.all(constraints >= self._lbg - tol) and np.all(constraints <= self._ubg + tol))

    def _warm_start(self, box_state: BoxPlanarState, robot_xy: Sequence[float]) -> np.ndarray:
        """Shift the previous solve's trajectory forward by one horizon step.

        Reoptimizing from scratch every 0.2s gave BONMIN no reason to keep
        pursuing a face schedule across replans (only the switching cost
        within a single ~1s horizon discouraged flip-flopping), and made
        solves slow enough to occasionally hit the time limit. Continuing
        from last time's plan fixes both.
        """
        h = self.horizon
        cursor = 0
        box = self._last_solution[cursor:cursor + 6 * (h + 1)].reshape((6, h + 1), order="F"); cursor += 6 * (h + 1)
        robot = self._last_solution[cursor:cursor + 2 * (h + 1)].reshape((2, h + 1), order="F"); cursor += 2 * (h + 1)
        velocity = self._last_solution[cursor:cursor + 2 * h].reshape((2, h), order="F"); cursor += 2 * h
        force = self._last_solution[cursor:cursor + self.FACE_COUNT * h].reshape((self.FACE_COUNT, h), order="F"); cursor += self.FACE_COUNT * h
        location = self._last_solution[cursor:cursor + self.FACE_COUNT * h].reshape((self.FACE_COUNT, h), order="F"); cursor += self.FACE_COUNT * h
        face = self._last_solution[cursor:cursor + self.FACE_COUNT * h].reshape((self.FACE_COUNT, h), order="F"); cursor += self.FACE_COUNT * h
        free = self._last_solution[cursor:cursor + h].reshape((1, h), order="F"); cursor += h
        switching = self._last_solution[cursor:cursor + self.FACE_COUNT * h].reshape((self.FACE_COUNT, h), order="F")

        def shift(array: np.ndarray) -> np.ndarray:
            shifted = np.empty_like(array)
            shifted[:, :-1] = array[:, 1:]
            shifted[:, -1] = array[:, -1]
            return shifted

        box, robot = shift(box), shift(robot)
        box[:, 0] = box_state.vector()
        robot[:, 0] = np.asarray(robot_xy, dtype=float)
        velocity, force, location, face, free, switching = (
            shift(velocity), shift(force), shift(location), shift(face), shift(free), shift(switching))

        return np.r_[box.flatten(order="F"), robot.flatten(order="F"), velocity.flatten(order="F"),
                     force.flatten(order="F"), location.flatten(order="F"), face.flatten(order="F"),
                     free.flatten(order="F"), switching.flatten(order="F")]

    def _select_face(self, box_xy: np.ndarray, yaw: float, robot_xy: np.ndarray, direction: np.ndarray) -> int:
        """Pick the face whose outward normal best opposes `direction` (the unit
        box-to-goal vector). A plain argmax always resolves ties (e.g. a goal at
        exactly 45 degrees, where two faces are equally good) in favor of
        whichever face is listed first -- "rear" -- regardless of where the
        robot actually is. Among tied faces, prefer whichever standoff pose the
        robot can reach first.
        """
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        normals = np.array([[-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        midpoints = np.array(((-self.config["box_half_length"], 0.0),
                              (0.0, self.config["box_half_width"]),
                              (0.0, -self.config["box_half_width"])))
        scores = np.array([np.dot(-(rotation @ normal), direction) for normal in normals])
        candidates = np.flatnonzero(scores >= scores.max() - 1e-9)
        if candidates.size == 1:
            return int(candidates[0])
        standoffs = box_xy + (rotation @ (midpoints[candidates] +
                              self.config["robot_standoff"] * normals[candidates]).T).T
        distances = np.linalg.norm(standoffs - robot_xy, axis=1)
        return int(candidates[np.argmin(distances)])

    def _initial_guess(self, box_state: BoxPlanarState, robot_xy: Sequence[float]) -> np.ndarray:
        """Provide BONMIN with a feasible all-free trajectory warm start."""
        h = self.horizon
        box = np.tile(box_state.vector(), (h + 1, 1)).reshape(-1, order="F")
        robot = np.tile(np.asarray(robot_xy, dtype=float), (h + 1, 1)).reshape(-1, order="F")
        velocity = np.zeros(2 * h)
        force = np.zeros(self.FACE_COUNT * h)
        location = np.zeros(self.FACE_COUNT * h)
        faces = np.zeros(self.FACE_COUNT * h)
        free = np.ones(h)
        switching = np.zeros(self.FACE_COUNT * h)
        return np.r_[box, robot, velocity, force, location, faces, free, switching]

    def _approach_if_needed(self, box_state: BoxPlanarState, robot_xy: Sequence[float], goal_xy: Sequence[float]) -> Optional[ContactSchedule]:
        """Generate free-mode references until a short-horizon contact is reachable.

        This is the explicit ``z_free`` mode in operational form.  It avoids
        asking a short contact horizon to solve an impossible initial condition
        when Go1 begins well away from the selected box face.
        """
        robot_xy = np.asarray(robot_xy, dtype=float)
        goal_xy = np.asarray(goal_xy, dtype=float)
        direction = goal_xy - box_state.vector()[:2]
        if np.linalg.norm(direction) < 1e-6:
            return None
        direction /= np.linalg.norm(direction)
        # Choose the face whose inward normal best points toward the box goal.
        yaw = box_state.yaw
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        normals = np.array([[-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        face_index = self._select_face(box_state.vector()[:2], yaw, robot_xy, direction)
        midpoint = ((-self.config["box_half_length"], 0.0), (0.0, self.config["box_half_width"]),
                    (0.0, -self.config["box_half_width"]))[face_index]
        contact_pose = box_state.vector()[:2] + rotation @ (np.asarray(midpoint) + self.config["robot_standoff"] * normals[face_index])
        reachable_distance = self.horizon * self.dt * self.config["robot_speed_max"]
        if np.linalg.norm(contact_pose - robot_xy) <= reachable_distance:
            return None
        h = self.horizon
        delta = contact_pose - robot_xy
        velocity = delta / max(np.linalg.norm(delta), 1e-9) * self.config["robot_speed_max"]
        robot = np.zeros((h + 1, 2)); robot[0] = robot_xy
        for k in range(h):
            remaining = contact_pose - robot[k]
            step = velocity * self.dt if np.linalg.norm(remaining) > np.linalg.norm(velocity * self.dt) else remaining
            robot[k + 1] = robot[k] + step
        box = np.tile(box_state.vector(), (h + 1, 1))
        return ContactSchedule(box, robot, np.tile(velocity, (h, 1)), np.zeros((h, self.FACE_COUNT)),
                               np.zeros((h, self.FACE_COUNT)), -np.ones(h, dtype=int), True, "free-mode approach")

    def _unpack(self, values: np.ndarray, success: bool, status: str) -> ContactSchedule:
        h = self.horizon
        cursor = 0
        box = values[cursor:cursor + 6 * (h + 1)].reshape((6, h + 1), order="F").T; cursor += 6 * (h + 1)
        robot = values[cursor:cursor + 2 * (h + 1)].reshape((2, h + 1), order="F").T; cursor += 2 * (h + 1)
        velocity = values[cursor:cursor + 2 * h].reshape((2, h), order="F").T; cursor += 2 * h
        force = values[cursor:cursor + self.FACE_COUNT * h].reshape((self.FACE_COUNT, h), order="F").T; cursor += self.FACE_COUNT * h
        location = values[cursor:cursor + self.FACE_COUNT * h].reshape((self.FACE_COUNT, h), order="F").T; cursor += self.FACE_COUNT * h
        faces = values[cursor:cursor + self.FACE_COUNT * h].reshape((self.FACE_COUNT, h), order="F").T
        face_indices = np.where(np.max(faces, axis=1) > 0.5, np.argmax(faces, axis=1), -1)
        return ContactSchedule(box, robot, velocity, force, location, face_indices, success, status)

    def _fallback(self, box_state: BoxPlanarState, robot_xy: Sequence[float], goal_xy: Sequence[float], status: str) -> ContactSchedule:
        """Safe free-mode fallback if a MINLP solve fails or reaches its time limit."""
        h = self.horizon
        box = np.zeros((h + 1, 6)); box[0] = box_state.vector()
        robot = np.zeros((h + 1, 2)); robot[0] = np.asarray(robot_xy, dtype=float)
        velocity = np.zeros((h, 2))
        force = np.zeros((h, self.FACE_COUNT))
        locations = np.zeros((h, self.FACE_COUNT))
        faces = np.zeros(h, dtype=int)
        goal_direction = np.asarray(goal_xy, dtype=float) - box[0, :2]
        if np.linalg.norm(goal_direction) < 1e-6:
            return ContactSchedule(box, robot, velocity, force, locations, -np.ones(h, dtype=int), False, status)
        goal_direction /= np.linalg.norm(goal_direction)
        normals = np.array([[-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        face_index = self._select_face(box[0, :2], box[0, 2], np.asarray(robot_xy, dtype=float), goal_direction)
        midpoint = np.array(((-self.config["box_half_length"], 0.0), (0.0, self.config["box_half_width"]),
                             (0.0, -self.config["box_half_width"]))[face_index])
        nominal_force = min(float(self.config["force_max"]), 10.0)
        for k in range(h):
            yaw = box[k, 2]
            rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            applied_force = -nominal_force * rotation @ normals[face_index]
            acceleration = (applied_force - self.config["linear_damping"] * box[k, 3:5]) / self.config["box_mass"]
            box[k + 1, :2] = box[k, :2] + self.dt * box[k, 3:5]
            box[k + 1, 2] = box[k, 2]
            box[k + 1, 3:5] = box[k, 3:5] + self.dt * acceleration
            box[k + 1, 5] = 0.0
            robot[k + 1] = box[k + 1, :2] + rotation @ (midpoint + self.config["robot_standoff"] * normals[face_index])
            velocity[k] = (robot[k + 1] - robot[k]) / self.dt
            force[k, face_index] = nominal_force
            faces[k] = face_index
        return ContactSchedule(box, robot, velocity, force, locations, faces, False, f"fallback: {status}")
