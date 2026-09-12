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
        # Multi-rate discretization: continuous box/robot dynamics integrate
        # at the fine dynamics_dt (accuracy), while the discrete face/force/
        # location/switching decisions are only allowed to change once per
        # contact_dt block (keeps the combinatorial search small even at a
        # longer total lookahead). `horizon` counts contact_dt blocks, so
        # total lookahead = horizon * contact_dt. Configs that only specify
        # `scheduler_dt` (the original single-rate configs) fall back to
        # dynamics_dt = contact_dt = scheduler_dt, i.e. block_size = 1,
        # reproducing the original formulation exactly.
        self.dynamics_dt = float(self.config.get("dynamics_dt", self.config.get("scheduler_dt")))
        self.dt = float(self.config.get("contact_dt", self.config.get("scheduler_dt")))
        self.block_size = round(self.dt / self.dynamics_dt)
        if abs(self.block_size * self.dynamics_dt - self.dt) > 1e-9:
            raise ValueError(f"contact_dt ({self.dt}) must be an integer multiple of dynamics_dt ({self.dynamics_dt})")
        self.replan_period = 1.0 / float(self.config["replan_rate_hz"])
        if abs(self.replan_period - self.dt) > 1e-9:
            raise ValueError(
                "replan_rate_hz must equal 1/contact_dt (or 1/scheduler_dt for single-rate configs): "
                f"got {self.replan_period}s versus {self.dt}s")
        self._build_solver()
        # Raw decision vector from the last successful exact solve, used to
        # warm-start the next call instead of cold-starting from an all-free
        # guess every replan (BONMIN was otherwise re-discovering the same
        # face schedule from scratch every 0.2s, which both slowed it down
        # enough to occasionally hit solver_time_limit_s and gave it no
        # incentive to keep pursuing a left/right face switch across calls).
        self._last_solution: Optional[np.ndarray] = None

    def _build_solver(self) -> None:
        c = self.config
        blocks = self.horizon               # number of contact_dt blocks (discrete decisions)
        block_size = self.block_size
        n = blocks * block_size             # number of dynamics_dt fine steps
        dt = self.dynamics_dt
        # Rows are time; x_box = [x, y, yaw, vx, vy, yaw_rate]. box/robot/
        # velocity are at the fine dynamics_dt resolution; force/location/
        # face/free/switching are held constant within each contact_dt block
        # (block_size fine steps), i.e. one discrete decision per block.
        box = ca.MX.sym("box", 6, n + 1)
        robot = ca.MX.sym("robot", 2, n + 1)
        velocity = ca.MX.sym("velocity", 2, n)
        force = ca.MX.sym("force", self.FACE_COUNT, blocks)
        location = ca.MX.sym("location", self.FACE_COUNT, blocks)
        face = ca.MX.sym("face", self.FACE_COUNT, blocks)
        free = ca.MX.sym("free", 1, blocks)
        switching = ca.MX.sym("switching", self.FACE_COUNT, blocks)
        decision = ca.vertcat(ca.vec(box), ca.vec(robot), ca.vec(velocity), ca.vec(force),
                              ca.vec(location), ca.vec(face), ca.vec(free), ca.vec(switching))
        self._decision_size = int(decision.numel())
        self._face_slice_start = int(ca.vertcat(ca.vec(box), ca.vec(robot), ca.vec(velocity), ca.vec(force), ca.vec(location)).numel())
        self._free_slice_start = self._face_slice_start + self.FACE_COUNT * blocks

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

        # Loop-invariant constants, hoisted out of the per-block/per-step
        # loops below (same value every iteration; no need to re-cast from
        # the config dict or rebuild the small DM constants each time).
        force_max = float(c["force_max"])
        contact_tolerance = float(c["contact_tolerance"])
        big_m = float(c["big_m"])
        robot_standoff = float(c["robot_standoff"])
        box_mass = float(c["box_mass"])
        linear_damping = float(c["linear_damping"])
        box_yaw_inertia = float(c["box_yaw_inertia"])
        angular_damping = float(c["angular_damping"])
        robot_speed_max = float(c["robot_speed_max"])
        box_speed_max = float(c["box_speed_max"])
        box_yaw_rate_max = float(c["box_yaw_rate_max"])
        face_geometry = [(ca.DM(midpoints[s]), ca.DM(normals[s]), ca.DM(tangents[s])) for s in range(self.FACE_COUNT)]

        # Block-constant discrete-decision constraints and costs (one per
        # contact_dt block, not per fine dynamics step).
        for b in range(blocks):
            constrain(ca.sum1(face[:, b]) + free[0, b], 1.0, 1.0)
            if b:
                for s in range(self.FACE_COUNT):
                    constrain(switching[s, b] - face[s, b] + face[s, b - 1], 0.0, np.inf)
                    constrain(switching[s, b] + face[s, b] - face[s, b - 1], 0.0, np.inf)
            for s in range(self.FACE_COUNT):
                # force[s,b] >= 0 and switching[s,b] >= 0 are plain bounds on
                # raw decision variables -- set on _lbx below instead of as
                # general constraint rows.
                constrain(force[s, b] - force_max * face[s, b], -np.inf, 0.0)
                constrain(location[s, b] - half_face_lengths[s] * face[s, b], -np.inf, 0.0)
                constrain(location[s, b] + half_face_lengths[s] * face[s, b], 0.0, np.inf)
            objective += weight["force"] * ca.sumsqr(force[:, b]) + weight["contact_switch"] * ca.sum1(switching[:, b])

        # Fine-resolution dynamics and contact-geometry constraints.
        for k in range(n):
            b = k // block_size
            psi = box[2, k]
            rotation = ca.vertcat(ca.horzcat(ca.cos(psi), -ca.sin(psi)), ca.horzcat(ca.sin(psi), ca.cos(psi)))
            net_force = ca.MX.zeros(2, 1)
            net_torque = 0
            for s in range(self.FACE_COUNT):
                midpoint, normal, tangent = face_geometry[s]
                contact = midpoint + location[s, b] * tangent
                applied_force = -force[s, b] * rotation @ normal
                net_force += applied_force
                net_torque += contact[0] * (-force[s, b] * normal[1]) - contact[1] * (-force[s, b] * normal[0])
                push_pose = box[:2, k] + rotation @ (contact + robot_standoff * normal)
                # Componentwise big-M contact-pose condition.
                delta = robot[:, k] - push_pose
                constrain(delta - contact_tolerance - big_m * (1 - face[s, b]), -np.inf, 0.0)
                constrain(delta + contact_tolerance + big_m * (1 - face[s, b]), 0.0, np.inf)

            constrain(box[:2, k + 1] - box[:2, k] - dt * box[3:5, k])
            constrain(box[2, k + 1] - box[2, k] - dt * box[5, k])
            constrain(box[3:5, k + 1] - box[3:5, k] - dt / box_mass * (net_force - linear_damping * box[3:5, k]))
            constrain(box[5, k + 1] - box[5, k] - dt / box_yaw_inertia * (net_torque - angular_damping * box[5, k]))
            constrain(robot[:, k + 1] - robot[:, k] - dt * velocity[:, k])
            constrain(ca.sumsqr(velocity[:, k]), -np.inf, robot_speed_max ** 2)
            # box[3:5,k] (velocity) and box[5,k] (yaw rate) limits, and
            # force/switching >= 0 above, are all plain bounds on raw
            # decision variables -- set on _lbx/_ubx below instead of as
            # general constraint rows (cheaper for the NLP solver: bound
            # constraints need no Jacobian row, unlike a `constrain(...)` call).
            objective += weight["box_position"] * ca.sumsqr(box[:2, k] - goal)
            objective += weight["box_velocity"] * ca.sumsqr(box[3:5, k]) + weight["box_yaw_rate"] * box[5, k] ** 2
            objective += weight["robot_velocity"] * ca.sumsqr(velocity[:, k])

        objective += weight["terminal_box_position"] * ca.sumsqr(box[:2, n] - goal)
        discrete = [False] * self._decision_size
        for index in range(self._face_slice_start, self._free_slice_start + blocks):
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
        self._lbx[self._face_slice_start:self._free_slice_start + blocks] = 0.0
        self._ubx[self._face_slice_start:self._free_slice_start + blocks] = 1.0
        # box velocity (rows 3,4) and yaw rate (row 5) limits, as bounds on
        # the raw box decision variable instead of a per-step `constrain(...)`
        # row -- same feasible set, no Jacobian row needed for a plain bound.
        box_lb = np.tile([-np.inf, -np.inf, -np.inf, -box_speed_max, -box_speed_max, -box_yaw_rate_max], n + 1)
        box_ub = np.tile([np.inf, np.inf, np.inf, box_speed_max, box_speed_max, box_yaw_rate_max], n + 1)
        box_size = 6 * (n + 1)
        self._lbx[:box_size] = box_lb
        self._ubx[:box_size] = box_ub
        # force >= 0 and switching >= 0, likewise raw-variable bounds rather
        # than `constrain(...)` rows.
        force_offset = box_size + 2 * (n + 1) + 2 * n
        self._lbx[force_offset:force_offset + self.FACE_COUNT * blocks] = 0.0
        switching_offset = self._free_slice_start + blocks
        self._lbx[switching_offset:switching_offset + self.FACE_COUNT * blocks] = 0.0

    def plan(self, box_state: BoxPlanarState, robot_xy: Sequence[float], goal_xy: Sequence[float]) -> ContactSchedule:
        """Solve one short-horizon plan from the current measured state."""
        approach = self._approach_if_needed(box_state, robot_xy, goal_xy)
        if approach is not None:
            return approach
        parameters = np.r_[box_state.vector(), np.asarray(robot_xy, dtype=float), np.asarray(goal_xy, dtype=float)]
        used_warm_start = self._last_solution is not None
        x0 = self._warm_start(box_state, robot_xy) if used_warm_start else self._initial_guess(box_state, robot_xy)
        try:
            values, success, status = self._solve(x0, parameters)
        except RuntimeError as error:
            return self._fallback(box_state, robot_xy, goal_xy, str(error))
        if not success and status == "INFEASIBLE" and used_warm_start:
            # The warm start is the previous solve's trajectory shifted one
            # step -- a good guess only while the plan is still valid. Right
            # after a face switch (or any other event that invalidates it),
            # BONMIN can report the neighborhood around that stale guess as
            # infeasible even though the problem itself still has a feasible
            # solution from scratch. Retry cold before giving up on this replan.
            try:
                values, success, status = self._solve(self._initial_guess(box_state, robot_xy), parameters)
                if not success:
                    status = f"cold-retry: {status}"
            except RuntimeError as error:
                return self._fallback(box_state, robot_xy, goal_xy, str(error))
        if not success:
            if status == "LIMIT_EXCEEDED" and self._is_feasible_incumbent(values, parameters):
                self._last_solution = values.copy()
                return self._unpack(values, True, f"incumbent: {status}")
            return self._fallback(box_state, robot_xy, goal_xy, status)
        self._last_solution = values.copy()
        return self._unpack(values, success, status)

    def _solve(self, x0: np.ndarray, parameters: np.ndarray):
        """Run one BONMIN solve in a subprocess and return (values, success, status).

        Raises RuntimeError if the worker itself failed (crashed, or died
        without a result -- e.g. OOM-killed under memory pressure) rather
        than the solver returning a non-optimal status.
        """
        context = mp.get_context("fork")
        parent, child = context.Pipe(duplex=False)
        worker = context.Process(target=_solve_in_quiet_worker, args=(self._solver, {
            "x0": x0, "lbx": self._lbx, "ubx": self._ubx,
            "lbg": self._lbg, "ubg": self._ubg, "p": parameters}, child))
        worker.start()
        child.close()
        # BONMIN's own "bonmin.time_limit" option is not a hard guarantee --
        # it has been observed, rarely, to run for tens of minutes on a
        # particular problem instance well past the configured
        # solver_time_limit_s. Without an external deadline, one such solve
        # blocks the entire receding-horizon loop indefinitely. Give it a
        # generous multiple of the configured limit, then kill the worker.
        deadline = 3 * float(self.config["solver_time_limit_s"]) + 10.0
        if not parent.poll(deadline):
            worker.terminate()
            worker.join(timeout=5.0)
            if worker.is_alive():
                worker.kill()
                worker.join()
            raise RuntimeError(f"solver exceeded hard deadline of {deadline}s (ignored its own time limit)")
        try:
            solved, payload, stats = parent.recv()
        except EOFError as error:
            raise RuntimeError(f"worker died without a result: {error}") from error
        worker.join()
        if not solved:
            raise RuntimeError(payload)
        values = payload
        success, status = bool(stats.get("success", False)), stats.get("return_status", "unknown")
        return values, success, status

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
        """Shift the previous solve's trajectory forward by one replan period.

        Reoptimizing from scratch every replan gave BONMIN no reason to keep
        pursuing a face schedule across replans (only the switching cost
        within a single horizon discouraged flip-flopping), and made solves
        slow enough to occasionally hit the time limit. Continuing from last
        time's plan fixes both. box/robot/velocity are at the fine
        dynamics_dt resolution and shift by one full contact_dt block
        (block_size fine steps) since that's how much time elapses between
        replans; the block-resolution discrete variables shift by one block.
        """
        blocks, block_size = self.horizon, self.block_size
        n = blocks * block_size
        cursor = 0
        box = self._last_solution[cursor:cursor + 6 * (n + 1)].reshape((6, n + 1), order="F"); cursor += 6 * (n + 1)
        robot = self._last_solution[cursor:cursor + 2 * (n + 1)].reshape((2, n + 1), order="F"); cursor += 2 * (n + 1)
        velocity = self._last_solution[cursor:cursor + 2 * n].reshape((2, n), order="F"); cursor += 2 * n
        force = self._last_solution[cursor:cursor + self.FACE_COUNT * blocks].reshape((self.FACE_COUNT, blocks), order="F"); cursor += self.FACE_COUNT * blocks
        location = self._last_solution[cursor:cursor + self.FACE_COUNT * blocks].reshape((self.FACE_COUNT, blocks), order="F"); cursor += self.FACE_COUNT * blocks
        face = self._last_solution[cursor:cursor + self.FACE_COUNT * blocks].reshape((self.FACE_COUNT, blocks), order="F"); cursor += self.FACE_COUNT * blocks
        free = self._last_solution[cursor:cursor + blocks].reshape((1, blocks), order="F"); cursor += blocks
        switching = self._last_solution[cursor:cursor + self.FACE_COUNT * blocks].reshape((self.FACE_COUNT, blocks), order="F")

        def shift(array: np.ndarray, step: int) -> np.ndarray:
            shifted = np.empty_like(array)
            step = min(step, array.shape[1])
            if step < array.shape[1]:
                shifted[:, :-step] = array[:, step:]
            shifted[:, array.shape[1] - step:] = array[:, -1:]
            return shifted

        box, robot = shift(box, block_size), shift(robot, block_size)
        box[:, 0] = box_state.vector()
        robot[:, 0] = np.asarray(robot_xy, dtype=float)
        velocity = shift(velocity, block_size)
        force, location, face, free, switching = (
            shift(force, 1), shift(location, 1), shift(face, 1), shift(free, 1), shift(switching, 1))

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
        blocks = self.horizon
        n = blocks * self.block_size
        # (state_dim, n+1) then column-major flatten, matching the decision
        # vector's layout -- NOT np.tile(vec, (n+1,1)).reshape(-1, order="F"),
        # which scrambles state dimensions across timesteps (verified: for a
        # 6-dim state over 3 steps it produces [1,1,1,2,2,2,...] rather than
        # the intended constant-state-per-timestep [1,2,...,6,1,2,...,6,...]).
        box = np.tile(box_state.vector().reshape(-1, 1), (1, n + 1)).flatten(order="F")
        robot = np.tile(np.asarray(robot_xy, dtype=float).reshape(-1, 1), (1, n + 1)).flatten(order="F")
        velocity = np.zeros(2 * n)
        force = np.zeros(self.FACE_COUNT * blocks)
        location = np.zeros(self.FACE_COUNT * blocks)
        faces = np.zeros(self.FACE_COUNT * blocks)
        free = np.ones(blocks)
        switching = np.zeros(self.FACE_COUNT * blocks)
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
        # Tried adding hysteresis here (engage at reachable_distance, only
        # disengage past a wider threshold) to fix flicker between this
        # heuristic and real planning near the boundary. Measured mixed
        # results across goals (some improved, some got clearly worse --
        # e.g. fixed_lateral 0.272m -> 0.929m) rather than a net improvement,
        # so reverted to the single-threshold version.
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
        blocks, block_size = self.horizon, self.block_size
        n = blocks * block_size
        cursor = 0
        box = values[cursor:cursor + 6 * (n + 1)].reshape((6, n + 1), order="F").T; cursor += 6 * (n + 1)
        robot = values[cursor:cursor + 2 * (n + 1)].reshape((2, n + 1), order="F").T; cursor += 2 * (n + 1)
        velocity = values[cursor:cursor + 2 * n].reshape((2, n), order="F").T; cursor += 2 * n
        force = values[cursor:cursor + self.FACE_COUNT * blocks].reshape((self.FACE_COUNT, blocks), order="F").T; cursor += self.FACE_COUNT * blocks
        location = values[cursor:cursor + self.FACE_COUNT * blocks].reshape((self.FACE_COUNT, blocks), order="F").T; cursor += self.FACE_COUNT * blocks
        faces = values[cursor:cursor + self.FACE_COUNT * blocks].reshape((self.FACE_COUNT, blocks), order="F").T
        face_indices = np.where(np.max(faces, axis=1) > 0.5, np.argmax(faces, axis=1), -1)
        # box/robot/velocity are at the fine dynamics_dt resolution; downsample
        # to one entry per contact_dt block boundary so ContactSchedule keeps
        # the same "one entry per replan-period stage" shape that
        # reference_adapter.pushing_reference() (and everything downstream)
        # already expects, regardless of dynamics_dt/contact_dt.
        block_boundaries = np.arange(0, n + 1, block_size)
        return ContactSchedule(box[block_boundaries], robot[block_boundaries],
                               velocity[block_boundaries[:-1]], force, location, face_indices, success, status)

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
