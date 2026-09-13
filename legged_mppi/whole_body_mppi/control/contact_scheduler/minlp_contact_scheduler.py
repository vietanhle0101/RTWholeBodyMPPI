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
        # The previous *issued* mode is a solver parameter on the next call.
        # Without it, the first face in every receding-horizon problem can
        # change for free even though that change becomes a large yaw step at
        # the Go1 interface.
        self._previous_face_index = -1
        # The approach heuristic is necessary when the desired standoff is
        # outside the finite MINLP horizon.  It must not, however, override a
        # newly established physical push because a single measured position
        # happens to straddle that reachability boundary.
        self._phase = "approach"
        self._committed_face_index = -1
        self._contact_hold_replans = 0
        self._release_violations = 0
        self._last_standoff_error = np.inf
        self._min_contact_hold_replans = max(
            0, int(np.ceil(float(self.config.get("contact_min_hold_s", 1.0)) / self.replan_period)))
        self._contact_release_margin = float(self.config.get("contact_release_margin", 0.10))
        self._contact_release_replans = max(1, int(self.config.get("contact_release_replans", 3)))
        if self._contact_release_margin < 0.0:
            raise ValueError("contact_release_margin must be non-negative")
        # The planner's planar robot state is the trunk origin, whereas the
        # desired physical contact is made by the forward bumper.  Keeping
        # these offsets explicit makes the outer staging policy agree with
        # the MINLP's root-level `robot_standoff` constraint.
        self._bumper_forward_offset = float(self.config.get(
            "bumper_forward_offset", self.config["robot_standoff"]))
        self._bumper_clearance = float(self.config.get(
            "bumper_clearance", float(self.config["robot_standoff"]) - self._bumper_forward_offset))
        if self._bumper_forward_offset < 0.0 or self._bumper_clearance < 0.0:
            raise ValueError("bumper_forward_offset and bumper_clearance must be non-negative")
        if not np.isclose(float(self.config["robot_standoff"]),
                          self._bumper_forward_offset + self._bumper_clearance):
            raise ValueError("robot_standoff must equal bumper_forward_offset + bumper_clearance")
        self._align_yaw_tolerance = np.deg2rad(float(self.config.get("align_yaw_tolerance_deg", 12.5)))
        self._align_creep_speed = float(self.config.get("align_creep_speed", 0.05))
        self._approach_face_score_margin = float(self.config.get("approach_face_score_margin", 0.10))
        self._approach_reselect_cooldown_replans = max(
            0, int(np.ceil(float(self.config.get("approach_reselect_cooldown_s", 0.6)) / self.replan_period)))
        self._approach_reselect_cooldown = 0

    @property
    def diagnostics(self) -> Mapping[str, object]:
        """State needed to diagnose approach/contact transitions in a log."""
        return {
            "phase": self._phase,
            "committed_face_index": self._committed_face_index,
            "standoff_error": self._last_standoff_error,
            "contact_hold_replans": self._contact_hold_replans,
            "release_violations": self._release_violations,
        }

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
        previous_face = ca.MX.sym("previous_face", self.FACE_COUNT)
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
            preceding_face = previous_face if b == 0 else face[:, b - 1]
            for s in range(self.FACE_COUNT):
                constrain(switching[s, b] - face[s, b] + preceding_face[s], 0.0, np.inf)
                constrain(switching[s, b] + face[s, b] - preceding_face[s], 0.0, np.inf)
            for s in range(self.FACE_COUNT):
                # force[s,b] >= 0 and switching[s,b] >= 0 are plain bounds on
                # raw decision variables -- set on _lbx below instead of as
                # general constraint rows.
                constrain(force[s, b] - force_max * face[s, b], -np.inf, 0.0)
                constrain(location[s, b] - half_face_lengths[s] * face[s, b], -np.inf, 0.0)
                constrain(location[s, b] + half_face_lengths[s] * face[s, b], 0.0, np.inf)
            switch_weight = (weight.get("initial_contact_switch", 8.0 * weight["contact_switch"])
                             if b == 0 else weight["contact_switch"])
            objective += weight["force"] * ca.sumsqr(force[:, b]) + switch_weight * ca.sum1(switching[:, b])

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
                                  "g": ca.vertcat(*g), "p": ca.vertcat(initial, goal, previous_face)},
                                 {"discrete": discrete, "print_time": False,
                                  "bonmin": {"time_limit": float(c["solver_time_limit_s"]),
                                             "bb_log_level": 0, "nlp_log_level": 0,
                                             "print_level": 0}})
        # Used to sanity-check incumbents accepted from a time-limited solve
        # (see plan()) -- BONMIN reports "success" only for a proven-optimal
        # termination, but on LIMIT_EXCEEDED it still returns its best
        # feasible incumbent when the search found one at all.
        self._g_func = ca.Function("g", [decision, ca.vertcat(initial, goal, previous_face)], [ca.vertcat(*g)])
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

    def plan(self, box_state: BoxPlanarState, robot_xy: Sequence[float], goal_xy: Sequence[float],
             robot_yaw: Optional[float] = None) -> ContactSchedule:
        """Solve one short-horizon plan from the current measured state."""
        approach = self._approach_if_needed(box_state, robot_xy, goal_xy, robot_yaw)
        if approach is not None:
            return self._finalize_schedule(approach, box_state, robot_xy, goal_xy)
        parameters = np.r_[box_state.vector(), np.asarray(robot_xy, dtype=float), np.asarray(goal_xy, dtype=float),
                           self._previous_face_vector()]
        used_warm_start = self._last_solution is not None
        x0 = self._warm_start(box_state, robot_xy) if used_warm_start else self._initial_guess(box_state, robot_xy)
        try:
            values, success, status = self._solve(x0, parameters)
        except RuntimeError as error:
            return self._finalize_schedule(self._fallback(box_state, robot_xy, goal_xy, str(error)),
                                           box_state, robot_xy, goal_xy)
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
                return self._finalize_schedule(self._fallback(box_state, robot_xy, goal_xy, str(error)),
                                               box_state, robot_xy, goal_xy)
        if not success:
            if status == "LIMIT_EXCEEDED" and self._is_feasible_incumbent(values, parameters):
                self._last_solution = values.copy()
                return self._finalize_schedule(self._unpack(values, True, f"incumbent: {status}"),
                                               box_state, robot_xy, goal_xy)
            return self._finalize_schedule(self._fallback(box_state, robot_xy, goal_xy, status),
                                           box_state, robot_xy, goal_xy)
        self._last_solution = values.copy()
        return self._finalize_schedule(self._unpack(values, success, status), box_state, robot_xy, goal_xy)

    def _previous_face_vector(self) -> np.ndarray:
        """One-hot active face from the prior command; all zero denotes free."""
        previous = np.zeros(self.FACE_COUNT)
        if 0 <= self._previous_face_index < self.FACE_COUNT:
            previous[self._previous_face_index] = 1.0
        return previous

    def _remember_issued_mode(self, schedule: ContactSchedule) -> ContactSchedule:
        """Use the first mode actually handed downstream as the next boundary mode."""
        self._previous_face_index = int(schedule.face_indices[0]) if len(schedule.face_indices) else -1
        return schedule

    def _finalize_schedule(self, schedule: ContactSchedule, box_state: BoxPlanarState,
                           robot_xy: Sequence[float], goal_xy: Sequence[float]) -> ContactSchedule:
        """Track contact hold without overriding an intentional MINLP regrasp."""
        face_index = int(schedule.face_indices[0]) if len(schedule.face_indices) else -1
        if self._phase == "hold":
            # Hold suppresses only the out-of-MINLP reachability heuristic.
            # A free first stage from BONMIN is an intentional regrasp and
            # must reach MPPI unchanged; replacing it with a synthetic,
            # centred rear push discarded the optimized contact offset and
            # prevented lateral progress on shallow goals.
            self._contact_hold_replans += 1

        if face_index >= 0:
            if self._phase != "hold" or face_index != self._committed_face_index:
                self._phase = "hold"
                self._committed_face_index = face_index
                self._contact_hold_replans = 1
                self._release_violations = 0
        return self._remember_issued_mode(schedule)

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

    def _face_geometry(self, face_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return a candidate face's box-frame midpoint and outward normal."""
        midpoints = ((-self.config["box_half_length"], 0.0),
                     (0.0, self.config["box_half_width"]),
                     (0.0, -self.config["box_half_width"]))
        normals = ((-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
        return np.asarray(midpoints[face_index], dtype=float), np.asarray(normals[face_index], dtype=float)

    @staticmethod
    def _rotation(yaw: float) -> np.ndarray:
        return np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])

    def _face_scores(self, yaw: float, direction: np.ndarray) -> np.ndarray:
        """Goal-direction alignment of each face's inward push direction."""
        rotation = self._rotation(yaw)
        normals = np.array([[-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        return np.array([np.dot(-(rotation @ normal), direction) for normal in normals])

    def _face_heading(self, box_yaw: float, face_index: int) -> float:
        """World yaw that points Go1's forward bumper inward through a face."""
        _, normal = self._face_geometry(face_index)
        inward = -(self._rotation(box_yaw) @ normal)
        return float(np.arctan2(inward[1], inward[0]))

    def _standoff_pose(self, box_state: BoxPlanarState, face_index: int) -> tuple[np.ndarray, float]:
        """Root position/yaw that places the physical forward bumper at a face."""
        midpoint, normal = self._face_geometry(face_index)
        box_rotation = self._rotation(box_state.yaw)
        contact = box_state.vector()[:2] + box_rotation @ midpoint
        outward = box_rotation @ normal
        heading = self._face_heading(box_state.yaw, face_index)
        bumper_world = self._rotation(heading) @ np.array([self._bumper_forward_offset, 0.0])
        root = contact + self._bumper_clearance * outward - bumper_world
        return root, heading

    def _select_face(self, box_xy: np.ndarray, yaw: float, robot_xy: np.ndarray, direction: np.ndarray) -> int:
        """Choose the best goal-aligned face, breaking exact ties by travel distance."""
        scores = self._face_scores(yaw, direction)
        candidates = np.flatnonzero(scores >= scores.max() - 1e-9)
        if candidates.size == 1:
            return int(candidates[0])
        rotation = self._rotation(yaw)
        standoffs = []
        for face_index in candidates:
            midpoint, normal = self._face_geometry(int(face_index))
            standoffs.append(box_xy + rotation @ (midpoint + float(self.config["robot_standoff"]) * normal))
        distances = np.linalg.norm(np.asarray(standoffs) - robot_xy, axis=1)
        return int(candidates[np.argmin(distances)])

    @staticmethod
    def _angle_error(target: float, current: float) -> float:
        return float(np.arctan2(np.sin(target - current), np.cos(target - current)))

    def _commit_approach_face(self, face_index: int) -> None:
        """Start staging a new face and discard a trajectory tied to the old one."""
        if face_index == self._committed_face_index:
            return
        self._committed_face_index = face_index
        self._last_solution = None
        self._previous_face_index = -1
        self._approach_reselect_cooldown = self._approach_reselect_cooldown_replans

    def _motion_schedule(self, box_state: BoxPlanarState, robot_xy: np.ndarray, target_xy: np.ndarray,
                         speed: float, status: str, desired_yaw: Optional[float] = None) -> ContactSchedule:
        """Build a kinematically consistent, force-free staging schedule."""
        robot = np.zeros((self.horizon + 1, 2))
        velocity = np.zeros((self.horizon, 2))
        robot[0] = robot_xy
        for k in range(self.horizon):
            delta = target_xy - robot[k]
            distance = np.linalg.norm(delta)
            step = (np.zeros(2) if distance < 1e-9 else
                    delta / distance * min(distance, speed * self.dt))
            robot[k + 1] = robot[k] + step
            velocity[k] = step / self.dt
        box = np.tile(box_state.vector(), (self.horizon + 1, 1))
        return ContactSchedule(box, robot, velocity, np.zeros((self.horizon, self.FACE_COUNT)),
                               np.zeros((self.horizon, self.FACE_COUNT)), -np.ones(self.horizon, dtype=int),
                               True, status, desired_yaw)

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

    def _approach_if_needed(self, box_state: BoxPlanarState, robot_xy: Sequence[float],
                            goal_xy: Sequence[float], robot_yaw: Optional[float] = None) -> Optional[ContactSchedule]:
        """Stage a goal-aligned bumper contact before allowing a push solve.

        This wrapper owns only approach/alignment.  It never geometrically
        reselects a face while in ``hold``: changing an established physical
        push is the separate measured-progress reassessment problem.
        """
        robot_xy = np.asarray(robot_xy, dtype=float)
        goal_xy = np.asarray(goal_xy, dtype=float)
        direction = goal_xy - box_state.vector()[:2]
        if np.linalg.norm(direction) < 1e-6:
            return None
        direction /= np.linalg.norm(direction)
        reachable_distance = self.horizon * self.dt * self.config["robot_speed_max"]

        # Hold only suppresses this outer staging heuristic.  If contact has
        # been lost for long enough, fully clear the old face so fresh
        # geometry—not the stale commitment—selects the next approach.
        if self._phase == "hold":
            held_face = (self._committed_face_index if self._committed_face_index >= 0 else
                         self._select_face(box_state.vector()[:2], box_state.yaw, robot_xy, direction))
            held_target, _ = self._standoff_pose(box_state, held_face)
            self._last_standoff_error = float(np.linalg.norm(held_target - robot_xy))
            release_distance = reachable_distance + self._contact_release_margin
            if (self._contact_hold_replans >= self._min_contact_hold_replans
                    and self._last_standoff_error > release_distance):
                self._release_violations += 1
            else:
                self._release_violations = 0
            if self._release_violations < self._contact_release_replans:
                return None
            self._phase = "approach"
            self._committed_face_index = -1
            self._previous_face_index = -1
            self._contact_hold_replans = 0
            self._release_violations = 0
            self._last_solution = None

        # Revalidate a pre-contact commitment on every replan.  This avoids a
        # reachable but now counterproductive face trapping the robot after
        # box drift or yaw.  Exact/near ties stay committed to avoid chatter.
        scores = self._face_scores(box_state.yaw, direction)
        candidate = self._select_face(box_state.vector()[:2], box_state.yaw, robot_xy, direction)
        committed = self._committed_face_index
        if committed < 0:
            self._commit_approach_face(candidate)
        else:
            if self._approach_reselect_cooldown > 0:
                self._approach_reselect_cooldown -= 1
            current_score = scores[committed]
            candidate_score = scores[candidate]
            should_reselect = (current_score < 0.0 or
                               (candidate != committed and self._approach_reselect_cooldown == 0 and
                                candidate_score > current_score + self._approach_face_score_margin))
            if should_reselect:
                self._phase = "approach"
                self._commit_approach_face(candidate)

        face_index = self._committed_face_index
        target_xy, target_yaw = self._standoff_pose(box_state, face_index)
        self._last_standoff_error = float(np.linalg.norm(target_xy - robot_xy))
        if self._last_standoff_error > reachable_distance:
            self._phase = "approach"
            return self._motion_schedule(box_state, robot_xy, target_xy,
                                         float(self.config["robot_speed_max"]), "free-mode approach")

        # MPPI's yaw command is slew-limited downstream.  Creep rather than
        # engage while it turns, so the trunk bumper—not a swinging front
        # leg—reaches the box first.  Omitting robot_yaw retains the legacy
        # API behavior for non-MuJoCo callers.
        if robot_yaw is not None and abs(self._angle_error(target_yaw, float(robot_yaw))) > self._align_yaw_tolerance:
            self._phase = "align"
            return self._motion_schedule(box_state, robot_xy, target_xy,
                                         self._align_creep_speed, "align", target_yaw)
        return None

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
        """Return a nominal contact reference after a failed solve."""
        h = self.horizon
        box = np.zeros((h + 1, 6)); box[0] = box_state.vector()
        robot = np.zeros((h + 1, 2)); robot[0] = np.asarray(robot_xy, dtype=float)
        velocity = np.zeros((h, 2))
        force = np.zeros((h, self.FACE_COUNT))
        locations = np.zeros((h, self.FACE_COUNT))
        faces = np.zeros(h, dtype=int)
        normals = np.array([[-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        goal_direction = np.asarray(goal_xy, dtype=float) - box[0, :2]
        if np.linalg.norm(goal_direction) < 1e-6:
            return ContactSchedule(box, robot, velocity, force, locations, -np.ones(h, dtype=int), False, status)
        goal_direction /= np.linalg.norm(goal_direction)
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
