"""KSOS + MPPI (Global-MPPI) on the reduced trunk push model.

One `solve` is one control cycle of global_mppi.simulation.deterministic.run_interactive, without the
logging: for each pass, roll out KSOS candidates (LSE-smoothed; uniform, or from a trained flow in the
first pass), fit the kernel bandwidth, solve KSOS, then refine with one MPPI rollout pass (best sample,
see refine). The plan is a cubic spline of controls, knots at absolute times `tk`.
"""
import time

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from global_mppi.algs import GlobalMPPI
from global_mppi.simulation.deterministic import neg_lml
from mujoco import mjx
from scipy.optimize import minimize


class KSOSPlanner:
    def __init__(self, task, horizon=4.0, num_knots=8, ksos_samples=64, lse_samples=16, mppi_samples=256,
                 passes=1, seed=0, flow=None, lse_sigma=0.1, mppi_noise=0.3, impl="jax"):
        """flow: a directory from ksos_train_flow.py; its sampler then draws the first-pass candidates.
        lse_sigma, mppi_noise: first-pass LSE smoothing noise and MPPI sampling noise (both 0.3 in
        Global-MPPI; 0.1 LSE was best on the push over 20 seeds, 0.3 and 0.03 had longer stalls).
        impl: MJX backend, "jax" or "warp" (MuJoCo Warp, much faster rollouts in contact)."""
        if impl == "warp":  # 50 line-search iterations: no overflow warnings, and faster than the XML's 10
            task.mj_model.opt.ls_iterations = 50
            task.model = mjx.put_model(task.mj_model, impl="warp")
        sampler = None
        if flow is not None:
            from whole_body_mppi.ksos.flow_sampler import make_flow_sampler
            sampler = make_flow_sampler(flow, task, num_knots, lse_sigma)
        self.ctrl = GlobalMPPI(task, num_samples=mppi_samples, noise_level=0.1, temperature=0.1,
                               spline_type="cubic", plan_horizon=horizon, num_knots=num_knots, iterations=1,
                               ksos_solver="newton-rs", ksos_sampling="uniform" if flow is None else "flow",
                               ksos_num_samples=ksos_samples, elite_num=min(80, ksos_samples),
                               ksos_flow_sampler=sampler)
        self.ctrl.lse_num_samples = lse_samples
        # by default both are the same array; .at makes new ones, so they are set independently
        self.ctrl.lse_smoothing_sigma = self.ctrl.lse_smoothing_sigma.at[0].set(lse_sigma)
        self.ctrl.mppi_noise = self.ctrl.mppi_noise.at[0].set(mppi_noise)
        self.ctrl.ksos_warm_start = True  # the shifted previous plan is one of the KSOS candidates
        self.passes = passes
        self._rollout = jax.jit(self.ctrl.ksos_rollout)
        self._refine = jax.jit(self.refine)
        self._interp = jax.jit(self.ctrl.interp_func)
        kw = {}
        if impl == "warp":  # one contact buffer for the whole batch: the KSOS pass (candidates + LSE samples)
            kw = dict(naconmax=32 * max(ksos_samples * (lse_samples + 1), 2 * mppi_samples), njmax=128)
        self.data = mjx.put_data(task.mj_model, mujoco.MjData(task.mj_model), impl=impl, **kw)
        self.reset(seed)

    def reset(self, seed=0):
        """New episode: zero plan and a fresh rng (keeps the jitted functions)."""
        self.params = self.ctrl.init_params(seed=seed)
        self.last_cost, self.last_time = np.nan, np.nan

    def _state(self, d):
        return self.data.replace(qpos=jnp.array(d.qpos), qvel=jnp.array(d.qvel), ctrl=jnp.array(d.ctrl),
                                 mocap_pos=jnp.array(d.mocap_pos), mocap_quat=jnp.array(d.mocap_quat),
                                 time=jnp.array(d.time, jnp.float32))

    def _fit_sigma(self, p):
        """Kernel bandwidth by GP marginal likelihood, as run_interactive does before each KSOS solve."""
        X = np.asarray(p.knots).reshape(p.lse_cost.shape[0], -1)
        Xn = (X - X.mean(0)) / (X.std(0) + 1e-8)
        yn = np.asarray(p.lse_cost) - float(np.mean(p.lse_cost))
        grid = np.log(np.linspace(0.01, 30.0, 10))
        x0 = grid[int(np.argmin([neg_lml(s, Xn, yn) for s in grid]))]
        res = minimize(neg_lml, x0=np.array([x0]), args=(Xn, yn), method="L-BFGS-B",
                       bounds=[(np.log(1e-2), np.log(30.0))])
        self.ctrl.ksos_sigma = float(np.exp(res.x[0]))

    def refine(self, state, p):
        """MPPI step as GlobalMPPI.safeguarded_optimize samples it (noise around the KSOS mean and around
        the best candidate, sample 0 of each is the start itself), but the plan is the best sample, not
        the softmax mean. Its cost is known from this one rollout pass, so no rollout to check it, and it
        is never worse than either start. (At temperature 0.1 the softmax is nearly an argmin anyway;
        averaging near-ties often gave a worse plan, which then needed the check.)"""
        c = self.ctrl
        tk = jnp.linspace(0.0, c.plan_horizon, c.num_knots) + state.time
        shift = lambda knots: c.interp_func(tk, p.tk, knots[None])[0]
        p = p.replace(tk=tk)
        samples = []
        for start in (shift(p.mean), shift(p.knots[jnp.argmin(p.cand_cost)])):
            knots, sampled = c.sample_knots(p.replace(mean=start))  # same noise for both starts
            samples.append(jnp.clip(knots, c.task.u_min, c.task.u_max))
        knots = jnp.concatenate(samples)
        costs = c.rollout(state, tk, knots).costs.sum(-1)
        best = jnp.argmin(costs)
        return sampled.replace(mean=knots[best], rng=jax.random.split(sampled.rng)[0]), costs[best]

    def solve(self, d):
        """Replan from MuJoCo data d (trunk model layout). Returns the cost of the new plan."""
        t0 = time.time()
        state = self._state(d)
        p = self.params
        for i in range(self.passes):
            p = self._rollout(state, p.replace(iter=i))
            self._fit_sigma(p)
            p = self.ctrl.solve_ksos(p, state)
            p = p.replace(mean=p.ksos_mean)
            p, cost = self._refine(state, p)
        self.params = p
        self.last_cost, self.last_time = float(cost), time.time() - t0
        return self.last_cost

    def controls(self, tq):
        """Planned controls at absolute times tq, (len(tq), nu)."""
        tq = jnp.clip(jnp.asarray(tq, jnp.float32), self.params.tk[0], self.params.tk[-1])
        return np.asarray(self._interp(tq, self.params.tk, self.params.mean[None])[0])

    def rollout(self, m, d):
        """Roll the current plan out from d on a copy, in C MuJoCo at m's timestep.
        Returns the trunk path t (N,), xy (N, 2), world xy velocity (N, 2), yaw (N,)."""
        d2 = mujoco.MjData(m)
        mujoco.mj_copyData(d2, m, d)
        d = d2
        U = self.controls(d.time + m.opt.timestep * np.arange(int(round(self.ctrl.plan_horizon / m.opt.timestep))))
        T, P, V, Y = [d.time], [d.qpos[7:9].copy()], [d.qvel[6:8].copy()], [d.qpos[9]]
        for u in U:
            d.ctrl[:] = u
            mujoco.mj_step(m, d)
            T.append(d.time); P.append(d.qpos[7:9].copy()); V.append(d.qvel[6:8].copy()); Y.append(d.qpos[9])
        return np.array(T), np.array(P), np.array(V), np.array(Y)


def yaw_of(q):
    """Yaw of a wxyz quaternion."""
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class KSOSPushPlanner:
    """Global trunk plan for the Go1 push, same interface as GeometricPushPlanner: plan(t, qpos, qvel) on
    the Go1 state (qpos = [box 7, robot 19], qvel = [box 6, robot 18]); it also returns a heading.

    The Go1 trunk and box are copied into the trunk model, KSOS + MPPI plan its controls, and the plan
    is rolled out in C MuJoCo (10 ms) to give the trunk path (t, P, V, yaw) for BodyReference.
    """

    def __init__(self, goal, done_tol=0.1, rollout_dt=0.01, **planner_kwargs):
        from whole_body_mppi.ksos.task import XML, Go1PushTask
        self.goal, self.done_tol = np.asarray(goal, float), done_tol
        self.ksos = KSOSPlanner(Go1PushTask(), **planner_kwargs)
        self.m = mujoco.MjModel.from_xml_path(XML)
        self.m.opt.timestep = rollout_dt
        self.d = mujoco.MjData(self.m)
        self.mode, self.solve_times = "ksos", []

    def reset(self, seed=0):
        self.ksos.reset(seed)
        self.mode, self.solve_times = "ksos", []

    def plan(self, t, qpos, qvel):
        m, d = self.m, self.d
        d.time = t
        d.qpos[0:7], d.qvel[0:6] = qpos[0:7], qvel[0:6]           # box
        d.qpos[2], d.qvel[2] = 0.19, 0.0                          # resting on the floor
        d.qpos[7:9], d.qpos[9] = qpos[7:9], yaw_of(qpos[10:14])  # trunk xy, yaw
        d.qvel[6:8], d.qvel[8] = qvel[6:8], qvel[11]             # world xy velocity, yaw rate (base z)
        d.ctrl[:] = self.ksos.controls([t])[0]                   # what the previous plan commands now
        d.mocap_pos[0, :2] = self.goal
        mujoco.mj_forward(m, d)
        if np.linalg.norm(qpos[0:2] - self.goal) < self.done_tol:
            self.mode = "done"
            return np.array([t, t + 1.0]), np.repeat(d.qpos[None, 7:9], 2, 0), np.zeros((2, 2)), np.full(2, d.qpos[9])

        self.ksos.solve(d)
        self.solve_times.append(self.ksos.last_time)
        return self.ksos.rollout(m, d)
