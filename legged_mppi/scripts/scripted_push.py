"""Step 3, item 3: push the box along a hand-made trunk path (what KSOS will produce later).

    .venv-ksos/bin/python legged_mppi/scripts/scripted_push.py --seeds 0-4
    .venv-ksos/bin/python legged_mppi/scripts/scripted_push.py --seeds 0 --view

The trunk walks from its start to a point behind the box (opposite the push direction), then walks
through the box towards the goal, ending where its center would be if the box sat on the goal.
Box cost stays on. Logs to logs/scripted_push/<time>/ with a plot.
"""
import argparse
import os
from datetime import datetime
from multiprocessing import Pool

import mujoco
import numpy as np

DT = 0.01
SCENE = "models/go1/go1_scene_mppi_pyr_push_box_14in.xml"
CONTACT = 0.47   # trunk center -> box center while pushing (measured in the baseline)


def polyline_plan(points, speeds, hold, t_end):
    """Constant-speed segments through points; holds the start for `hold` s and the end afterwards."""
    t, P, V = [0.0], [points[0]], [np.zeros(2)]
    t.append(hold); P.append(points[0]); V.append(np.zeros(2))
    for a, b, v in zip(points[:-1], points[1:], speeds):
        seg = np.linalg.norm(b - a)
        n = max(int(seg / v / DT), 1)
        for i in range(1, n + 1):
            t.append(t[-1] + DT); P.append(a + (b - a) * i / n); V.append((b - a) / seg * v)
    V[-1] = np.zeros(2)
    t.append(max(t_end, t[-1] + DT)); P.append(points[-1]); V.append(np.zeros(2))
    return np.array(t), np.array(P), np.array(V)


def run(seed, args, out_dir):
    import whole_body_mppi
    from whole_body_mppi.control.controllers.mppi_locomanipulation import MPPI_box_push
    from whole_body_mppi.control.reference import BodyReference

    goal = np.array(args.goal)
    ctrl = MPPI_box_push("push_box")
    ctrl.random_generator = np.random.default_rng(seed)
    ctrl.x_box_ref[:2] = goal

    model = mujoco.MjModel.from_xml_path(os.path.join(os.path.dirname(whole_body_mppi.__file__), SCENE))
    model.opt.timestep = DT
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    box0, start = data.qpos[:2].copy(), data.qpos[7:9].copy()
    d = (goal - box0) / np.linalg.norm(goal - box0)
    A = box0 - args.standoff * d          # behind the box, clear of it
    E = goal - CONTACT * d                # trunk center when the box sits on the goal
    t, P, V = polyline_plan([start, A, E], [args.approach_speed, args.push_speed], args.hold, 0)
    t_plan_end = t[-1]
    ref = BodyReference(yaw0=0.0)
    ref.set_plan(t, P, V)

    viewer = None
    if args.view:
        import mujoco_viewer
        viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=True)
        viewer.cam.distance, viewer.cam.elevation = 3.5, -35
        viewer.cam.lookat[:] = [1.0, 0.0, 0.2]

    box, trunk, refs = [], [], []
    t_goal = np.nan
    for k in range(int((t_plan_end + args.settle) / DT)):
        ctrl.ref_traj = ref.horizon(data.time)
        ctrl.gait_scheduler = ctrl.gaits[ref.gait(data.time)]
        data.ctrl[:] = ctrl.update(np.r_[data.qpos, data.qvel])
        mujoco.mj_step(model, data)
        box.append(data.qpos[:7].copy()); trunk.append(data.qpos[7:14].copy()); refs.append(ctrl.ref_traj[0, :3].copy())
        if np.isnan(t_goal) and np.linalg.norm(data.qpos[:2] - goal) < args.tol:
            t_goal = data.time

        if viewer is not None:
            if not viewer.is_alive:
                break
            if k % 2 == 0:
                for pos, rgba in ((ctrl.ref_traj[0, :3], [1, 0, 1, 1]), (np.r_[goal, 0.2], [1, 1, 0, 1])):
                    viewer.add_marker(pos=pos * 1, size=[0.06] * 3, rgba=rgba,
                                      type=mujoco.mjtGeom.mjGEOM_SPHERE, label="")
                viewer.render()
    ctrl.shutdown()
    if viewer is not None and viewer.is_alive:
        viewer.close()

    box, trunk, refs = np.array(box), np.array(trunk), np.array(refs)
    np.savez(os.path.join(out_dir, f"seed{seed}.npz"), box=box, trunk=trunk, ref=refs, plan_t=t, plan_P=P,
             goal=goal, A=A, E=E, t_goal=t_goal, t_plan_end=t_plan_end)
    report(seed, box, trunk, refs, goal, box0, d, t_goal)


def report(seed, box, trunk, refs, goal, box0, d, t_goal):
    n = np.array([-d[1], d[0]])
    pushing = np.linalg.norm(np.gradient(box[:, :2], DT, axis=0), axis=1) > 0.03
    box_off = np.abs((box[:, :2] - box0) @ n)                       # box distance from the planned box line
    trunk_err = np.linalg.norm(trunk[:, :2] - refs[:, :2], axis=1)
    q = box[-1, 3:7]
    print(f"seed {seed}: {'OK ' if np.isfinite(t_goal) else 'FAIL'} t_goal={t_goal:.1f}s  "
          f"final box err {np.linalg.norm(box[-1, :2] - goal):.3f} m  box off-line max {box_off.max():.3f} m  "
          f"trunk-ref err while pushing median {np.median(trunk_err[pushing]) if pushing.any() else np.nan:.3f} m  "
          f"box yaw end {np.degrees(2 * np.arctan2(q[3], q[0])):.0f} deg  min z {trunk[100:, 2].min():.3f}", flush=True)


def plot(out_dir):
    import glob
    import matplotlib.pyplot as plt
    files = sorted(glob.glob(f"{out_dir}/seed*.npz"))
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, f in enumerate(files):
        r = np.load(f)
        c = f"C{i}"
        ax.plot(r["trunk"][:, 0], r["trunk"][:, 1], c=c, lw=1, label=f"trunk {os.path.basename(f)[:-4]}")
        ax.plot(r["box"][:, 0], r["box"][:, 1], c=c, lw=2, ls=":")
    r = np.load(files[0])
    ax.plot(r["plan_P"][:, 0], r["plan_P"][:, 1], "k--", lw=1, label="planned trunk path")
    ax.add_patch(plt.Circle(r["goal"], 0.3, fill=False, ls="--", color="tab:green", label="goal (0.3 m)"))
    ax.add_patch(plt.Rectangle((r["box"][0, 0] - 0.19, r["box"][0, 1] - 0.19), 0.38, 0.38, fill=False, color="gray",
                               label="box start"))
    ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.legend(fontsize=7)
    ax.set_title("solid: trunk, dotted: box center")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/scripted_push.png", dpi=150)
    print("saved", f"{out_dir}/scripted_push.png")


def seed_list(spec):
    if "-" in spec:
        a, b = map(int, spec.split("-"))
        return list(range(a, b + 1))
    return [int(s) for s in spec.split(",")]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--goal", type=float, nargs=2, default=[2.0, -0.75])
    p.add_argument("--approach-speed", type=float, default=0.2)
    p.add_argument("--push-speed", type=float, default=0.1)
    p.add_argument("--standoff", type=float, default=0.75, help="approach point distance behind the box center")
    p.add_argument("--hold", type=float, default=1.0)
    p.add_argument("--settle", type=float, default=5.0)
    p.add_argument("--tol", type=float, default=0.3)
    p.add_argument("--seeds", default="0-4")
    p.add_argument("--jobs", type=int, default=5)
    p.add_argument("--view", action="store_true", help="live viewer; runs only the first seed")
    args = p.parse_args()

    out = os.path.join("logs", "scripted_push", f"{datetime.now():%Y%m%d_%H%M%S}_goal{args.goal[0]:g}_{args.goal[1]:g}")
    os.makedirs(out)
    print("logging to", out)
    seeds = seed_list(args.seeds)
    if args.view:
        run(seeds[0], args, out)
    else:
        with Pool(args.jobs) as pool:
            pool.starmap(run, [(s, args, out) for s in seeds])
    plot(out)
