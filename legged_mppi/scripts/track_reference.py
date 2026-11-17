"""Step 3: how well does their whole-body MPPI track a time-varying trunk reference?

    .venv-ksos/bin/python legged_mppi/scripts/track_reference.py --speed 0.1 0.2 --seeds 0-2
    .venv-ksos/bin/python legged_mppi/scripts/track_reference.py --speed 0.1 --seeds 0 --view

Uses the push-box scene with the box moved 10 m away and its cost switched off, so only tracking is
tested. The reference holds the start for --hold s, then runs a straight line of --length m at
--speed m/s in direction --angle (deg, 0 = the robot's initial heading), then holds the end.
Logs to logs/track/<time>_line/ (one npz per speed and seed) and saves a plot.
"""
import argparse
import os
from datetime import datetime
from multiprocessing import Pool

import mujoco
import numpy as np

DT = 0.01
SCENE = "models/go1/go1_scene_mppi_pyr_push_box_14in.xml"


def line_plan(speed, length, angle, hold, t_end):
    t = np.arange(0, t_end, DT)
    d = np.array([np.cos(angle), np.sin(angle)])
    s = np.clip((t - hold) * speed, 0, length)
    P = s[:, None] * d
    V = np.where(((t > hold) & (s < length))[:, None], speed * d, 0.0)
    return t, P, V


def run(seed, speed, args, out_dir):
    import whole_body_mppi
    from whole_body_mppi.control.controllers.mppi_locomanipulation import MPPI_box_push
    from whole_body_mppi.control.reference import BodyReference

    ctrl = MPPI_box_push("push_box")
    ctrl.random_generator = np.random.default_rng(seed)
    ctrl.Q_box = np.zeros((3, 3))
    if args.gait != "auto" and args.gait not in ctrl.gaits:  # a gait file name, e.g. MED/walking_gait_raibert_MED_0_2_10cm_100hz
        from whole_body_mppi.control.controllers.mppi_locomanipulation import GAIT_DIR
        from whole_body_mppi.control.gait_scheduler.scheduler import GaitScheduler
        ctrl.gaits[args.gait] = GaitScheduler(gait_path=os.path.join(GAIT_DIR, args.gait + ".tsv"), name=args.gait)

    model = mujoco.MjModel.from_xml_path(os.path.join(os.path.dirname(whole_body_mppi.__file__), SCENE))
    model.opt.timestep = DT
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[:3] = [10, 10, 0.19]  # box out of the way
    mujoco.mj_forward(model, data)

    angle = np.radians(args.angle)
    t_end = args.hold + args.length / speed + args.settle
    t, P, V = line_plan(speed, args.length, angle, args.hold, t_end + 1)
    ref = BodyReference(yaw0=0.0, yaw_rate=args.yaw_rate)
    ref.set_plan(t, P + data.qpos[7:9], V)

    viewer = None
    if args.view:
        import mujoco_viewer
        viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=True)
        viewer.cam.distance, viewer.cam.elevation = 3.0, -25
        line_end = np.r_[P[-1] + data.qpos[7:9], 0.27]

    trunk, refs = [], []
    for k in range(int(t_end / DT)):
        ctrl.ref_traj = ref.horizon(data.time)
        if args.gait == "auto":
            ctrl.gait_scheduler = ctrl.gaits[ref.gait(data.time)]
        else:
            ctrl.gait_scheduler = ctrl.gaits[args.gait if ref.moving(data.time) else "in_place"]
        data.ctrl[:] = ctrl.update(np.r_[data.qpos, data.qvel])
        mujoco.mj_step(model, data)
        trunk.append(data.qpos[7:14].copy())
        refs.append(ctrl.ref_traj[0, :7].copy())

        if viewer is not None:
            if not viewer.is_alive:
                break
            if k % 2 == 0:
                viewer.cam.lookat[:] = [data.qpos[7], data.qpos[8], 0.2]  # follow the trunk
                for pos, rgba in ((ctrl.ref_traj[0, :3], [1, 0, 1, 1]), (line_end, [1, 1, 0, 1])):
                    viewer.add_marker(pos=pos * 1, size=[0.05] * 3, rgba=rgba,
                                      type=mujoco.mjtGeom.mjGEOM_SPHERE, label="")
                viewer.render()
    ctrl.shutdown()
    if viewer is not None and viewer.is_alive:
        viewer.close()

    trunk, refs = np.array(trunk), np.array(refs)
    np.savez(os.path.join(out_dir, f"v{speed:g}_seed{seed}.npz"), trunk=trunk, ref=refs, speed=speed,
             angle=angle, hold=args.hold, length=args.length)
    m = metrics(trunk, refs, angle)
    print(f"v={speed:g} seed {seed}: cross-track max {m['xt_max']:.3f} m  lag median {m['lag']:.3f} m  "
          f"end err {m['end']:.3f} m  heading err max {m['yaw_max']:.1f} deg  min z {m['z_min']:.3f}", flush=True)


def yaw_of(q):
    return np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def metrics(trunk, refs, angle):
    d = np.array([np.cos(angle), np.sin(angle)])
    n = np.array([-d[1], d[0]])
    rel = trunk[:, :2] - refs[0, :2]                         # relative to the line's start
    moving = np.linalg.norm(np.diff(refs[:, :2], axis=0, prepend=refs[:1, :2]), axis=1) > 0
    lag = (refs[:, :2] - trunk[:, :2]) @ d                   # how far the trunk is behind the reference
    yaw_err = np.degrees(np.abs((yaw_of(trunk[:, 3:7]) - yaw_of(refs[:, 3:7]) + np.pi) % (2 * np.pi) - np.pi))
    return dict(xt_max=np.abs(rel @ n).max(), lag=np.median(lag[moving]) if moving.any() else 0.0,
                end=np.linalg.norm(trunk[-1, :2] - refs[-1, :2]), yaw_max=yaw_err[100:].max(),
                z_min=trunk[100:, 2].min())


def plot(out_dir):
    import glob
    import matplotlib.pyplot as plt
    files = sorted(glob.glob(f"{out_dir}/*.npz"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for f in files:
        r = np.load(f)
        d = np.array([np.cos(r["angle"]), np.sin(r["angle"])])
        tt = np.arange(len(r["trunk"])) * DT
        ax1.plot(r["trunk"][:, 0], r["trunk"][:, 1], lw=1, label=os.path.basename(f)[:-4])
        ax2.plot(tt, (r["ref"][:, :2] - r["trunk"][:, :2]) @ d, lw=1)
    r = np.load(files[0])
    ax1.plot(r["ref"][:, 0], r["ref"][:, 1], "k--", lw=1, label="reference")
    ax1.set_aspect("equal"); ax1.set_xlabel("x [m]"); ax1.set_ylabel("y [m]"); ax1.legend(fontsize=7)
    ax2.set_xlabel("time [s]"); ax2.set_ylabel("along-track lag [m]"); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/tracking.png", dpi=150)
    print("saved", f"{out_dir}/tracking.png")


def seed_list(spec):
    if "-" in spec:
        a, b = map(int, spec.split("-"))
        return list(range(a, b + 1))
    return [int(s) for s in spec.split(",")]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--speed", type=float, nargs="+", default=[0.1, 0.2])
    p.add_argument("--length", type=float, default=1.5)
    p.add_argument("--angle", type=float, default=0.0)
    p.add_argument("--hold", type=float, default=1.0)
    p.add_argument("--settle", type=float, default=3.0)
    p.add_argument("--yaw-rate", type=float, default=1.0)
    p.add_argument("--gait", default="auto",
                   help="auto (pick by planned speed), walk, walk_0.2, trot, or a gait file under gaits/ without .tsv")
    p.add_argument("--seeds", default="0-2")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--view", action="store_true", help="live viewer; runs only the first speed and seed")
    args = p.parse_args()

    out = os.path.join("logs", "track", f"{datetime.now():%Y%m%d_%H%M%S}_line{args.angle:g}_{os.path.basename(args.gait)}")
    os.makedirs(out)
    print("logging to", out)
    jobs = [(s, v, args, out) for v in args.speed for s in seed_list(args.seeds)]
    if args.view:
        run(*jobs[0])
    else:
        with Pool(args.jobs) as pool:
            pool.starmap(run, jobs)
    plot(out)
