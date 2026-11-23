"""Render a logged run_push.py seed to an mp4 at real-time speed (offscreen, no window).

    .venv-ksos/bin/python legged_mppi/scripts/render_push.py logs/run_push/<run>/seed0.npz

Writes seed0.mp4 next to the log. Pink: trunk reference, yellow: goal (and the goal pose, with a yaw goal). Needs a log with `qpos`
(runs from before the full state was logged cannot be rendered).
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen on the GPU; set MUJOCO_GL=osmesa for CPU

import imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from run_push import DT, SCENE  # noqa: E402


def render(path, fps=50, width=1280, height=720, goal_yaw=None):
    import whole_body_mppi
    r = np.load(path)
    assert "qpos" in r, f"{path} has no qpos: rerun run_push.py to log the full state"
    Q, refs, goal = r["qpos"], r["ref"], r["goal"]
    model = mujoco.MjModel.from_xml_path(os.path.join(os.path.dirname(whole_body_mppi.__file__), SCENE))
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height, width)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 4.0, -35, 90
    cam.lookat[:] = np.r_[(Q[0, :2] + goal) / 2, 0.2]  # between the box start and the goal

    every = max(int(round(1 / (fps * DT))), 1)  # sim steps per frame
    out = os.path.splitext(path)[0] + ".mp4"
    with imageio.get_writer(out, fps=1 / (every * DT), codec="libx264", quality=8) as writer:
        for k in range(0, len(Q), every):
            data.qpos[:] = Q[k]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            scn = renderer.scene
            for pos, rgba in ((refs[k], [1, 0, 1, 1]), (np.r_[goal, 0.2], [1, 1, 0, 1])):
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.06, 0, 0]),
                                    np.asarray(pos, float), np.eye(3).ravel(), np.array(rgba, np.float32))
                scn.ngeom += 1
            if goal_yaw is not None:  # the goal pose: a flat square the size of the box
                c, s = np.cos(goal_yaw), np.sin(goal_yaw)
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_BOX, np.array([0.19, 0.19, 0.003]),
                                    np.r_[goal, 0.003], np.array([c, -s, 0, s, c, 0, 0, 0, 1.0]),
                                    np.array([1, 1, 0, 0.5], np.float32))
                scn.ngeom += 1
            writer.append_data(renderer.render())
    print(f"saved {out} ({len(Q) * DT:.1f} s at {1 / (every * DT):.0f} fps)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("log", help="a seed*.npz from run_push.py")
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--goal-yaw", type=float, help="draw the goal pose at this yaw [deg] (default: the logged one)")
    args = p.parse_args()
    yaw = args.goal_yaw
    if yaw is None and np.isfinite(np.load(args.log).get("goal_yaw", np.nan)):
        yaw = float(np.load(args.log)["goal_yaw"])
    render(args.log, args.fps, goal_yaw=None if yaw is None else np.radians(yaw))
