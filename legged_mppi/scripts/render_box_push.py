#!/usr/bin/env python3
"""Render a video from a trajectory saved by simulate_minlp_box_push.py's
--save-data. Decoupled from simulation on purpose: rendering (offscreen
MuJoCo viewer + video encoding) competes with the MINLP solver for CPU and
memory, and has been the source of OOM kills and multiprocessing-pipe crashes
when run live inside the sim loop.
"""

import argparse
import shutil

import imageio
import mujoco
import numpy as np


def _has_imageio_ffmpeg() -> bool:
    try:
        import imageio_ffmpeg  # noqa: F401
        return True
    except ImportError:
        return False


def read_pixels_with_markers(viewer, markers, depth=False):
    """mujoco_viewer's offscreen read_pixels() never draws add_marker() markers
    -- that only happens in the window-mode render() method. This replicates
    the offscreen render path but also draws the given markers.
    """
    import glfw

    viewer.viewport.width, viewer.viewport.height = glfw.get_framebuffer_size(viewer.window)
    mujoco.mjv_updateScene(
        viewer.model, viewer.data, viewer.vopt, viewer.pert, viewer.cam,
        mujoco.mjtCatBit.mjCAT_ALL.value, viewer.scn)
    for marker in markers:
        viewer._add_marker_to_scene(marker)
    mujoco.mjr_render(viewer.viewport, viewer.scn, viewer.ctx)
    shape = glfw.get_framebuffer_size(viewer.window)
    if depth:
        rgb_img = np.zeros((shape[1], shape[0], 3), dtype=np.uint8)
        depth_img = np.zeros((shape[1], shape[0], 1), dtype=np.float32)
        mujoco.mjr_readPixels(rgb_img, depth_img, viewer.viewport, viewer.ctx)
        return (np.flipud(rgb_img), np.flipud(depth_img))
    img = np.zeros((shape[1], shape[0], 3), dtype=np.uint8)
    mujoco.mjr_readPixels(img, None, viewer.viewport, viewer.ctx)
    return np.flipud(img)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Path to the .npz saved by simulate_minlp_box_push.py --save-data.")
    parser.add_argument("--out", default=None, help="Output video path (default: <data>.mp4/.gif).")
    parser.add_argument("--fps", type=float, default=30.0, help="Output video frame rate.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (2.0 = 2x real time).")
    args = parser.parse_args()

    log = np.load(args.data, allow_pickle=True)
    time = log["time"]
    qpos = log["qpos"]
    goal_xy = log["goal_xy"]
    body_ref_xy = log["body_ref_xy"]
    sim_path = str(log["sim_path"])

    dt = float(time[1] - time[0]) if len(time) > 1 else 0.002
    # Pick a playback stride so the output plays at (approximately) the
    # requested speed at the requested fps, rather than dumping one video
    # frame per logged simulation step (which -- at a 0.002s physics step --
    # produced a ~10x slow-motion video the first time this was tried).
    stride = max(1, round(1.0 / (dt * args.fps * args.speed)))

    model = mujoco.MjModel.from_xml_path(sim_path)
    data = mujoco.MjData(model)

    import mujoco_viewer
    viewer = mujoco_viewer.MujocoViewer(model, data, "offscreen")
    viewer.cam.distance = 3.5
    viewer.cam.lookat[:] = [1, 1, 0]

    if args.out is not None:
        out_path = args.out
    elif shutil.which("ffmpeg") or _has_imageio_ffmpeg():
        out_path = args.data.rsplit(".", 1)[0] + ".mp4"
    else:
        out_path = args.data.rsplit(".", 1)[0] + ".gif"
    writer = imageio.get_writer(out_path, fps=args.fps)

    for i in range(0, len(time), stride):
        data.qpos[:] = qpos[i]
        mujoco.mj_forward(model, data)
        markers = [
            dict(pos=np.r_[goal_xy, 0.15], size=[0.05, 0.05, 0.15],
                 rgba=[1, 1, 0, 1], type=mujoco.mjtGeom.mjGEOM_SPHERE, label=""),
            dict(pos=np.r_[body_ref_xy[i], 0.15], size=[0.05, 0.05, 0.15],
                 rgba=[1, 0, 1, 1], type=mujoco.mjtGeom.mjGEOM_SPHERE, label=""),
        ]
        writer.append_data(read_pixels_with_markers(viewer, markers))

    writer.close()
    print(f"Saved animation to {out_path} ({len(range(0, len(time), stride))} frames at {args.fps} fps)")


if __name__ == "__main__":
    main()
