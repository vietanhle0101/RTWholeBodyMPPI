# Imports
import numpy as np
import copy as cp
import os
import subprocess
import shutil

# MuJoCo's GLFW-based viewer needs a working GLX context. If the NVIDIA
# driver is broken/mismatched on this host (nvidia-smi fails), fall back to
# Mesa's software renderer instead of crashing on context creation.
try:
    _nvidia_ok = subprocess.run(
        ["nvidia-smi"], capture_output=True, timeout=5
    ).returncode == 0
except FileNotFoundError:
    _nvidia_ok = True  # no NVIDIA GPU present; default GLX path is fine
if not _nvidia_ok:
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "mesa")

# Mujoco
import mujoco
import mujoco_viewer
import glfw

# mujoco_viewer's read_pixels() (offscreen mode) never draws markers added via
# add_marker() -- that only happens in the window-mode render() method. This
# replicates read_pixels() but also draws the pending markers, so goal/target
# spheres actually show up in offscreen-rendered frames.
def read_pixels_with_markers(viewer, depth=False):
    viewer.viewport.width, viewer.viewport.height = glfw.get_framebuffer_size(viewer.window)
    mujoco.mjv_updateScene(
        viewer.model, viewer.data, viewer.vopt, viewer.pert, viewer.cam,
        mujoco.mjtCatBit.mjCAT_ALL.value, viewer.scn)
    for marker in viewer._markers:
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

# Controller functions
from whole_body_mppi.control.controllers.mppi_locomanipulation import MPPI_box_push

from whole_body_mppi.utils.tasks import get_task
from whole_body_mppi.utils.transforms import batch_world_to_local_velocity

# Visualization
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import matplotlib.pyplot as plt

# Task
task = 'push_box'
task_data = get_task(task)
import whole_body_mppi as _whole_body_mppi_pkg
whole_body_mppi_folder = os.path.dirname(_whole_body_mppi_pkg.__file__)
model_path = os.path.join(whole_body_mppi_folder, task_data["sim_path"])

# Model visualizer
model_sim = mujoco.MjModel.from_xml_path(model_path)
dt_sim = 0.01
model_sim.opt.timestep = dt_sim
data_sim = mujoco.MjData(model_sim)
viewer = mujoco_viewer.MujocoViewer(model_sim, data_sim, 'offscreen')

# Reset robot (keyframes are defined in the xml)
mujoco.mj_resetDataKeyframe(model_sim, data_sim, 0) # stand position
mujoco.mj_forward(model_sim, data_sim)
q_init = cp.deepcopy(data_sim.qpos) # save reference pose
v_init = cp.deepcopy(data_sim.qvel) # save reference pose
print("Configuration: {}".format(q_init)) # save reference pose

img = viewer.read_pixels()
plt.imshow(img)

# Initialize controller
controller = MPPI_box_push(task=task)
controller.internal_ref = True
controller.reset_planner()

q_curr = cp.deepcopy(data_sim.qpos) # save reference pose
v_curr = cp.deepcopy(data_sim.qvel) # save reference pose
x = np.concatenate([q_curr, v_curr])

# Set desired box location
controller.x_box_ref[0] = 2 # x location in world coordinates (current forward direction)
controller.x_box_ref[1] = 1 # y location in world coordinates (current forward direction)

# Set simulation time
tfinal = 26
tvec = np.linspace(0,tfinal,int(np.ceil(tfinal/dt_sim))+1)

mujoco.mj_resetDataKeyframe(model_sim, data_sim, 0)
mujoco.mj_forward(model_sim, data_sim)

viewer.cam.distance = 3.5
viewer.cam.lookat[:] = [1,1,0]

viewer.add_marker(
        pos=controller.x_box_ref[:3]*1,         # Position of the marker
        size=[0.15, 0.15, 0.15],     # Size of the sphere
        rgba=[1, 1, 0, 1],           # Color of the sphere (red)
        type=mujoco.mjtGeom.mjGEOM_SPHERE, # Specify that this is a sphere
        label=""
    )
viewer.add_marker(
        pos=controller.body_ref[:3]*1,         # Position of the marker
        size=[0.15, 0.15, 0.15],     # Size of the sphere
        rgba=[1, 0, 1, 1],           # Color of the sphere (red)
        type=mujoco.mjtGeom.mjGEOM_SPHERE, # Specify that this is a sphere
        label=""
    )

img = read_pixels_with_markers(viewer)
plt.imshow(img)

q_curr = cp.deepcopy(data_sim.qpos) # save reference pose
v_curr = cp.deepcopy(data_sim.qvel) # save reference pose
x = np.concatenate([q_curr, v_curr])

# Run simulation
anim_imgs = []
sim_inputs = []
x_states = []
for ticks, ti in enumerate(tvec):
    q_curr = cp.deepcopy(data_sim.qpos) # save reference pose
    v_curr = cp.deepcopy(data_sim.qvel) # save reference pose
    x = np.concatenate([q_curr, v_curr])
    
    if ticks%1 == 0:
        u_joints = controller.update(x)  
        
    data_sim.ctrl[:] = u_joints
    mujoco.mj_step(model_sim, data_sim)

    error = np.linalg.norm(np.array(controller.body_ref[:3]) - np.array(data_sim.qpos[7:10]))

    viewer._markers.clear()  # avoid accumulating markers from previous frames
    viewer.add_marker(
        pos=controller.body_ref[:3]*1,         # Position of the marker
        size=[0.15, 0.15, 0.15],     # Size of the sphere
        rgba=[1, 0, 1, 1],           # Color of the sphere (red)
        type=mujoco.mjtGeom.mjGEOM_SPHERE, # Specify that this is a sphere
        label=""
    )
    
    viewer.add_marker(
        pos=controller.x_box_ref[:3]*1,         # Position of the marker
        size=[0.15, 0.15, 0.15],     # Size of the sphere
        rgba=[1, 1, 0, 1],           # Color of the sphere (red)
        type=mujoco.mjtGeom.mjGEOM_SPHERE, # Specify that this is a sphere
        label=""
    )
    if error < 0.1:
        controller.next_goal()

    if ticks % 2 == 1:
        img = read_pixels_with_markers(viewer)
        anim_imgs.append(img)
    sim_inputs.append(u_joints)
    x_states.append(x)

x_states_np = np.array(x_states)

box_state = np.zeros((tvec.shape[0], 13))
robot_state = np.zeros((tvec.shape[0], 37))

# Animation
image_height, image_width = anim_imgs[0].shape[:2]

fig, ax = plt.subplots(figsize=(image_width / 100, image_height / 100), dpi=100)
skip_frames = 10
interval = dt_sim * 1000 * skip_frames
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.set_position([0, 0, 1, 1])

def animate(i):
    ax.clear()
    ax.imshow(anim_imgs[i * skip_frames])
    ax.axis('off')

ani = FuncAnimation(fig, animate, frames=len(anim_imgs) // skip_frames, interval=interval)

# Save the animation. Prefer ffmpeg (mp4); fall back to Pillow (gif) if
# ffmpeg isn't installed on this machine.
if shutil.which('ffmpeg'):
    out_path = f'{task}.mp4'
    ani.save(out_path, writer='ffmpeg', fps=20, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
else:
    out_path = f'{task}.gif'
    ani.save(out_path, writer='pillow', fps=20)
print(f"Saved animation to {out_path}")

