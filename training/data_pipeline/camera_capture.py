"""Render/save the camera frame the JetRacer's onboard camera would see at a
decision instant -- the VLM/YOLO arms' actual training input.

Depends on Isaac Sim's `isaacsim.*` APIs, same caveat as obstacle_randomizer.py
/ pedestrian_randomizer.py / car_pose.py: only importable inside Isaac Sim's
own Python process, and only meaningful with a live stage + Play active.

CAMERA_PRIM_PATH confirmed 2026-09-24 directly against the live hallway
stage: /leatherback_ROS/Rigid_Bodies/Chassis has three Camera prims --
Camera_Left, Camera_Right, Camera_Chase. Camera_Chase sits ~1m BEHIND and
~0.5m ABOVE the chassis (world x more negative, z higher, than base_link) --
a third-person spectator/debug view, not a real sensor. Camera_Left and
Camera_Right both sit forward of the chassis center at bumper height
(z=0.24-0.37m), consistent with a forward-facing onboard pair -- confirmed
by rendering an actual frame from Camera_Left: a first-person hallway view
with a pedestrian's legs and a downstream obstacle both visible, exactly the
VLM/YOLO arms' expected input. Camera_Left is the one in use (Camera_Right
exists as a spare/stereo option, not currently read by anything).

dataset_builder.py consumes this as an optional
capture_camera_frame(decision) -> image_path callable
(build_dataset_for_scene's `capture_camera_frame` param) -- pass `capture`
directly, no wrapping needed.
"""

import math
import os

import numpy as np

from training.common.dataset_schema import DecisionPoint

CAMERA_PRIM_PATH = "/leatherback_ROS/Rigid_Bodies/Chassis/Camera_Left"
CAMERA_RESOLUTION = (640, 480)
# How many app-update ticks to wait after moving the car/camera before a
# render is guaranteed to have propagated through -- confirmed live that
# get_rgba() reliably returns real data by tick ~4-10 after initialize(),
# and returns None (not stale data) before that, so padding is cheap safety,
# not guesswork.
RENDER_SETTLE_TICKS = 15

# One Camera object is reused across every capture() call in a run (moving
# it is much cheaper than re-attaching a render product + rgb annotator
# every time) -- module-level cache, populated lazily on first use.
_camera = None


def _get_camera():
    global _camera
    if _camera is not None:
        return _camera
    from isaacsim.sensors.camera import Camera

    _camera = Camera(prim_path=CAMERA_PRIM_PATH, resolution=CAMERA_RESOLUTION)
    _camera.initialize()
    return _camera


def capture(decision: DecisionPoint, output_dir: str = "/home/itri/jetracer_nav/training_data/camera_frames") -> str:
    """Move the JetRacer to (decision.car_x, decision.car_y, decision.car_yaw),
    render one frame from its onboard camera, save it as a PNG under
    `output_dir`, and return the saved path.

    Synchronous on purpose (dataset_builder.py's build_dataset_for_scene
    calls this as a plain callable, not awaited) -- internally pumps the app
    update loop itself via omni.kit.app rather than requiring the caller to
    be an async context.
    """
    import imageio
    import omni.kit.app
    import omni.timeline
    from isaacsim.core.prims import SingleArticulation
    from tf_transformations import quaternion_from_euler

    from scene_randomizer.car_pose import JETRACER_PRIM_PATH

    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        timeline.play()

    articulation = SingleArticulation(JETRACER_PRIM_PATH)
    articulation.initialize()
    qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, decision.car_yaw)
    articulation.set_world_pose(
        position=np.array([decision.car_x, decision.car_y, 0.0]),
        orientation=np.array([qw, qx, qy, qz]),
    )
    articulation.set_linear_velocity(np.zeros(3))
    articulation.set_angular_velocity(np.zeros(3))
    articulation.set_joint_velocities(np.zeros(articulation.num_dof))

    camera = _get_camera()
    app = omni.kit.app.get_app()

    async def _render_and_get_frame():
        rgba = None
        for _ in range(RENDER_SETTLE_TICKS):
            await app.next_update_async()
            rgba = camera.get_rgba()
            if rgba is not None and rgba.size > 0:
                break
        return rgba

    import asyncio

    loop_result = {}

    async def _run():
        loop_result["rgba"] = await _render_and_get_frame()

    task = asyncio.ensure_future(_run())
    # capture() is called synchronously from build_dataset_for_scene's decision
    # loop -- so we can't `await` here without turning capture() itself into
    # an async function and rippling that up through dataset_builder.py.
    # Driving the app loop directly via manual app.update() pumping keeps
    # this a plain synchronous callable, matching the
    # Callable[[DecisionPoint], str] signature build_dataset_for_scene
    # already expects. Hard-capped, not a bare `while not task.done()` --
    # generating_scene.md documents a real prior incident where an
    # analogous unbounded per-frame loop froze Isaac Sim outright and needed
    # a manual kill -9 to recover; RENDER_SETTLE_TICKS*4 is generous slack
    # over the ~4-10 ticks get_rgba() needs in practice.
    max_pump_ticks = RENDER_SETTLE_TICKS * 4
    for _ in range(max_pump_ticks):
        if task.done():
            break
        app.update()
    else:
        raise RuntimeError(
            f"camera_capture: render task did not complete within {max_pump_ticks} app.update() ticks"
        )

    rgba = loop_result.get("rgba")
    if rgba is None or rgba.size == 0:
        raise RuntimeError(
            f"camera_capture: no valid frame from {CAMERA_PRIM_PATH} after "
            f"{RENDER_SETTLE_TICKS} render ticks at decision t={decision.t}"
        )

    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(
        output_dir, f"scene{decision.scene_index:02d}_t{decision.t:.2f}_x{decision.car_x:.2f}_y{decision.car_y:.2f}.png"
    )
    imageio.imwrite(image_path, rgba)
    return image_path
