"""test_pedestrian_spawn.py

Full-pipeline test in the real hallway_scene.usd: read the current obstacle
layout straight off the live stage ("get the map"), plan an obstacle-aware
pedestrian route with pedestrian_planner.py's existing A* (including a random
turn or stop event -- "plan a path"), then spawn and drive a character along
that exact route using Isaac's real Behavior Agent API ("use this path for
the pedestrian") -- CreateBehaviorAgentCommand + IBehaviorAgent.move_along/
idle. See project memory "pedestrian_animation_solved_behavior_agent_api" for
why this replaced the earlier manual skeletal-animation approach, and
"pedestrian_animation..." / pedestrian_planner.py for how the route itself
already avoids obstacles.

This reuses the actual production planning code (scene_randomizer's
pedestrian_planner.py) unmodified -- if this test works, the same
obstacle-read + spawn + drive logic is what should go into
pedestrian_randomizer.py's spawn_pedestrians/move_character_along_commands.

Run by pasting into Isaac Sim's Script Editor with hallway_scene.usd open and
Play active.
"""

import asyncio
import random
import sys
import time
import traceback
import types

import numpy as np
import omni.anim.behavior.core as bh_core
import omni.kit.app
import omni.kit.commands
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdSkel

CHARACTER_SCALE = 0.75  # shrink the spawned character to 75% of its authored height

# --- make scene_randomizer / jetracer_navigation importable from Isaac Sim's
# own Python interpreter, and stub out tf_transformations (a ROS2-only
# package that obstacle_randomizer.py imports at module level for its
# spawn_static_obstacles' quaternion math -- not something this script needs,
# since it spawns via CreateBehaviorAgentCommand instead, but the import is
# still transitively triggered by `import scene_randomizer.pedestrian_planner`) ---
_WS_SRC = "/home/itri/jetracer_nav/jetracer_dynamic_nav_ws/src"
for _pkg_dir in ("scene_randomizer", "jetracer_navigation"):
    _path = f"{_WS_SRC}/{_pkg_dir}"
    if _path not in sys.path:
        sys.path.insert(0, _path)

if "tf_transformations" not in sys.modules:
    _stub = types.ModuleType("tf_transformations")
    _stub.quaternion_from_euler = lambda *a, **k: (0.0, 0.0, 0.0, 1.0)
    sys.modules["tf_transformations"] = _stub

# Isaac Sim's Script Editor runs in one long-lived Python process, so once
# scene_randomizer's modules are imported once they stay cached in
# sys.modules -- re-running this script would keep reusing that stale
# in-memory version instead of picking up edits made to pedestrian_planner.py
# / obstacle_randomizer.py / scenario_io.py on disk. Drop them before
# importing so every run reflects the current file contents.
for _mod_name in list(sys.modules):
    if _mod_name == "scene_randomizer" or _mod_name.startswith("scene_randomizer."):
        del sys.modules[_mod_name]

from scene_randomizer.pedestrian_planner import plan_pedestrian_commands, sample_pedestrian_path
from scene_randomizer.scenario_io import StaticObstacleSpec

# y_min/y_max corrected 2026-09-23 via dense raycast (51 x-positions x 6
# heights) against real wall collision geometry (was -1.16/1.16, the outer
# wall reference -- see scene_randomizer_node.py's hallway_bounds param for
# the full measurement, including why the wall profile isn't flat).
HALLWAY_BOUNDS = {"x_min": -7.93, "x_max": 7.93, "y_min": -0.98, "y_max": 0.94}

CHARACTER_ASSET_URL = (
    "https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/"
    "Isaac/People/Characters/male_adult_construction_03/male_adult_construction_03.usd"
)
MOTION_LIBRARY_ASSET_URL = (
    "https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/"
    "Isaac/People/MotionLibrary/HumanMotionLibrary.usd"
)
AGENT_PRIM_PATH = "/World/TestPedestrian"
MOTION_LIBRARY_PRIM_PATH = "/World/HumanMotionLibrary"


def read_live_obstacles(stage: Usd.Stage) -> list:
    """"Get the map": build a StaticObstacleSpec list from whatever is
    actually under /World/hallway_obstacles on the live stage right now,
    rather than assuming we still have the original Python objects from
    whatever process spawned them."""
    from isaacsim.core.prims import XFormPrim

    obstacles_scope = stage.GetPrimAtPath("/World/hallway_obstacles")
    if not obstacles_scope.IsValid():
        print("[test_pedestrian_spawn] No /World/hallway_obstacles found -- planning with zero obstacles.")
        return []

    obstacles = []
    for child in obstacles_scope.GetChildren():
        prim_path = str(child.GetPath())
        position, _orientation = XFormPrim(prim_path).get_world_poses()
        obstacles.append(StaticObstacleSpec(
            asset_path="(read from stage)",
            position=[float(position[0][0]), float(position[0][1]), float(position[0][2])],
            yaw_deg=0.0,
        ))
    print(f"[test_pedestrian_spawn] Read {len(obstacles)} obstacle(s) from the live stage.")
    return obstacles


# Matches obstacle_randomizer.spawn_static_obstacles' position=[xx, yy, 0.0] --
# a downward raycast was tried first, but it kept hitting the scene's default
# `GroundPlane` collision prim (a leftover flat physics plane sitting well
# above the visible corridor floor, at z~1.835 regardless of x/y -- confirmed
# by getting the identical z at two different x probes) before ever reaching
# the real floor mesh, which is occluded from directly above. Obstacles
# already spawn correctly at a hardcoded z=0.0, so just match that.
HALLWAY_FLOOR_Z = 0.0


def spawn_pedestrian(stage: Usd.Stage, x: float, y: float) -> str:
    z = HALLWAY_FLOOR_Z
    omni.kit.commands.execute(
        "CreateBehaviorAgentCommand",
        character_prim_path=AGENT_PRIM_PATH,
        character_asset_url=CHARACTER_ASSET_URL,
        motion_library_prim_path=MOTION_LIBRARY_PRIM_PATH,
        motion_library_asset_url=MOTION_LIBRARY_ASSET_URL,
        motion_library_skeleton_rig="Human",
        position=Gf.Vec3d(x, y, z),
    )

    # Scale down before the behavior agent registers/measures the character,
    # so the smaller size is what it picks up from the start. Reuse an
    # existing scale xformOp if the payload already has one instead of
    # blindly adding a second one (which can conflict in xformOpOrder).
    agent_prim = stage.GetPrimAtPath(AGENT_PRIM_PATH)
    xformable = UsdGeom.Xformable(agent_prim)
    scale_value = Gf.Vec3f(CHARACTER_SCALE, CHARACTER_SCALE, CHARACTER_SCALE)
    existing_scale_op = next(
        (op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale), None
    )
    if existing_scale_op:
        existing_scale_op.Set(scale_value)
    else:
        xformable.AddScaleOp().Set(scale_value)
    print(f"[test_pedestrian_spawn] xformOpOrder after scale: {[op.GetOpName() for op in xformable.GetOrderedXformOps()]}")

    skelroot_path = None
    for prim in Usd.PrimRange(stage.GetPrimAtPath(AGENT_PRIM_PATH)):
        if prim.IsA(UsdSkel.Root):
            skelroot_path = str(prim.GetPath())
            break
    if skelroot_path is None:
        raise RuntimeError(f"No SkelRoot found under {AGENT_PRIM_PATH} after spawning")
    return skelroot_path


async def wait_for_agent(prim_path: str, tries: int = 30):
    bh = bh_core.acquire_interface()
    for _ in range(tries):
        agent = bh.get_agent(prim_path)
        if agent:
            return agent
        await omni.kit.app.get_app().next_update_async()
    return None


async def wait_for_task(agent) -> None:
    app = omni.kit.app.get_app()
    task_id = agent.get_action_task_id()
    while agent.is_task_running(task_id):
        await app.next_update_async()


async def wait_seconds(duration_s: float) -> None:
    app = omni.kit.app.get_app()
    deadline = time.time() + duration_s
    while time.time() < deadline:
        await app.next_update_async()


def group_into_legs(commands: list) -> list:
    """Group consecutive "goto" commands (between "idle" breaks) into
    move_along-ready waypoint arrays, one (kind, payload) tuple per leg."""
    legs = []
    current_leg = []
    for command in commands:
        if command["type"] == "goto":
            current_leg.append([command["x"], command["y"], 0.0])
        elif command["type"] == "idle":
            if current_leg:
                legs.append(("walk", np.array(current_leg, dtype=np.float32)))
                current_leg = []
            legs.append(("idle", command["duration_s"]))
    if current_leg:
        legs.append(("walk", np.array(current_leg, dtype=np.float32)))
    return legs


def _world_translate(stage: Usd.Stage, prim_path: str) -> str:
    """Read a prim's actual world-space translation (its xformOps only) off
    the stage right now. Z is printed first and alone so it's never cut off
    by the Console panel's width, unlike a plain tuple repr."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return "prim not valid"
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    t = matrix.ExtractTranslation()
    return f"z={float(t[2]):.3f} (x={float(t[0]):.3f}, y={float(t[1]):.3f})"


def _world_bbox_z(stage: Usd.Stage, prim_path: str) -> str:
    """Read the prim's actual RENDERED world-space bounding box Z range.

    Unlike _world_translate (which only reflects the prim's own xformOps),
    this reflects the real visual/skinned mesh position -- for a skeletal
    character, the joints' animated pose can carry its own offset that a
    plain xformOp translate check would never see. Comparing the two tells
    us whether a "teleport to the roof" is an xformOp problem or a
    skeleton/animation-pose problem.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return "prim not valid"
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=True)
    rng = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    return f"z_min={rng.GetMin()[2]:.3f}, z_max={rng.GetMax()[2]:.3f}"


async def run_test():
    try:
        stage = omni.usd.get_context().get_stage()

        # Diagnostics: sanity-check the scene's actual coordinate frame before
        # touching anything, since the pedestrian keeps landing above the
        # floor even at z=0 -- see if the hallway/GroundPlane/NavMesh prims
        # carry a hidden Z offset that we're not accounting for.
        for diag_path in ("/World/hallway", "/World/GroundPlane", "/World/NavMeshVolume"):
            print(f"[test_pedestrian_spawn] DIAG world translate of {diag_path}: {_world_translate(stage, diag_path)}")

        # 1. get the map
        obstacles = read_live_obstacles(stage)
        if obstacles:
            print(f"[test_pedestrian_spawn] DIAG first obstacle world position: {obstacles[0].position}")

        # 2. plan a path -- A*-routed around those obstacles, with a random
        # turn-or-stop event baked in by the existing planner
        rng = random.Random()
        spec = sample_pedestrian_path(rng, HALLWAY_BOUNDS, character_id="test_pedestrian")
        commands = plan_pedestrian_commands(spec, obstacles, HALLWAY_BOUNDS)
        print(
            f"[test_pedestrian_spawn] Planned {len(commands)} command(s), heading={spec.heading}, "
            f"turn_event={spec.turn_event}, stop_event={spec.stop_event}"
        )

        # 3. use this path for the pedestrian
        start_x, start_y, _start_z = spec.start_position
        print(f"[test_pedestrian_spawn] Spawning at start=({start_x:.2f}, {start_y:.2f})...")
        skelroot_path = spawn_pedestrian(stage, start_x, start_y)
        print(f"[test_pedestrian_spawn] Spawned, SkelRoot={skelroot_path}. Waiting for behavior agent...")
        print(f"[test_pedestrian_spawn] DIAG xformOp translate right after spawn: "
              f"{_world_translate(stage, AGENT_PRIM_PATH)}")
        print(f"[test_pedestrian_spawn] DIAG RENDERED bbox right after spawn: "
              f"{_world_bbox_z(stage, skelroot_path)}")

        agent = await wait_for_agent(skelroot_path)
        if agent is None:
            print("[test_pedestrian_spawn] ERROR: Failed to acquire behavior agent after spawning.")
            return
        print(f"[test_pedestrian_spawn] DIAG xformOp translate once agent is ready: "
              f"{_world_translate(stage, AGENT_PRIM_PATH)}")
        print(f"[test_pedestrian_spawn] DIAG RENDERED bbox once agent is ready: "
              f"{_world_bbox_z(stage, skelroot_path)}")
        print(f"[test_pedestrian_spawn] DIAG agent.get_world_translation(): {agent.get_world_translation()}")
        print(f"[test_pedestrian_spawn] Agent ready, walking planned route ({len(group_into_legs(commands))} legs)...")

        for i, (kind, payload) in enumerate(group_into_legs(commands)):
            if kind == "walk":
                print(f"[test_pedestrian_spawn] Leg {i}: walking {len(payload)} waypoint(s)...")
                task_id = agent.move_along(payload)
                print(f"[test_pedestrian_spawn] Leg {i}: task_id={task_id}")
                print(f"[test_pedestrian_spawn] DIAG RENDERED bbox right after move_along() call: "
                      f"{_world_bbox_z(stage, skelroot_path)}")
                await wait_for_task(agent)
                print(f"[test_pedestrian_spawn] DIAG RENDERED bbox after leg {i} finished: "
                      f"{_world_bbox_z(stage, skelroot_path)}")
            else:
                print(f"[test_pedestrian_spawn] Leg {i}: idling for {payload:.1f}s...")
                agent.idle()
                await wait_seconds(payload)

        print("[test_pedestrian_spawn] Done -- pedestrian completed its planned, obstacle-aware route.")
    except Exception:
        print("[test_pedestrian_spawn] EXCEPTION:")
        traceback.print_exc()


asyncio.ensure_future(run_test())
