"""Pedestrian spawning and movement execution (Isaac Sim-dependent half).

Path *planning* (heading/turn/stop sampling, A* routing around the known
obstacle layout) is pure geometry with no Isaac Sim dependency — see
pedestrian_planner.py, kept separate so it's unit-testable on its own (same
split obstacle_randomizer.py uses for statics).

Character instantiation and driving both go through Isaac Sim's Behavior
Agent API (`CreateBehaviorAgentCommand` + `IBehaviorAgent`), confirmed working
end-to-end against the real hallway scene (see test_pedestrian_spawn.py and
the debugging session that root-caused/fixed the pedestrian-teleport bug via
a NavMesh fix in hallway_scene.usd itself — that fix lives in the .usd file,
not here). This supersedes two earlier approaches that didn't work out:
Isaac Sim Replicator Agent (IRA)'s "Actor SDG" panel was never scripted
successfully, and the original move_character_along_commands here drove
characters via raw XFormPrim.set_world_poses() straight-line interpolation
(no real walk-cycle animation, and it deliberately bypassed NavMesh entirely
since baking was — and to a real degree still is, see
pedestrian_planner.py's own module docstring — unreliable in this
environment). plan_pedestrian_commands' A*-routed waypoints are unaffected by
any of this; they're still what feeds movement, just executed differently
now.
"""

import math
import time
from typing import List, Tuple

import numpy as np

from scene_randomizer.pedestrian_planner import (
    plan_pedestrian_commands,
    sample_pedestrian_path,
    sample_pedestrian_paths,
)
from scene_randomizer.scenario_io import PedestrianPathSpec, StaticObstacleSpec

__all__ = [
    "sample_pedestrian_path",
    "sample_pedestrian_paths",
    "spawn_pedestrians",
    "clear_pedestrians",
    "move_character_along_commands",
]

CHARACTER_BASE_URL = (
    "https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/"
    "Isaac/People/Characters"
)
MOTION_LIBRARY_ASSET_URL = (
    "https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/"
    "Isaac/People/MotionLibrary/HumanMotionLibrary.usd"
)
MOTION_LIBRARY_PRIM_PATH = "/World/HumanMotionLibrary"

# The only character asset confirmed to actually work end-to-end (spawn +
# real walk-cycle animation) in this environment as of 2026-09-22. `asset_id`
# on a PedestrianPathSpec is treated as an Isaac People character *folder
# name* (Isaac/People/Characters/<name>/<name>.usd), falling back to this
# default when empty -- which it will be by default, since the
# scene_randomizer_node 'character_pool' ROS param defaults to ['']. If you
# want real character variety, populate character_pool with other verified
# Isaac/People/Characters/<name> folder names.
DEFAULT_CHARACTER_NAME = "male_adult_construction_03"

# Shrinks the spawned character to this fraction of its authored height.
CHARACTER_SCALE = 0.82

# IBehaviorAgent.get_speed()/.set_speed() units aren't documented anywhere
# found so far; inferred from a single observed default reading of 150.0,
# plausible as ~1.5 m/s expressed in cm/s (a normal human walk pace). NOT
# independently verified with a timed walk -- sanity-check this conversion
# (e.g. time a known-distance move_along() call at a known set_speed() value)
# before trusting per-pedestrian speed variation in real data collection.
_SPEED_MPS_TO_AGENT_UNITS = 100.0

_AGENT_REGISTER_TIMEOUT_FRAMES = 90

# Hard safety cap on _drive_concurrently's main loop. Without this, a single
# agent whose task never reports finished (is_task_running() staying True
# indefinitely -- something intermittently observed in this environment while
# debugging the original teleport bug) hangs this call forever, and since it
# blocks via repeated omni.kit.app.get_app().update() calls inside Isaac
# Sim's own process, that hang froze the entire Isaac Sim application, not
# just this script -- confirmed 2026-09-22 when a live run hung and every
# subsequent MCP call, including trivial read-only ones, stopped responding
# and needed an Isaac Sim restart to recover. Any entry still incomplete past
# this point is force-marked done rather than looped on forever.
_MAX_DRIVE_DURATION_S = 180.0

# How often to sample each pedestrian's actual walked position while driving,
# for scene_NN_trajectory.json (see scenario_io.save_pedestrian_trajectories).
# 0.1s matches a reasonable oracle-scoring resolution -- fine enough to catch
# a car/pedestrian near-miss, coarse enough not to bloat the output file.
TRAJECTORY_SAMPLE_INTERVAL_S = 0.1


def _character_asset_url(asset_id: str) -> str:
    name = asset_id or DEFAULT_CHARACTER_NAME
    return f"{CHARACTER_BASE_URL}/{name}/{name}.usd"


def _spawn_character(agent_prim_path: str, asset_id: str, start_position: List[float]) -> str:
    """Spawn one Behavior Agent character, apply CHARACTER_SCALE, and return
    its SkelRoot prim path (required for bh_core.get_agent() -- the outer
    spawn wrapper path silently returns None instead)."""
    import omni.kit.commands
    import omni.usd
    from pxr import Gf, Usd, UsdGeom, UsdSkel

    stage = omni.usd.get_context().get_stage()
    omni.kit.commands.execute(
        "CreateBehaviorAgentCommand",
        character_prim_path=agent_prim_path,
        character_asset_url=_character_asset_url(asset_id),
        motion_library_prim_path=MOTION_LIBRARY_PRIM_PATH,
        motion_library_asset_url=MOTION_LIBRARY_ASSET_URL,
        motion_library_skeleton_rig="Human",
        position=Gf.Vec3d(*start_position),
    )

    agent_prim = stage.GetPrimAtPath(agent_prim_path)
    xformable = UsdGeom.Xformable(agent_prim)
    scale_value = Gf.Vec3f(CHARACTER_SCALE, CHARACTER_SCALE, CHARACTER_SCALE)
    existing_scale_op = next(
        (op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale), None
    )
    if existing_scale_op:
        existing_scale_op.Set(scale_value)
    else:
        xformable.AddScaleOp().Set(scale_value)

    for prim in Usd.PrimRange(agent_prim):
        if prim.IsA(UsdSkel.Root):
            return str(prim.GetPath())
    raise RuntimeError(f"No SkelRoot found under {agent_prim_path} after spawning")


def _group_into_legs(commands: List[dict]) -> List[Tuple[str, object]]:
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


def _start_next_leg(entry: dict) -> None:
    if entry["leg_index"] >= len(entry["legs"]):
        entry["state"] = "done"
        return
    kind, payload = entry["legs"][entry["leg_index"]]
    entry["leg_index"] += 1
    if kind == "walk":
        entry["task_id"] = entry["agent"].move_along(payload)
        entry["state"] = "walking"
    else:
        entry["agent"].idle()
        entry["idle_deadline"] = time.time() + payload
        entry["state"] = "idling"


def _sample_trajectory(entry: dict, elapsed_s: float) -> None:
    agent = entry["agent"]
    pos = agent.get_world_translation()
    try:
        facing = agent.get_facing_direction()
        yaw = math.atan2(facing[1], facing[0])  # radians -- matches
        # training.common.trajectory_io.TrajectorySample's convention (every
        # other yaw in that codebase is radians; this used to be yaw_deg and
        # broke loading real trajectory.json data as a silent
        # degrees-read-as-radians bug -- fixed 2026-09-23).
    except Exception:
        yaw = None
    entry["trajectory"].append({
        "t": round(elapsed_s, 3),
        "x": float(pos[0]),
        "y": float(pos[1]),
        "yaw": yaw,
    })
    entry["last_sample_t"] = elapsed_s


def _drive_concurrently(entries: List[dict]) -> None:
    """Drive every entry's planned legs to completion, all at once (not one
    pedestrian fully finishing before the next starts) -- a scene with
    multiple pedestrians should show them all walking simultaneously, matching
    what the scenario is actually meant to represent. Blocks the calling
    thread, stepping the app one frame at a time via omni.kit.app -- the
    right style for a synchronous ROS2 service-callback context (as opposed
    to test_pedestrian_spawn.py's asyncio-based approach, which assumes a
    Script Editor's already-running event loop).

    Also records each entry's actual walked position at TRAJECTORY_SAMPLE_INTERVAL_S
    into entry["trajectory"], timestamped from a single shared t=0 (when all
    entries' agents finished registering and started moving) so different
    pedestrians' trajectories in the same scene are directly comparable at a
    given time t -- required for oracle_scorer's "is the car near the
    pedestrian at time t" checks.
    """
    import omni.anim.behavior.core as bh_core
    import omni.kit.app

    app = omni.kit.app.get_app()
    bh = bh_core.acquire_interface()

    for _ in range(_AGENT_REGISTER_TIMEOUT_FRAMES):
        pending = [e for e in entries if e["agent"] is None]
        if not pending:
            break
        for e in pending:
            e["agent"] = bh.get_agent(e["skelroot_path"])
        app.update()

    missing = [e["skelroot_path"] for e in entries if e["agent"] is None]
    if missing:
        raise RuntimeError(f"Behavior agent(s) failed to register after spawning: {missing}")

    start_time = time.time()
    for e in entries:
        e.setdefault("trajectory", [])
        e["last_sample_t"] = -1.0
        if e.get("speed_mps") is not None:
            e["agent"].set_speed(e["speed_mps"] * _SPEED_MPS_TO_AGENT_UNITS)
        _start_next_leg(e)
        _sample_trajectory(e, 0.0)

    while any(e["state"] != "done" for e in entries):
        app.update()
        now = time.time()
        elapsed = now - start_time

        if elapsed > _MAX_DRIVE_DURATION_S:
            stuck = [e["skelroot_path"] for e in entries if e["state"] != "done"]
            print(
                f"[pedestrian_randomizer] WARNING: _drive_concurrently exceeded "
                f"{_MAX_DRIVE_DURATION_S}s, abandoning still-incomplete agent(s): {stuck}"
            )
            for e in entries:
                if e["state"] != "done":
                    _sample_trajectory(e, elapsed)
                    e["state"] = "done"
            break

        for e in entries:
            if e["state"] == "done":
                continue
            if elapsed - e["last_sample_t"] >= TRAJECTORY_SAMPLE_INTERVAL_S:
                _sample_trajectory(e, elapsed)
            if e["state"] == "walking" and not e["agent"].is_task_running(e["task_id"]):
                _start_next_leg(e)
            elif e["state"] == "idling" and now >= e["idle_deadline"]:
                _start_next_leg(e)

    end_elapsed = time.time() - start_time
    for e in entries:
        _sample_trajectory(e, end_elapsed)


def spawn_pedestrians(
    specs: List[PedestrianPathSpec],
    hallway_bounds: dict,
    obstacles: List[StaticObstacleSpec] = (),
) -> List[PedestrianPathSpec]:
    """Plan each pedestrian's obstacle-aware command sequence, spawn it via
    the Behavior Agent API, and drive all of them through their routes
    concurrently. Blocks until every pedestrian has completed its planned
    route (matches this function's pre-existing behavioral contract -- see
    module docstring for why that's a synchronous, not fire-and-forget,
    design here)."""
    entries = []
    for i, spec in enumerate(specs):
        character_id = f"pedestrian_{i}"
        agent_prim_path = f"/World/{character_id}"
        commands = plan_pedestrian_commands(spec, list(obstacles), hallway_bounds)
        skelroot_path = _spawn_character(agent_prim_path, spec.asset_id, spec.start_position)

        spec.character_id = character_id
        spec.prim_path = agent_prim_path

        entries.append({
            "spec": spec,
            "skelroot_path": skelroot_path,
            "legs": _group_into_legs(commands),
            "leg_index": 0,
            "state": "waiting",
            "agent": None,
            "task_id": None,
            "idle_deadline": None,
            "speed_mps": spec.speed,
        })

    _drive_concurrently(entries)
    for e in entries:
        e["spec"].trajectory = e["trajectory"]
    return specs


def clear_pedestrians(specs: List[PedestrianPathSpec]) -> None:
    """Remove previously spawned pedestrian characters."""
    import omni.kit.commands

    for spec in specs:
        if spec.prim_path is not None:
            omni.kit.commands.execute("DeletePrims", paths=[spec.prim_path])
        spec.character_id = None
        spec.prim_path = None


def move_character_along_commands(skelroot_path: str, commands: List[dict], speed_mps: float = None) -> List[dict]:
    """Drive one already-spawned Behavior Agent character (identified by its
    SkelRoot prim path -- NOT the outer spawn wrapper path, see
    _spawn_character) through `commands` directly. Blocks until finished.
    Returns the recorded trajectory (see _sample_trajectory).

    Each command is either {"type": "goto", "x", "y", "speed"} (grouped into
    move_along-ready waypoint legs) or {"type": "idle", "duration_s"} (hold
    position). Kept as a standalone entry point for driving a single
    already-spawned character outside of spawn_pedestrians' bulk flow;
    spawn_pedestrians itself drives all its pedestrians concurrently via
    _drive_concurrently rather than calling this once per character.
    """
    entry = {
        "skelroot_path": skelroot_path,
        "legs": _group_into_legs(commands),
        "leg_index": 0,
        "state": "waiting",
        "agent": None,
        "task_id": None,
        "idle_deadline": None,
        "speed_mps": speed_mps,
    }
    _drive_concurrently([entry])
    return entry["trajectory"]
