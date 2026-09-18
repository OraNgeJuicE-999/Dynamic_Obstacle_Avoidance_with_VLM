"""Pedestrian spawning and movement execution (Isaac Sim-dependent half).

Path *planning* (heading/turn/stop sampling, A* routing around the known
obstacle layout) is pure geometry with no Isaac Sim dependency — see
pedestrian_planner.py, kept separate so it's unit-testable on its own (same
split obstacle_randomizer.py uses for statics).

Two responsibilities live here:
  1. spawn_pedestrians/clear_pedestrians: instantiate/remove characters on the
     stage. Character *instantiation* still isn't wired to a verified API —
     Isaac Sim Replicator Agent (IRA) is what's actually been used to load a
     character in this project so far, driven manually through its "Actor
     SDG" UI panel; a scripted equivalent hasn't been confirmed yet.
  2. move_character_along_commands: drive an *already-spawned* character prim
     through a planned command sequence directly, bypassing IRA's `patrol`
     routine and the NavMesh entirely. This exists because NavMesh baking has
     proven unreliable in this environment (bakes hang regardless of volume
     size or excluded geometry) — since plan_pedestrian_commands already
     routes every waypoint around the known obstacle layout, no runtime
     pathfinding is needed here, just straight-line interpolation between
     already-safe waypoints.
"""

import math
from typing import List

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


def spawn_pedestrians(
    specs: List[PedestrianPathSpec],
    hallway_bounds: dict,
    obstacles: List[StaticObstacleSpec] = (),
) -> List[PedestrianPathSpec]:
    """Plan each pedestrian's obstacle-aware command sequence and instantiate it.

    NOTE: character *instantiation* (spawning a new character prim of type
    `spec.asset_id`) is NOT verified here — see module docstring. That is the
    only missing piece: command planning (plan_pedestrian_commands) and
    execution (move_character_along_commands) are both fully implemented.
    """
    for i, spec in enumerate(specs):
        character_id = f"pedestrian_{i}"
        commands = plan_pedestrian_commands(spec, list(obstacles), hallway_bounds)

        # TODO (verify against your Isaac Sim version's character-loading API,
        # e.g. IRA's scene_assembly loader): spawn a character of type
        # `spec.asset_id` at `spec.start_position`, get back its prim_path,
        # then: move_character_along_commands(prim_path, commands)

        spec.character_id = character_id

    return specs


def clear_pedestrians(specs: List[PedestrianPathSpec]) -> None:
    """Remove previously spawned pedestrian characters.

    NOTE: same caveat as spawn_pedestrians — the actual removal call isn't
    verified against a specific character-loading API here.
    """
    for spec in specs:
        if spec.character_id is not None:
            # TODO (verify against your Isaac Sim version's character API):
            #   remove/despawn the character identified by spec.character_id
            pass
        spec.character_id = None


def move_character_along_commands(prim_path: str, commands: List[dict]) -> None:
    """Drive an already-spawned character prim through `commands` directly.

    Each command is either {"type": "goto", "x", "y", "speed"} (straight-line
    move at constant speed, facing the direction of travel) or
    {"type": "idle", "duration_s"} (hold position). Blocks until finished,
    stepping the app one frame at a time via omni.kit.app so the viewport
    stays live during the move — this is the technique for procedural
    animation from a Script Editor script without registering a proper
    per-frame update callback.
    """
    import time

    import omni.kit.app
    from isaacsim.core.prims import XFormPrim

    app = omni.kit.app.get_app()
    prim = XFormPrim(prim_path)
    positions, _orientations = prim.get_world_poses()
    current_pos = np.array(positions[0], dtype=float)

    for command in commands:
        if command["type"] == "idle":
            deadline = time.time() + command["duration_s"]
            while time.time() < deadline:
                app.update()
            continue

        target = np.array([command["x"], command["y"], current_pos[2]])
        delta = target - current_pos
        distance = float(np.linalg.norm(delta[:2]))
        if distance < 1e-6:
            current_pos = target
            continue

        # (w, x, y, z) — Isaac Sim's XFormPrim quaternion order, yaw about Z.
        yaw = math.atan2(delta[1], delta[0])
        orientation = np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])

        speed = max(command["speed"], 1e-3)
        duration = distance / speed
        start_time = time.time()
        start_pos = current_pos.copy()

        while True:
            t = min((time.time() - start_time) / duration, 1.0)
            pos = start_pos + delta * t
            prim.set_world_poses(
                positions=np.array([pos]),
                orientations=np.array([orientation]),
            )
            app.update()
            if t >= 1.0:
                break

        current_pos = target
