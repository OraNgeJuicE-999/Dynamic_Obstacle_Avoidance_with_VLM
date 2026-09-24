"""Actor trajectory capture: turn a take into ActorTrajectory samples.

Two sources, same output shape (trajectory_io.ActorTrajectory) -- everything
downstream (decision_sampler, candidate_path_generator, oracle_scorer) is
built and testable against either one, so the pedestrian-animation work
happening elsewhere (see project memory "pedestrian_animation_solved_behavior_agent_api"
and test_pedestrian_spawn.py) doesn't block the rest of this pipeline:

  1. trajectory_from_commands -- analytic fallback, integrates
     pedestrian_planner.plan_pedestrian_commands' planned waypoint+speed
     sequence directly. No Isaac Sim needed, usable today.
  2. record_live_trajectory -- the real thing: polls an actor's actual live
     pose in Isaac Sim during a take. STUBBED. The exact polling API depends
     on how pedestrian spawning/driving settles (IBehaviorAgent's
     get_world_translation(), or the car's own localized pose once it's
     driving), which is still being finalized in test_pedestrian_spawn.py.
     Fill this in once that's confirmed working end-to-end -- it should
     mirror test_pedestrian_spawn.py's wait_for_task loop, appending a
     TrajectorySample from agent.get_world_translation() each frame instead
     of just waiting.
"""

import math
from typing import List

from training.common.trajectory_io import ActorTrajectory, TrajectorySample


def trajectory_from_commands(actor_id: str, commands: List[dict], start_t: float = 0.0) -> ActorTrajectory:
    """Integrate a planned command sequence into (t, x, y, yaw) samples.

    `commands` is pedestrian_planner.plan_pedestrian_commands' output:
    {"type": "goto", "x", "y", "speed"} / {"type": "idle", "duration_s"}.
    Walks each "goto" leg at constant speed in a straight line between
    consecutive waypoints.

    This assumes the plan executes exactly as planned -- no animation
    timing drift, no runtime deviation. That's a reasonable stand-in for
    building/testing the rest of the pipeline before record_live_trajectory
    is ready, but should be swapped for the real recorded trajectory once
    that's available, since actual playback won't match this exactly.
    """
    samples: List[TrajectorySample] = []
    t = start_t
    prev_x, prev_y, prev_yaw = None, None, 0.0

    for command in commands:
        if command["type"] == "goto":
            x, y = command["x"], command["y"]
            if prev_x is None:
                samples.append(TrajectorySample(t=t, x=x, y=y, yaw=prev_yaw))
                prev_x, prev_y = x, y
                continue

            dx, dy = x - prev_x, y - prev_y
            distance = math.hypot(dx, dy)
            if distance < 1e-9:
                prev_x, prev_y = x, y
                continue

            yaw = math.atan2(dy, dx)
            speed = max(command["speed"], 1e-3)
            t += distance / speed
            samples.append(TrajectorySample(t=t, x=x, y=y, yaw=yaw))
            prev_x, prev_y, prev_yaw = x, y, yaw

        elif command["type"] == "idle":
            t += command["duration_s"]
            if prev_x is not None:
                samples.append(TrajectorySample(t=t, x=prev_x, y=prev_y, yaw=prev_yaw))

    return ActorTrajectory(actor_id=actor_id, samples=samples)


def record_live_trajectory(actor_id: str, agent, dt: float) -> ActorTrajectory:
    """Poll `agent`'s real live pose in Isaac Sim every `dt` seconds for the
    duration of its current task -- the ground-truth counterpart to
    trajectory_from_commands' analytic approximation.

    NOT YET IMPLEMENTED: depends on the pedestrian spawn/drive interface
    settling in test_pedestrian_spawn.py / pedestrian_randomizer.py first --
    specifically what `agent` actually is (an IBehaviorAgent, a prim path,
    something else) and how task completion gets detected once the
    known "teleports above the floor" bug there is resolved. The signature
    here (actor_id, agent, dt) is a reasonable guess at the shape, not a
    commitment -- expect to adjust it once that's settled.
    """
    raise NotImplementedError(
        "record_live_trajectory needs the pedestrian spawn/drive interface finalized first -- "
        "use trajectory_from_commands() as a stand-in until then."
    )
