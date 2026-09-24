"""Pick the (car pose, t) decision instants to generate training examples
from, given a take's recorded/analytic trajectories.

Car pose at each decision instant is currently meant to be synthesized by
walking the hallway's long axis at a fixed step, with a random lateral
offset and a random instant within the pedestrian's route -- there's no real
car-driving policy in this loop yet (see plan.md's recovery-behavior section
for where an actual closed-loop controller eventually plugs in). This is
deliberately decoupled from how the car will eventually be driven: directly
sampling car poses gives a much faster way to generate diverse (state,
candidates, label) tuples than simulating a full closed-loop run per
example, and is a standard approach for pretraining trajectory-scoring
models.

TEMPLATE -- fill in the bodies below. Two gotchas already found by smoke
testing against candidate_path_generator.py, worth keeping in mind:
  - A sampled car pose can land inside a known obstacle's footprint, which
    the A* planner rejects outright ("not standable") -- reject-and-resample
    (or push-away, like pedestrian_planner._clear_of_obstacles) against
    `static_obstacle_footprints` before accepting a point.
  - Keep sampled y inset from the walls (DEFAULT_LATERAL_MARGIN_M), same
    idea as PEDESTRIAN_WALL_CLEARANCE_M in pedestrian_planner.py.
"""

import math
import random
from typing import List, Optional

from training.common.dataset_schema import DecisionPoint
from training.common.trajectory_io import SceneTrajectories

DEFAULT_STEP_M = 1.0  # spacing between sampled car positions along the hallway
DEFAULT_LATERAL_MARGIN_M = 0.3  # keep sampled y away from the walls
CAR_CLEARANCE_M = 0.25  # matches oracle_scorer.CAR_COLLISION_RADIUS_M
CLEAR_Y_TRIES = 10  # give up and accept whatever the last try was after this many rejections


def _inside_footprint(x: float, y: float, footprint: dict, margin: float) -> bool:
    """True if (x, y) falls within `footprint`'s footprint_min/footprint_max
    box, inflated by `margin` on each side."""
    if footprint['footprint_min'][0] - margin < x and footprint['footprint_max'][0] + margin > x:
        if footprint['footprint_min'][1] - margin < y and footprint['footprint_max'][1] + margin > y:
            return True
    return False


def _sample_clear_y(
    x: float, y_min: float, y_max: float, obstacle_footprints: List[dict], rng: random.Random,
) -> float:
    """Reject-and-resample a y at this x until it's clear of every known
    obstacle footprint (or give up gracefully after a few tries -- a car
    pose that ends up inside an obstacle anyway should degrade to zero
    candidates at that one decision point, not crash the run)."""
    tries = 0
    random_y = rng.uniform(y_min, y_max)
    while tries < CLEAR_Y_TRIES:
        random_y = rng.uniform(y_min, y_max)
        if not any(_inside_footprint(x, random_y, f, CAR_CLEARANCE_M) for f in obstacle_footprints):
            return random_y
        tries += 1

    return random_y

def sample_decision_points(
    scene_index: int,
    hallway_bounds: dict,
    trajectories: SceneTrajectories,
    goal_xy: tuple,
    static_obstacle_footprints: Optional[List[dict]] = None,
    step_m: float = DEFAULT_STEP_M,
    rng: Optional[random.Random] = None,
) -> List[DecisionPoint]:
    """Sample decision points along the hallway's long axis, one every
    `step_m`, each paired with a random instant `t` within the take's
    duration -- so different car positions see the pedestrian at different
    points along its route, not all frozen at t=0.

    Returns a list of DecisionPoint (see dataset_schema.py) -- scene_index,
    t, car_x/car_y/car_yaw, goal_x/goal_y.
    """
    rng = rng or random.Random()
    static_obstacle_footprints = static_obstacle_footprints or []

    x_min, x_max = hallway_bounds["x_min"], hallway_bounds["x_max"]
    y_min = hallway_bounds["y_min"] + DEFAULT_LATERAL_MARGIN_M
    y_max = hallway_bounds["y_max"] - DEFAULT_LATERAL_MARGIN_M
    goal_x, goal_y = goal_xy

    # A scene can have more than one pedestrian (see
    # scene_randomizer.pedestrian_randomizer.spawn_pedestrians, which drives
    # every pedestrian in a take concurrently) -- sample t across the
    # longest one's duration, so t can range across whichever pedestrian is
    # still moving latest. A pedestrian that's already finished by some t
    # just resolves to its last known position via ActorTrajectory.pose_at's
    # nearest-neighbor lookup, which is a reasonable "it's standing there
    # now" behavior.
    pedestrians = [a for a in trajectories.actors if a.actor_id.startswith("pedestrian")]
    duration = max((p.duration() for p in pedestrians), default=0.0)

    points = []
    x = x_min
    while x <= x_max:
        y = _sample_clear_y(x, y_min, y_max, static_obstacle_footprints, rng)
        t = rng.uniform(0.0, duration) if duration > 0 else 0.0
        # Face the car generally toward the goal -- candidate_path_generator
        # biases its search around this heading, it isn't the final answer.
        yaw = 0.0 if goal_x >= x else math.pi
        points.append(
            DecisionPoint(
                scene_index=scene_index, t=t, car_x=x, car_y=y, car_yaw=yaw,
                goal_x=goal_x, goal_y=goal_y,
            )
        )
        x += step_m

    return points

