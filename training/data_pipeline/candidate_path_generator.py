"""Local, goal-biased A* candidate path generation.

Generates a decision point's full candidate set, per plan.md's "Candidate
path generation" section:
  - generated *locally* against a per-scan occupancy grid, not solved
    globally straight to the goal (the goal may be outside sensed range)
  - only walls are privileged beyond the sensed range (taken from the known
    hallway floorplan); obstacles/the pedestrian are never privileged here --
    candidates can turn out to be dead ends or run into unsensed obstacles,
    that's expected, not a bug
  - candidate generation stays pedestrian-agnostic even during dataset
    construction -- ground truth is never baked into which candidates exist,
    only into how oracle_scorer.py scores them afterward
  - negative examples for training: random-subgoal candidates (very
    different, very bad) and small perturbations of the best-biased
    candidate (harder, near-miss negatives)

Reuses jetracer_navigation's grid A* -- the same planner pedestrian_planner.py
routes pedestrians with -- rather than a separate implementation.

TEMPLATE -- fill in the bodies below. Two gotchas already found by smoke
testing:
  - A goal projected purely by sensing range/angle can land outside
    hallway_bounds (the grid's actual extent) near either end of the
    corridor -- clamp every synthesized goal into bounds before planning
    against it (see _clamp_to_bounds).
  - AStarImplementation.plan() raises ValueError for an out-of-bounds or
    occupied start/goal -- expected for the speculative random-negative
    goals, so catch it in _plan_one and return [] rather than letting it
    propagate and crash the whole generation run.
"""

import math
import random
import sys
from typing import List, Optional

import numpy as np

_WS_SRC = "/home/itri/jetracer_nav/jetracer_dynamic_nav_ws/src"
for _pkg_dir in ("jetracer_navigation",):
    _path = f"{_WS_SRC}/{_pkg_dir}"
    if _path not in sys.path:
        sys.path.insert(0, _path)

from jetracer_navigation.a_star import AStarImplementation 
from jetracer_navigation.utils import pixel_to_world, smooth_path_generation, world_to_pixel  # noqa: E402

from training.common.dataset_schema import CandidatePath, DecisionPoint  # noqa: E402

GRID_RESOLUTION_M = 0.05
CAR_BODY_RADIUS_M = 0.2
GOAL_THRESHOLD_M = 0.15
LOCAL_SENSING_RANGE_M = 3.0  # how far the "local per-scan map" extends around the car
NUM_BIAS_CANDIDATES = 3  # goal-biased candidates, spread across a few angles
NUM_RANDOM_NEGATIVES = 3  # random-subgoal negatives (plan.md)
NUM_PERTURBED_NEGATIVES = 2  # perturbations of the best-biased candidate (plan.md)
PERTURBATION_STD_M = 0.4


class _Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class _Origin:
    def __init__(self, x, y):
        self.position = _Position(x, y)


class _MapInfo:
    """Duck-typed stand-in for nav_msgs/OccupancyGrid's `.info`, matching
    pedestrian_planner.py's own _MapInfo so world_to_pixel/pixel_to_world
    work unmodified here too."""

    def __init__(self, origin_x, origin_y, resolution):
        self.origin = _Origin(origin_x, origin_y)
        self.resolution = resolution


def build_local_grid(
    decision: DecisionPoint,
    hallway_bounds: dict,
    known_obstacle_footprints: List[dict],
    resolution: float = GRID_RESOLUTION_M,
):
    """Build the occupancy grid a candidate is planned against: the full
    hallway extent is always free/walled per hallway_bounds (walls are
    privileged, per module docstring), and only the obstacle footprints
    within LOCAL_SENSING_RANGE_M of the car should be stamped in --
    simulating "only locally-sensed obstacles are known" at this decision
    instant.

    Returns (grid, map_info) -- a numpy uint8 array (1=blocked) and a
    _MapInfo, same convention as pedestrian_planner.build_occupancy_grid.
    """
    x_min, x_max = hallway_bounds["x_min"], hallway_bounds["x_max"]
    y_min, y_max = hallway_bounds["y_min"], hallway_bounds["y_max"]
    width = int(math.ceil((x_max - x_min) / resolution) + 1)
    height = int(math.ceil((y_max - y_min) / resolution) + 1)

    grid = np.zeros((height, width), dtype=np.uint8)
    map_info = _MapInfo(x_min, y_min, resolution)

    for obstacle in known_obstacle_footprints:
        x1 = obstacle["footprint_max"][0]
        y1 = obstacle["footprint_max"][1]
        x2 = obstacle["footprint_min"][0]
        y2 = obstacle["footprint_min"][1]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        distance = math.sqrt((decision.car_x - center_x)**2 + (decision.car_y - center_y)**2)
        if distance > LOCAL_SENSING_RANGE_M:
            continue

        # Same origin (x_min/y_min) for both the lo and hi bound -- using
        # x_max/y_max here would compute an unrelated pixel index, and the
        # +1 pedestrian_planner/obstacle_randomizer's own versions of this
        # rasterization don't add inside the floor/ceil, only afterward
        # (via the +1 in the clamp below).
        x_l = int(math.floor((x2 - x_min) / resolution))
        x_h = int(math.ceil((x1 - x_min) / resolution))
        y_l = int(math.floor((y2 - y_min) / resolution))
        y_h = int(math.ceil((y1 - y_min) / resolution))

        x_l = max(0, x_l)
        x_h = min(width, x_h + 1)
        y_l = max(0, y_l)
        y_h = min(height, y_h + 1)

        # grid is (height, width) -- rows are y, columns are x -- so index
        # [y, x], not [x, y] (matches obstacle_randomizer.py's own
        # grid[py_lo:py_hi, px_lo:px_hi] convention).
        grid[y_l:y_h, x_l:x_h] = 1

    return grid, map_info

def _clamp_to_bounds(x: float, y: float, hallway_bounds: dict, margin: float = GRID_RESOLUTION_M * 2) -> tuple:
    """Clamp (x, y) into the grid's actual extent (inset by `margin`)."""
    x_min, x_max = hallway_bounds["x_min"], hallway_bounds["x_max"]
    y_min, y_max = hallway_bounds["y_min"], hallway_bounds["y_max"]
    xx = min(max(x_min + margin, x), x_max - margin)
    yy = min(max(y_min + margin, y), y_max - margin)
    return [xx, yy]


def _local_goal(decision: DecisionPoint, hallway_bounds: dict, bias_angle_rad: float) -> List[tuple]:
    """Five points LOCAL_SENSING_RANGE_M out from the car, spread across the
    goal direction at -2*bias_angle_rad, -bias_angle_rad, 0, +bias_angle_rad,
    +2*bias_angle_rad -- what makes candidates "biased toward a general
    direction rather than solving straight to the goal" (plan.md). Each is
    clamped into hallway_bounds before returning (see _clamp_to_bounds)."""
    heading = math.atan2(decision.goal_y - decision.car_y, decision.goal_x - decision.car_x)
    h1 = heading - bias_angle_rad * 2
    h2 = heading - bias_angle_rad
    h3 = heading
    h4 = heading + bias_angle_rad
    h5 = heading + bias_angle_rad * 2

    g1 = _clamp_to_bounds(decision.car_x + LOCAL_SENSING_RANGE_M * math.cos(h1), decision.car_y + LOCAL_SENSING_RANGE_M * math.sin(h1), hallway_bounds)
    g2 = _clamp_to_bounds(decision.car_x + LOCAL_SENSING_RANGE_M * math.cos(h2), decision.car_y + LOCAL_SENSING_RANGE_M * math.sin(h2), hallway_bounds)
    g3 = _clamp_to_bounds(decision.car_x + LOCAL_SENSING_RANGE_M * math.cos(h3), decision.car_y + LOCAL_SENSING_RANGE_M * math.sin(h3), hallway_bounds)
    g4 = _clamp_to_bounds(decision.car_x + LOCAL_SENSING_RANGE_M * math.cos(h4), decision.car_y + LOCAL_SENSING_RANGE_M * math.sin(h4), hallway_bounds)
    g5 = _clamp_to_bounds(decision.car_x + LOCAL_SENSING_RANGE_M * math.cos(h5), decision.car_y + LOCAL_SENSING_RANGE_M * math.sin(h5), hallway_bounds)

    return [g1, g2, g3, g4, g5]

def _plan_one(start_xy: tuple, goal_xy: tuple, grid, map_info, resolution: float) -> List[tuple]:
    """Run AStarImplementation from start_xy to goal_xy on `grid`, return
    smoothed world-space waypoints (empty list if no path / invalid
    start-goal -- see module docstring's second gotcha)."""
    start_px = world_to_pixel(start_xy[0], start_xy[1], map_info)
    goal_px = world_to_pixel(goal_xy[0], goal_xy[1], map_info)

    goal_threshold_px = GOAL_THRESHOLD_M / resolution
    inflation_px = max(1, round(CAR_BODY_RADIUS_M / resolution))

    planner = AStarImplementation(
        grid,
        start_px,
        goal_px,
        goal_threshold=goal_threshold_px,
        inflation_radius=inflation_px
    )
    try:
        path, visited_path = planner.plan()
    except ValueError:
        # Start/goal out of bounds or landed on an occupied cell -- expected
        # for the speculative random-negative goals (see module docstring's
        # second gotcha), not a crash.
        return []
    if not path:
        return []

    # smooth_path_generation returns [] for a path shorter than 4 nodes
    # (see its own early-return) -- fall back to the raw (unsmoothed) path
    # in that case instead of silently discarding a legitimately short one.
    smooth_path = smooth_path_generation(path, occupancy_map=planner.inflated_obstacle_map)
    path = smooth_path or path

    return [pixel_to_world(node.coords.x, node.coords.y, map_info) for node in path]

def generate_candidates(
    decision: DecisionPoint,
    hallway_bounds: dict,
    known_obstacle_footprints: List[dict],
    rng: Optional[random.Random] = None,
) -> List[CandidatePath]:
    """Generate this decision point's full candidate set: goal-biased
    candidates (the ones a real deployment would actually consider) plus the
    negative examples plan.md specifies for training (random-subgoal +
    perturbed-best-candidate)."""
    rng = rng or random.Random()
    grid, map_info = build_local_grid(decision, hallway_bounds, known_obstacle_footprints)
    start_xy = (decision.car_x, decision.car_y)

    candidates: List[CandidatePath] = []
    biased_waypoints: List[List[tuple]] = []

    # Spread the 5 goals _local_goal returns across a total +/-45 deg arc
    # around the straight-to-goal heading (2 * bias_angle_rad = the extreme
    # offset, per _local_goal's own docstring).
    goals = _local_goal(decision, hallway_bounds, math.pi / 8)
    for index, goal in enumerate(goals):
        waypoints = _plan_one(start_xy, goal, grid, map_info, GRID_RESOLUTION_M)
        if waypoints:
            candidates.append(CandidatePath(candidate_id=f"biased_{index}", waypoints=waypoints))
            biased_waypoints.append(waypoints)

    for i in range(int(NUM_RANDOM_NEGATIVES)):
        angle = rng.uniform(-math.pi, math.pi)  # radians, not degrees -- math.cos/sin expect radians
        dist = rng.uniform(0.1, LOCAL_SENSING_RANGE_M)
        xx = decision.car_x + dist * math.cos(angle)
        yy = decision.car_y + dist * math.sin(angle)
        goal_xy = _clamp_to_bounds(xx, yy, hallway_bounds)

        waypoints = _plan_one(start_xy, goal_xy, grid, map_info, GRID_RESOLUTION_M)
        if waypoints:
            candidates.append(
                CandidatePath(candidate_id=f"random_neg_{i}", waypoints=waypoints, is_negative_example=True)
            )

    if biased_waypoints:
        base = biased_waypoints[len(biased_waypoints) // 2]
        for i in range(int(NUM_PERTURBED_NEGATIVES)):
            perturbed = [
                (x + rng.gauss(0, PERTURBATION_STD_M), y + rng.gauss(0, PERTURBATION_STD_M))
                for x, y in base
            ]
            candidates.append(
                CandidatePath(candidate_id=f"perturbed_neg_{i}", waypoints=perturbed, is_negative_example=True)
            )

    return candidates
