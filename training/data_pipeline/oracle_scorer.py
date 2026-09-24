"""Oracle scoring: the shared, deterministic "answer key" every arm trains
against (plan.md's "Training data / oracle labeling").

Given a candidate path and the scenario's ground truth (static obstacle
footprints + the pedestrian's actual/planned trajectory), scores it on a
weighted multi-category rubric -- collision-freeness (static AND dynamic),
path complexity, goal progress -- and combines them into one scalar training
target.

This is bird's-eye/privileged scoring: it uses the full ground-truth
pedestrian trajectory (including positions relative to the decision instant
that a camera/lidar snapshot alone couldn't give you), not whatever a sensor
could actually see at that instant -- that's the whole point of an oracle
label. What each arm can actually sense (camera frames, etc.) is a separate
concern (camera_capture.py); this module never touches that, and never
influences which candidates exist (candidate_path_generator.py) -- only how
they're scored afterward.

Category weights (SCORE_WEIGHTS) are NOT tuned yet (plan.md's "Open / not
yet decided") -- pick a reasonable starting point and expect to revisit
empirically once real labeled examples exist to sanity-check against.

TEMPLATE -- fill in the bodies below. A candidate needs to be checked
against a *moving* pedestrian, not just its start/end position -- walk the
candidate forward at an assumed speed to get its own (x, y, t) schedule
(see _path_time_schedule), then compare each point in that schedule against
the pedestrian's ground-truth position at that same t
(ActorTrajectory.pose_at(t) in trajectory_io.py), not at the decision
instant alone.
"""

import math
from typing import List

from training.common.dataset_schema import CandidatePath, DecisionPoint, OracleLabel
from training.common.trajectory_io import ActorTrajectory

CAR_ASSUMED_SPEED_MPS = 1.0  # TODO: tie to the DWA controller's actual planned speed once available
CAR_COLLISION_RADIUS_M = 0.25  # JetRacer footprint + margin
STATIC_OBSTACLE_INFLATION_M = 0.25  # matches OBSTACLE_FOOTPRINT_RADIUS_M's role in obstacle_randomizer.py

SCORE_WEIGHTS = {
    "collision_free": 0.5,
    "clearance": 0.2,
    "complexity": 0.1,
    "goal_progress": 0.2,
}


def _path_length(waypoints: List[tuple]) -> float:
    """Total length of the polyline through `waypoints`."""
    total = 0
    for i, waypoint in enumerate(waypoints):
        if i != 0:
            dist = math.sqrt((waypoints[i][0] - waypoints[i-1][0]) ** 2 + (waypoints[i][1] - waypoints[i-1][1]) ** 2)

            total += dist

    return total


def _path_time_schedule(waypoints: List[tuple], start_t: float, speed: float) -> List[tuple]:
    """(x, y, t) samples along the candidate, walked at constant `speed`
    starting from `start_t`."""
    schedule = [(waypoints[0][0], waypoints[0][1], start_t)]
    t = start_t
    for i in range(len(waypoints) - 1):
        x0, y0 = waypoints[i]
        x1, y1 = waypoints[i + 1]
        distance = math.hypot(x1 - x0, y1 - y0)
        t += distance / max(speed, 1e-3)
        schedule.append((x1, y1, t))
    return schedule


def _static_collision(waypoints: List[tuple], obstacle_footprints: List[dict]) -> bool:
    """True if any waypoint falls inside a known obstacle's footprint
    (inflated by STATIC_OBSTACLE_INFLATION_M)."""
    for x, y in waypoints:
        for footprint in obstacle_footprints:
            f_min, f_max = footprint.get("footprint_min"), footprint.get("footprint_max")
            if f_min is None or f_max is None:
                continue
            if (
                f_min[0] - STATIC_OBSTACLE_INFLATION_M <= x <= f_max[0] + STATIC_OBSTACLE_INFLATION_M
                and f_min[1] - STATIC_OBSTACLE_INFLATION_M <= y <= f_max[1] + STATIC_OBSTACLE_INFLATION_M
            ):
                return True
    return False


def _pedestrian_clearance(schedule: List[tuple], pedestrians: List[ActorTrajectory]) -> float:
    """Minimum distance, over the candidate's own time schedule, between the
    car and where ANY pedestrian's ground-truth trajectory says it actually
    is at each of those times -- min_pedestrian_clearance_m is a single
    worst-case number across every pedestrian in the scene, not just one of
    them (a real scene routinely has more than one, see
    scene_randomizer.pedestrian_randomizer.spawn_pedestrians, which drives
    every pedestrian in a take concurrently -- fixed 2026-09-23 to take a
    list here instead of one ActorTrajectory, matching that)."""
    valid_pedestrians = [p for p in pedestrians if p.samples]
    if not valid_pedestrians:
        return float("inf")
    return min(
        math.hypot(x - p.pose_at(t).x, y - p.pose_at(t).y)
        for x, y, t in schedule
        for p in valid_pedestrians
    )


def score_candidate(
    candidate: CandidatePath,
    decision: DecisionPoint,
    static_obstacle_footprints: List[dict],
    pedestrians: List[ActorTrajectory],
) -> OracleLabel:
    """Score one candidate: collision_free (static AND dynamic, dynamic
    meaning clear of EVERY pedestrian in `pedestrians`, not just one),
    min_pedestrian_clearance_m (the worst-case/minimum across all of them),
    path_complexity (length / straight-line distance), goal_progress
    (fraction of remaining straight-line distance to goal closed), combined
    into `score` via SCORE_WEIGHTS."""
    waypoints = candidate.waypoints
    if len(waypoints) < 2:
        return OracleLabel(
            candidate_id=candidate.candidate_id, collision_free=False,
            min_pedestrian_clearance_m=0.0, path_complexity=1.0, goal_progress=0.0, score=0.0,
        )

    schedule = _path_time_schedule(waypoints, decision.t, CAR_ASSUMED_SPEED_MPS)
    static_hit = _static_collision(waypoints, static_obstacle_footprints)
    clearance = _pedestrian_clearance(schedule, pedestrians)
    dynamic_hit = clearance < CAR_COLLISION_RADIUS_M
    collision_free = not (static_hit or dynamic_hit)

    straight_line = math.hypot(waypoints[-1][0] - waypoints[0][0], waypoints[-1][1] - waypoints[0][1])
    length = _path_length(waypoints)
    complexity = length / straight_line if straight_line > 1e-6 else 1.0

    start_to_goal = math.hypot(decision.goal_x - decision.car_x, decision.goal_y - decision.car_y)
    end_to_goal = math.hypot(decision.goal_x - waypoints[-1][0], decision.goal_y - waypoints[-1][1])
    goal_progress = max(0.0, (start_to_goal - end_to_goal) / start_to_goal) if start_to_goal > 1e-6 else 0.0

    clearance_score = 1.0 if clearance == float("inf") else min(clearance / (CAR_COLLISION_RADIUS_M * 4.0), 1.0)
    complexity_score = 1.0 / complexity  # 1.0 for a straight line, lower for winding paths

    score = (
        SCORE_WEIGHTS["collision_free"] * (1.0 if collision_free else 0.0)
        + SCORE_WEIGHTS["clearance"] * clearance_score
        + SCORE_WEIGHTS["complexity"] * complexity_score
        + SCORE_WEIGHTS["goal_progress"] * goal_progress
    )

    return OracleLabel(
        candidate_id=candidate.candidate_id,
        collision_free=collision_free,
        min_pedestrian_clearance_m=clearance,
        path_complexity=complexity,
        goal_progress=goal_progress,
        score=score,
    )


def score_all(
    candidates: List[CandidatePath],
    decision: DecisionPoint,
    static_obstacle_footprints: List[dict],
    pedestrians: List[ActorTrajectory],
) -> List[OracleLabel]:
    """score_candidate for every candidate, same order."""
    return [score_candidate(c, decision, static_obstacle_footprints, pedestrians) for c in candidates]
