"""Pedestrian path *planning*: heading/event sampling + A*-routed waypoints.

Pure geometry, no Isaac Sim dependency (like obstacle_randomizer.py's sampling
half) — easy to unit test, and importable from a plain Python process.

Routing reuses jetracer_navigation's grid A* (the same planner the robot
itself uses for global planning) instead of Isaac Sim's NavMesh, which has
proven unreliable in this environment (bakes hang regardless of volume size
or excluded geometry). Obstacle positions are already known at
scenario-generation time — we placed them ourselves — so a small occupancy
grid built directly from that layout needs no collision bake at all.
"""

import math
from typing import List, Tuple

import numpy as np

from jetracer_navigation.a_star import AStarImplementation
from jetracer_navigation.utils import pixel_to_world, smooth_path_generation, world_to_pixel

from scene_randomizer.obstacle_randomizer import OBSTACLE_FOOTPRINT_RADIUS_M
from scene_randomizer.scenario_io import PedestrianPathSpec, StaticObstacleSpec

HEADINGS = ("straight", "diagonal_left", "diagonal_right")
DIAGONAL_ANGLE_DEG = 30.0     # how far off the hallway axis "diagonal" means
MIN_SPEED_MPS = 0.6
MAX_SPEED_MPS = 1.4
STOP_DURATION_RANGE_S = (2.0, 5.0)
EVENT_FRACTION_RANGE = (0.3, 0.7)  # only turn/stop partway through, not at the very ends
PEDESTRIAN_WALL_CLEARANCE_M = 0.2
MIN_PEDESTRIAN_START_SPACING_M = 0.5  # keep pedestrian start points from spawning on top of each other
# Keep pedestrian start points this far in from the hallway's x_min/x_max --
# spawning exactly at the boundary can land in an architectural edge feature
# (a doorway, alcove, or end-cap room) rather than the open corridor floor.
# Same idea as obstacle_randomizer.OBSTACLE_START_CLEARANCE_M.
PEDESTRIAN_START_CLEARANCE_M = 1.0

GRID_RESOLUTION_M = 0.05       # occupancy grid cell size for A* routing
# Inflation radius for both the A* grid (plan_path) and the extra margin
# added on top of each obstacle's own real footprint (plan_pedestrian_commands'
# obstacle_clearance). Bumped from 0.25m 2026-09-23 -- a pedestrian was
# observed ending up overlapping an obstacle (a printer) at runtime, likely
# a behavior-agent local-avoidance wobble that the original margin didn't
# leave enough room to absorb. 0.35m is still a realistic human-body-plus-sway
# radius, not an unreasonably large planning margin.
PEDESTRIAN_BODY_RADIUS_M = 0.35
GOAL_THRESHOLD_M = 0.15

# Pedestrians route within a region inset this far from hallway_bounds on
# every side -- on top of PEDESTRIAN_WALL_CLEARANCE_M's own start-position
# inset, which only affects where a pedestrian is first placed, not the
# whole region an A*-planned path is free to pass through. Without this, a
# planned route can graze right up against the raw hallway_bounds edge,
# which is itself only a few cm of margin below the *measured* real wall
# (see scene_randomizer_node.py's hallway_bounds param) -- combined with a
# behavior agent's own runtime local-avoidance wobble, that was tight enough
# to leave a pedestrian stuck oscillating against a wall/obstacle. Added
# 2026-09-23. See [[jetracer-nav-pedestrian-bug]].
PEDESTRIAN_BOUNDS_INSET_M = 0.15


def _pedestrian_bounds(hallway_bounds: dict) -> dict:
    return {
        "x_min": hallway_bounds["x_min"] + PEDESTRIAN_BOUNDS_INSET_M,
        "x_max": hallway_bounds["x_max"] - PEDESTRIAN_BOUNDS_INSET_M,
        "y_min": hallway_bounds["y_min"] + PEDESTRIAN_BOUNDS_INSET_M,
        "y_max": hallway_bounds["y_max"] - PEDESTRIAN_BOUNDS_INSET_M,
    }


def sample_pedestrian_path(
    rng,
    hallway_bounds: dict,
    character_id: str,
) -> PedestrianPathSpec:
    """Sample one pedestrian's start point, heading, speed, and optional event."""
    hallway_bounds = _pedestrian_bounds(hallway_bounds)
    heading = rng.choice(HEADINGS)
    speed = rng.uniform(MIN_SPEED_MPS, MAX_SPEED_MPS)
    event = rng.choice(("turn", "stop"))
    event_time = rng.uniform(*EVENT_FRACTION_RANGE)
    turn = None
    stop = None
    if event == "turn":
        new_turn = rng.choice(HEADINGS)
        while new_turn == heading:
            new_turn = rng.choice(HEADINGS)
        turn = {"at_fraction": event_time, "new_heading": new_turn}
    else:
        stop_time = rng.uniform(*STOP_DURATION_RANGE_S)
        stop = {"at_fraction": event_time, "duration_s": stop_time}

    starting_y = rng.uniform(hallway_bounds["y_min"] + PEDESTRIAN_WALL_CLEARANCE_M, hallway_bounds["y_max"] - PEDESTRIAN_WALL_CLEARANCE_M)
    starting_x = rng.choice([
        hallway_bounds["x_min"] + PEDESTRIAN_START_CLEARANCE_M,
        hallway_bounds["x_max"] - PEDESTRIAN_START_CLEARANCE_M,
    ])
    return PedestrianPathSpec(
        asset_id=character_id,
        start_position=[starting_x, starting_y, 0.0],
        heading=heading,
        speed=speed,
        turn_event=turn,
        stop_event=stop,
        character_id=character_id,
    )


def sample_pedestrian_paths(
    num_pedestrians: int,
    character_pool: List[str],
    rng,
    hallway_bounds: dict,
) -> List[PedestrianPathSpec]:
    """Sample `num_pedestrians` paths, cycling/sampling from character_pool.

    Reject-and-resample against already-accepted pedestrians' start points,
    same idea as sample_obstacle_positions's spacing check.
    """
    peds = []
    while len(peds) < num_pedestrians:
        character_id = rng.choice(character_pool)
        candidate = sample_pedestrian_path(rng, hallway_bounds, character_id)

        too_close = False
        for other in peds:
            dx = candidate.start_position[0] - other.start_position[0]
            dy = candidate.start_position[1] - other.start_position[1]
            if math.sqrt(dx ** 2 + dy ** 2) < MIN_PEDESTRIAN_START_SPACING_M:
                too_close = True
                break

        if not too_close:
            peds.append(candidate)

    return peds


def _heading_to_vector(heading: str, direction_sign: float = 1.0) -> Tuple[float, float]:
    """Convert a heading name to a unit (dx, dy) along/across the hallway axis.

    Computed as if walking from x_min toward x_max, then scaled by
    `direction_sign` (+1.0 or -1.0) — negating both components together is
    what keeps "diagonal_left"/"diagonal_right" meaning left/right relative
    to which way the pedestrian is actually facing.
    """
    if heading == "straight":
        dx, dy = 1.0, 0.0
    elif heading == "diagonal_left":
        dx, dy = math.cos(math.radians(DIAGONAL_ANGLE_DEG)), math.sin(math.radians(DIAGONAL_ANGLE_DEG))
    else:
        dx, dy = math.cos(math.radians(-DIAGONAL_ANGLE_DEG)), math.sin(math.radians(-DIAGONAL_ANGLE_DEG))

    return (dx * direction_sign, dy * direction_sign)


class _Position:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class _Origin:
    def __init__(self, x: float, y: float):
        self.position = _Position(x, y)


class _MapInfo:
    """Duck-typed stand-in for nav_msgs/OccupancyGrid's `.info` field, so
    jetracer_navigation.utils' world_to_pixel/pixel_to_world work unmodified
    outside of a ROS map topic."""

    def __init__(self, origin_x: float, origin_y: float, resolution: float):
        self.origin = _Origin(origin_x, origin_y)
        self.resolution = resolution


def build_occupancy_grid(
    hallway_bounds: dict,
    obstacles: List[StaticObstacleSpec],
    resolution: float = GRID_RESOLUTION_M,
):
    """Rasterize the known obstacles into a binary grid (1=blocked).

    Origin is pinned to (x_min, y_min) so grid coordinates are always >= 0,
    matching what world_to_pixel/pixel_to_world expect. Walls are NOT
    pre-stamped as a blocked band here: the grid's own array edges already
    hard-constrain any plan to [x_min, x_max] x [y_min, y_max] via
    is_in_bounds, and "stay clear of the wall" is handled by the callers that
    pick start/goal points (sample_pedestrian_path's PEDESTRIAN_WALL_CLEARANCE_M
    inset, and plan_pedestrian_commands' goal-clearance margin below). Stamping
    a second, separately-inflatable wall band here previously caused planning
    to fail outright: obstacle inflation (see plan_path) would eat into that
    band from both sides, and a start/goal point sampled near the boundary of
    the two combined margins could get stranded as an isolated free pixel
    surrounded entirely by inflated wall cells, with A* correctly reporting no
    path.
    """
    x_min, x_max = hallway_bounds["x_min"], hallway_bounds["x_max"]
    y_min, y_max = hallway_bounds["y_min"], hallway_bounds["y_max"]

    width = int(math.ceil((x_max - x_min) / resolution)) + 1
    height = int(math.ceil((y_max - y_min) / resolution)) + 1
    grid = np.zeros((height, width), dtype=np.uint8)
    map_info = _MapInfo(x_min, y_min, resolution)

    for obstacle in obstacles:
        fmin, fmax = obstacle.footprint_min, obstacle.footprint_max
        if fmin is not None and fmax is not None:
            # Stamp the real measured rectangle directly -- matches the
            # pattern obstacle_randomizer.build_ground_truth_occupancy_grid
            # already uses. A circular approximation (this code's original
            # approach, and briefly a half-diagonal-radius circle on
            # 2026-09-22) over-inflates elongated obstacles hugely on their
            # short axis -- a 1.17x0.52m desk's half-diagonal is 0.64m, but
            # its real short-axis half-width is only 0.26m, so a circular
            # stamp claimed 2.5x more clearance than the real object needs on
            # that side. That over-inflation was severe enough to make most
            # pedestrian routes in a 2.32m-wide hallway fail to plan at all.
            # See [[jetracer-nav-pedestrian-bug]].
            px_lo = int(math.floor((fmin[0] - x_min) / resolution))
            px_hi = int(math.ceil((fmax[0] - x_min) / resolution))
            py_lo = int(math.floor((fmin[1] - y_min) / resolution))
            py_hi = int(math.ceil((fmax[1] - y_min) / resolution))
            px_lo, px_hi = max(0, px_lo), min(width, px_hi + 1)
            py_lo, py_hi = max(0, py_lo), min(height, py_hi + 1)
            grid[py_lo:py_hi, px_lo:px_hi] = 1
        else:
            # No real footprint known (e.g. a StaticObstacleSpec built from a
            # live-stage position read with no bbox measured -- see
            # test_pedestrian_spawn.py's read_live_obstacles) -- fall back to
            # a uniform circular approximation.
            center = world_to_pixel(obstacle.position[0], obstacle.position[1], map_info)
            radius_px = int(round(OBSTACLE_FOOTPRINT_RADIUS_M / resolution))
            _stamp_circle(grid, center.x, center.y, radius_px)

    return grid, map_info


def _stamp_circle(grid: np.ndarray, cx: int, cy: int, radius_px: int) -> None:
    height, width = grid.shape
    x_lo, x_hi = max(0, cx - radius_px), min(width, cx + radius_px + 1)
    y_lo, y_hi = max(0, cy - radius_px), min(height, cy + radius_px + 1)
    for yy in range(y_lo, y_hi):
        for xx in range(x_lo, x_hi):
            if (xx - cx) ** 2 + (yy - cy) ** 2 <= radius_px ** 2:
                grid[yy, xx] = 1


def _project_to_bounds(
    start_xy: Tuple[float, float],
    direction_xy: Tuple[float, float],
    hallway_bounds: dict,
    clearance: float,
) -> Tuple[float, float]:
    """Follow `direction_xy` from `start_xy` until hitting the
    (clearance-inset) hallway rectangle boundary; return that point.

    Replaces the old fixed-overshoot-distance approach (which relied on
    downstream clamping) with an actual in-bounds target A* can plan to.
    """
    sx, sy = start_xy
    dx, dy = direction_xy
    x_min = hallway_bounds["x_min"] + clearance
    x_max = hallway_bounds["x_max"] - clearance
    y_min = hallway_bounds["y_min"] + clearance
    y_max = hallway_bounds["y_max"] - clearance

    candidates = []
    if dx > 0:
        candidates.append((x_max - sx) / dx)
    elif dx < 0:
        candidates.append((x_min - sx) / dx)
    if dy > 0:
        candidates.append((y_max - sy) / dy)
    elif dy < 0:
        candidates.append((y_min - sy) / dy)

    t = min(t for t in candidates if t > 0)
    return (sx + dx * t, sy + dy * t)


def _project_partial(
    start_xy: Tuple[float, float],
    direction_xy: Tuple[float, float],
    hallway_bounds: dict,
    clearance: float,
    fraction: float,
) -> Tuple[float, float]:
    """Point `fraction` of the way from start_xy to where `direction_xy` would
    hit the hallway boundary — used to place a turn/stop partway through a leg."""
    full_x, full_y = _project_to_bounds(start_xy, direction_xy, hallway_bounds, clearance)
    return (
        start_xy[0] + (full_x - start_xy[0]) * fraction,
        start_xy[1] + (full_y - start_xy[1]) * fraction,
    )


def plan_path(
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    grid: np.ndarray,
    map_info,
    resolution: float = GRID_RESOLUTION_M,
) -> List[Tuple[float, float]]:
    """Run A* from start_xy to goal_xy on `grid`, return smoothed world-space waypoints."""
    start_px = world_to_pixel(start_xy[0], start_xy[1], map_info)
    goal_px = world_to_pixel(goal_xy[0], goal_xy[1], map_info)
    inflation_px = max(1, int(round(PEDESTRIAN_BODY_RADIUS_M / resolution)))
    goal_threshold_px = max(1, int(round(GOAL_THRESHOLD_M / resolution)))

    planner = AStarImplementation(
        grid, start_px, goal_px,
        goal_threshold=goal_threshold_px,
        inflation_radius=inflation_px,
    )
    path, _visited = planner.plan()
    if not path:
        raise RuntimeError(f"No path found from {start_xy} to {goal_xy} around the current obstacle layout")

    smoothed = smooth_path_generation(path, occupancy_map=planner.inflated_obstacle_map)
    path = smoothed or path

    return [pixel_to_world(node.coords.x, node.coords.y, map_info) for node in path]


def _goto_commands(waypoints: List[Tuple[float, float]], speed: float) -> List[dict]:
    return [{"type": "goto", "x": x, "y": y, "speed": speed} for x, y in waypoints]


def _index_at_fraction(waypoints: List[Tuple[float, float]], fraction: float) -> int:
    """Index into `waypoints` whose cumulative arc-length is closest to
    `fraction` of the total path length."""
    if len(waypoints) < 2:
        return 0
    seg_lengths = [math.dist(waypoints[i], waypoints[i + 1]) for i in range(len(waypoints) - 1)]
    target = sum(seg_lengths) * fraction
    covered = 0.0
    for i, seg_len in enumerate(seg_lengths):
        covered += seg_len
        if covered >= target:
            return i + 1
    return len(waypoints) - 1


def _push_clear_of_one_obstacle(
    x: float, y: float, obstacle: StaticObstacleSpec, margin: float,
) -> Tuple[float, float, bool]:
    """If (x, y) is within `margin` of `obstacle`'s real footprint rectangle
    (or, lacking real footprint data, within margin + OBSTACLE_FOOTPRINT_RADIUS_M
    of its center), push it to just outside that margin and return
    (new_x, new_y, True); otherwise return the point unchanged with False.

    Pushes to the real rectangle's nearest edge, not radially from the
    obstacle's center circle -- a circular approximation (even one sized
    from the real footprint's half-diagonal, tried briefly on 2026-09-22)
    over-inflates elongated obstacles on their short axis badly enough to
    make most routes in a 2.32m-wide hallway fail to plan. See
    [[jetracer-nav-pedestrian-bug]].
    """
    fmin, fmax = obstacle.footprint_min, obstacle.footprint_max
    if fmin is None or fmax is None:
        ox, oy = obstacle.position[0], obstacle.position[1]
        dx, dy = x - ox, y - oy
        dist = math.hypot(dx, dy)
        min_clearance = OBSTACLE_FOOTPRINT_RADIUS_M + margin
        if dist >= min_clearance:
            return x, y, False
        if dist < 1e-9:
            dx, dy, dist = 1.0, 0.0, 1.0
        scale = min_clearance / dist
        return ox + dx * scale, oy + dy * scale, True

    lo_x, lo_y = fmin[0] - margin, fmin[1] - margin
    hi_x, hi_y = fmax[0] + margin, fmax[1] + margin
    if not (lo_x <= x <= hi_x and lo_y <= y <= hi_y):
        return x, y, False
    # Inside the margin-expanded rectangle -- push out through the nearest edge.
    d_left, d_right = x - lo_x, hi_x - x
    d_bottom, d_top = y - lo_y, hi_y - y
    closest = min(d_left, d_right, d_bottom, d_top)
    if closest == d_left:
        return lo_x, y, True
    if closest == d_right:
        return hi_x, y, True
    if closest == d_bottom:
        return x, lo_y, True
    return x, hi_y, True


def _clear_of_obstacles(
    point: Tuple[float, float],
    obstacles: List[StaticObstacleSpec],
    pedestrian_margin: float,
    passes: int = 3,
) -> Tuple[float, float]:
    """Push `point` clear of every obstacle it's within `pedestrian_margin`
    of, using each obstacle's real measured footprint rectangle (or a
    fallback circle -- see _push_clear_of_one_obstacle).

    Needed because start/turn/goal points are picked purely from hallway
    geometry (walk straight to the wall), with no awareness of obstacles —
    plan_path's own inflation forcibly frees the exact start/goal pixel
    regardless of surroundings (see build_occupancy_grid's docstring), so a
    point picked inside an obstacle's inflated footprint becomes an isolated
    free pixel with no reachable neighbors instead of a normal collision
    rejection. Repeated a few passes since pushing clear of one obstacle can
    move a point into another's margin.
    """
    x, y = point
    for _ in range(passes):
        moved = False
        for obstacle in obstacles:
            x, y, did_move = _push_clear_of_one_obstacle(x, y, obstacle, pedestrian_margin)
            moved = moved or did_move
        if not moved:
            break
    return (x, y)


def plan_pedestrian_commands(
    spec: PedestrianPathSpec,
    obstacles: List[StaticObstacleSpec],
    hallway_bounds: dict,
    resolution: float = GRID_RESOLUTION_M,
) -> List[dict]:
    """A*-routed replacement for a straight-line command sequence.

    Keeps pedestrian_randomizer's original "at most one turn or stop event"
    design, but each leg is planned around the known obstacle layout instead
    of assuming a clear straight line — so a pedestrian whose designed
    trajectory would clip through a chair or desk detours around it instead.
    """
    hallway_bounds = _pedestrian_bounds(hallway_bounds)
    grid, map_info = build_occupancy_grid(hallway_bounds, obstacles, resolution)
    # Per-obstacle real footprint radius is now added inside _clear_of_obstacles
    # itself (see _obstacle_effective_radius) -- this is just the pedestrian's
    # own body margin on top of that.
    obstacle_clearance = PEDESTRIAN_BODY_RADIUS_M + resolution

    sx, sy, _ = spec.start_position
    start_xy = _clear_of_obstacles((sx, sy), obstacles, obstacle_clearance)
    # sx starts inset from whichever edge it's nearest to (see
    # PEDESTRIAN_START_CLEARANCE_M), not exactly on hallway_bounds' x_min/x_max
    # anymore, so pick direction by nearest edge rather than exact equality.
    dist_to_min = abs(sx - hallway_bounds["x_min"])
    dist_to_max = abs(sx - hallway_bounds["x_max"])
    direction_sign = 1.0 if dist_to_min <= dist_to_max else -1.0
    clearance = PEDESTRIAN_WALL_CLEARANCE_M

    dx, dy = _heading_to_vector(spec.heading, direction_sign)
    commands: List[dict] = []

    if spec.turn_event is not None:
        turn_xy = _project_partial(start_xy, (dx, dy), hallway_bounds, clearance, spec.turn_event["at_fraction"])
        turn_xy = _clear_of_obstacles(turn_xy, obstacles, obstacle_clearance)
        commands += _goto_commands(plan_path(start_xy, turn_xy, grid, map_info, resolution), spec.speed)

        new_dx, new_dy = _heading_to_vector(spec.turn_event["new_heading"], direction_sign)
        final_xy = _project_to_bounds(turn_xy, (new_dx, new_dy), hallway_bounds, clearance)
        final_xy = _clear_of_obstacles(final_xy, obstacles, obstacle_clearance)
        commands += _goto_commands(plan_path(turn_xy, final_xy, grid, map_info, resolution), spec.speed)

    elif spec.stop_event is not None:
        final_xy = _project_to_bounds(start_xy, (dx, dy), hallway_bounds, clearance)
        final_xy = _clear_of_obstacles(final_xy, obstacles, obstacle_clearance)
        waypoints = plan_path(start_xy, final_xy, grid, map_info, resolution)
        stop_index = _index_at_fraction(waypoints, spec.stop_event["at_fraction"])

        commands += _goto_commands(waypoints[:stop_index + 1], spec.speed)
        commands.append({"type": "idle", "duration_s": spec.stop_event["duration_s"]})
        commands += _goto_commands(waypoints[stop_index:], spec.speed)

    else:
        final_xy = _project_to_bounds(start_xy, (dx, dy), hallway_bounds, clearance)
        final_xy = _clear_of_obstacles(final_xy, obstacles, obstacle_clearance)
        commands += _goto_commands(plan_path(start_xy, final_xy, grid, map_info, resolution), spec.speed)

    return commands
