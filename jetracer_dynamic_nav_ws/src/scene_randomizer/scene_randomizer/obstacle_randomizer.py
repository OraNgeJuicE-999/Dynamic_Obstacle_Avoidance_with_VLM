"""Static obstacle scattering.

IMPORTANT: this module imports Isaac Sim's `omni.*` APIs, which only resolve
inside Isaac Sim's own Python process (its Kit application, or a standalone
script/extension run through Isaac Sim's python.sh) — they are not installable
via pip into a normal ROS2 environment. See scene_randomizer_node.py for how
this gets wired up.

Two responsibilities live here, kept separate from the ROS glue:
  1. sampling *where* obstacles go (pure geometry, no Isaac Sim dependency —
     easy to unit test on its own)
  2. actually spawning/clearing them on the USD stage (Isaac Sim dependency)
"""

import random
from typing import List
from pathlib import Path
import math
import numpy as np

from tf_transformations import quaternion_from_euler
from scene_randomizer.scenario_io import StaticObstacleSpec

WALL_CLEARANCE_M = 0.15       # keep obstacles this far from the walls
MIN_OBSTACLE_SPACING_M = 0.5  # keep obstacles this far from each other
MIN_PATH_WIDTH_M = 0.6        # never let obstacles close off a corridor narrower than this
# Keep obstacles this far from both ends of the hallway (along its long/x axis)
# so the car/pedestrian start poses -- which can be at either end, see
# car_pose.py -- always have clear room around them instead of an obstacle
# potentially spawning right at the entrance.
OBSTACLE_START_CLEARANCE_M = 2.0

# StaticObstacleSpec has no real per-asset size yet, so treat every obstacle as
# a circle of this radius for corridor-width purposes. Revisit once assets
# carry real dimensions.
OBSTACLE_FOOTPRINT_RADIUS_M = 0.25



def load_obstacle_prefab_library(assets_dir: str) -> List[str]:
    """Return the list of obstacle asset paths (.glb/.usd) to sample from.

    Scans `assets_dir` (this repo's `hallway_assets/static_obstacles/`) for
    usable prop assets. Does not filter by footprint size — see that folder's
    README for the "only drop obstacle-scale props here" convention instead.
    """
    glb_list = [str(p) for p in Path(assets_dir).glob('*.glb')]
    usd_list = [str(p) for p in Path(assets_dir).glob('*.usd')]
    return glb_list + usd_list


def sample_obstacle_positions(
    num_obstacles: int,
    prefab_library: List[str],
    rng: random.Random,
    hallway_bounds: dict,
) -> List[StaticObstacleSpec]:
    """Sample non-overlapping obstacle placements inside hallway_bounds.

    TODO: reject-and-resample loop —
      - pick a random (x, y) + yaw within HALLWAY_BOUNDS
      - reject if within WALL_CLEARANCE_M of a wall
      - reject if within MIN_OBSTACLE_SPACING_M of an already-placed obstacle
      - reject if placing it would leave less than MIN_PATH_WIDTH_M of clear
        corridor at that x (so the hallway never becomes fully blocked)
      - pick asset_path via rng.choice(prefab_library)
    Return one StaticObstacleSpec per accepted obstacle (prim_path left None
    until spawn_static_obstacles fills it in).
    """
    obj_list = []
    num = 0
    
    while num < num_obstacles:
      xx = rng.uniform(
          hallway_bounds["x_min"] + OBSTACLE_START_CLEARANCE_M,
          hallway_bounds["x_max"] - OBSTACLE_START_CLEARANCE_M,
      )
      yy = rng.uniform(hallway_bounds["y_min"] + WALL_CLEARANCE_M, hallway_bounds["y_max"] - WALL_CLEARANCE_M)
      yaw = rng.uniform(-180, 180)
      check = True

      # within another obj
      if obj_list:
        for node in obj_list:
          x = node.position[0]
          y = node.position[1]
          dist = math.sqrt((xx - x) ** 2 + (yy - y) ** 2)
          if dist < MIN_OBSTACLE_SPACING_M:
            check = False

        # open path with a certain width
        if _min_corridor_width_at(xx, obj_list, hallway_bounds) < MIN_PATH_WIDTH_M:
          check = False

      if check:
        num += 1
        obstacle = StaticObstacleSpec(
           asset_path=rng.choice(prefab_library),
           position=[xx, yy, 0.0],
           yaw_deg=yaw,
        )
        obj_list.append(obstacle)
           
    return obj_list


def spawn_static_obstacles(specs: List[StaticObstacleSpec]) -> List[StaticObstacleSpec]:
    """Reference each obstacle asset onto the USD stage and place it.

    Isaac Sim references .glb files directly (confirmed empirically — no
    separate .glb->.usd conversion step needed), so .usd/.glb asset_paths are
    both just passed straight through to add_reference_to_stage.

    Mutates each spec's `prim_path` in place and returns the same list, so
    clear_static_obstacles can later remove exactly these prims. Also mutates
    `footprint_min`/`footprint_max` with the prim's real measured world-space
    bounding box — this can only happen after the prim is actually on the
    stage and posed, which is why it's not part of sample_obstacle_positions.
    """
    from isaacsim.core.prims import XFormPrim
    from isaacsim.core.utils.stage import add_reference_to_stage
    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=True)

    for i, spec in enumerate(specs):
        prim_path = f"/World/hallway_obstacles/obstacle_{i}"

        add_reference_to_stage(usd_path=spec.asset_path, prim_path=prim_path)

        # tf_transformations returns (x, y, z, w); Isaac Sim's XFormPrim wants
        # (w, x, y, z) — reorder, don't pass it straight through.
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, math.radians(spec.yaw_deg))
        orientation = np.array([qw, qx, qy, qz])

        XFormPrim(prim_path).set_world_poses(
            positions=np.array([spec.position]),
            orientations=np.array([orientation]),
        )

        spec.prim_path = prim_path

        # Measure the real footprint now that the prim is placed and rotated —
        # the world-space bbox already accounts for yaw_deg, no separate
        # rotation math needed.
        bbox_cache.Clear()
        prim_range = bbox_cache.ComputeWorldBound(stage.GetPrimAtPath(prim_path)).ComputeAlignedRange()
        min_pt, max_pt = prim_range.GetMin(), prim_range.GetMax()
        spec.footprint_min = [float(min_pt[0]), float(min_pt[1])]
        spec.footprint_max = [float(max_pt[0]), float(max_pt[1])]

    return specs


def build_ground_truth_occupancy_grid(
    hallway_bounds: dict,
    obstacles: List[StaticObstacleSpec],
    resolution: float = 0.05,
):
    """Rasterize obstacles' real measured footprints into a binary grid (1=blocked).

    Unlike pedestrian_planner.build_occupancy_grid (which approximates every
    obstacle as a uniform-radius circle for cheap pre-spawn A* planning), this
    stamps each obstacle's actual footprint_min/footprint_max — so it can only
    be built *after* spawn_static_obstacles has measured them. Same grid
    convention as pedestrian_planner's grid (origin pinned to (x_min, y_min),
    row-major [y, x]) so callers can use it the same way.

    Returns (grid, map_info) where map_info exposes .origin.position.x/y and
    .resolution, matching pedestrian_planner's duck-typed OccupancyGrid.info.
    """
    x_min, x_max = hallway_bounds["x_min"], hallway_bounds["x_max"]
    y_min, y_max = hallway_bounds["y_min"], hallway_bounds["y_max"]

    width = int(math.ceil((x_max - x_min) / resolution)) + 1
    height = int(math.ceil((y_max - y_min) / resolution)) + 1
    grid = np.zeros((height, width), dtype=np.uint8)

    for obstacle in obstacles:
        if obstacle.footprint_min is None or obstacle.footprint_max is None:
            continue

        px_lo = int(math.floor((obstacle.footprint_min[0] - x_min) / resolution))
        px_hi = int(math.ceil((obstacle.footprint_max[0] - x_min) / resolution))
        py_lo = int(math.floor((obstacle.footprint_min[1] - y_min) / resolution))
        py_hi = int(math.ceil((obstacle.footprint_max[1] - y_min) / resolution))

        px_lo, px_hi = max(0, px_lo), min(width, px_hi + 1)
        py_lo, py_hi = max(0, py_lo), min(height, py_hi + 1)
        grid[py_lo:py_hi, px_lo:px_hi] = 1

    class _Position:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    class _Origin:
        def __init__(self, x, y):
            self.position = _Position(x, y)

    class _MapInfo:
        def __init__(self, origin_x, origin_y, resolution):
            self.origin = _Origin(origin_x, origin_y)
            self.resolution = resolution

    return grid, _MapInfo(x_min, y_min, resolution)


def clear_static_obstacles(specs: List[StaticObstacleSpec]) -> None:
    """Remove previously spawned obstacle prims from the stage.

    Call this before spawning a new layout so obstacles don't accumulate
    across regenerations.
    """
    from isaacsim.core.utils.prims import delete_prim, is_prim_path_valid

    for spec in specs:
        if spec.prim_path and is_prim_path_valid(spec.prim_path):
            delete_prim(spec.prim_path)
        spec.prim_path = None


def _min_corridor_width_at(x: float, placed: List[StaticObstacleSpec], hallway_bounds: dict) -> float:
    """Widest clear gap across the hallway's y-extent at hallway position `x`.

    Obstacles are approximated as circles of radius OBSTACLE_FOOTPRINT_RADIUS_M
    (see that constant's comment for why). Only obstacles whose x-position is
    within one footprint radius of `x` can block the corridor at this x-slice —
    everything else is too far away along the hallway to matter here.
    """
    y_min = hallway_bounds["y_min"]
    y_max = hallway_bounds["y_max"]

    blocked_intervals = []
    for obstacle in placed:
        ox, oy = obstacle.position[0], obstacle.position[1]
        if abs(ox - x) > OBSTACLE_FOOTPRINT_RADIUS_M:
            continue
        lo = max(y_min, oy - OBSTACLE_FOOTPRINT_RADIUS_M)
        hi = min(y_max, oy + OBSTACLE_FOOTPRINT_RADIUS_M)
        blocked_intervals.append((lo, hi))

    if not blocked_intervals:
        return y_max - y_min

    blocked_intervals.sort()
    merged = [blocked_intervals[0]]
    for lo, hi in blocked_intervals[1:]:
        last_lo, last_hi = merged[-1]
        if lo <= last_hi:
            merged[-1] = (last_lo, max(last_hi, hi))
        else:
            merged.append((lo, hi))

    max_gap = merged[0][0] - y_min
    for i in range(1, len(merged)):
        gap = merged[i][0] - merged[i - 1][1]
        if gap > max_gap:
            max_gap = gap
    max_gap = max(max_gap, y_max - merged[-1][1])

    return max_gap
