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

import json
import random
from typing import List
from pathlib import Path
import math
import numpy as np

from tf_transformations import quaternion_from_euler
from scene_randomizer.scenario_io import StaticObstacleSpec

WALL_CLEARANCE_M = 0.15       # keep obstacles this far from the walls
MIN_OBSTACLE_SPACING_M = 0.5  # keep obstacles this far from each other, ON TOP OF their own footprints
MIN_PATH_WIDTH_M = 0.6        # never let obstacles close off a corridor narrower than this
# Keep obstacles this far from both ends of the hallway (along its long/x axis)
# so the car/pedestrian start poses -- which can be at either end, see
# car_pose.py -- always have clear room around them instead of an obstacle
# potentially spawning right at the entrance.
OBSTACLE_START_CLEARANCE_M = 2.0

# Fallback used only when a prefab has no <name>.json sidecar (see
# _asset_footprint_radius) -- real per-asset sizes are used whenever
# available. This existed as the ONLY radius everywhere until 2026-09-22,
# when scenes_tester's visualize_scene.py caught pedestrian routes clipping
# through real obstacle geometry -- real prefabs measured up to ~3x larger
# than this (e.g. wooden_table's real half-diagonal is ~0.68m). See
# [[jetracer-nav-pedestrian-bug]].
OBSTACLE_FOOTPRINT_RADIUS_M = 0.25


def _asset_footprint_radius(asset_path: str) -> float:
    """Real half-diagonal footprint radius for `asset_path`, from its sibling
    <name>.json sidecar's width_m/depth_m (see hallway_assets/
    static_obstacles/*.json) -- falls back to OBSTACLE_FOOTPRINT_RADIUS_M if
    no sidecar exists or it can't be read. Used at PLACEMENT time, before any
    prim is spawned/measurable. Once an obstacle is actually spawned and its
    real footprint_min/footprint_max is measured (spawn_static_obstacles),
    pedestrian_planner._obstacle_effective_radius uses that live measurement
    instead -- this is the a-priori equivalent for before that exists.
    """
    sidecar_path = str(Path(asset_path).with_suffix(".json"))
    try:
        with open(sidecar_path) as f:
            meta = json.load(f)
        return math.hypot(meta["width_m"] / 2.0, meta["depth_m"] / 2.0)
    except Exception:
        return OBSTACLE_FOOTPRINT_RADIUS_M



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
      # Picked before the checks (not after, like this used to) so the
      # candidate's own real size can factor into the spacing check below.
      asset_path = rng.choice(prefab_library)
      candidate_radius = _asset_footprint_radius(asset_path)
      check = True

      # within another obj
      if obj_list:
        for node in obj_list:
          x = node.position[0]
          y = node.position[1]
          dist = math.sqrt((xx - x) ** 2 + (yy - y) ** 2)
          min_spacing = candidate_radius + _asset_footprint_radius(node.asset_path) + MIN_OBSTACLE_SPACING_M
          if dist < min_spacing:
            check = False

        # open path with a certain width
        if _min_corridor_width_at(xx, obj_list, hallway_bounds) < MIN_PATH_WIDTH_M:
          check = False

      if check:
        num += 1
        obstacle = StaticObstacleSpec(
           asset_path=asset_path,
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

    Also applies real PhysX collision to each spawned mesh -- confirmed
    2026-09-23 that `add_reference_to_stage` on its own gives a purely visual
    prim with NO collision at all (verified directly: a freshly spawned
    obstacle let overlap queries pass straight through it everywhere, at
    every height, until this was added). Without this, the actual
    car/pedestrian physics bodies could drive/walk straight through a
    "spawned" obstacle in the live simulation even though the ground-truth
    occupancy grid correctly marks it as blocked for planning purposes --
    the grid and the real physics scene would have silently disagreed.
    Uses convexDecomposition (not convexHull) specifically so real gaps in
    an asset's shape (e.g. a table's open space between its four legs)
    survive as real gaps in the physics collision too, not get filled in as
    one solid convex blob -- see build_lidar_height_occupancy_grid, which
    depends on this being real per-shape collision, not an approximation
    that erases the exact detail it's trying to capture.
    """
    from isaacsim.core.prims import XFormPrim
    from isaacsim.core.utils.stage import add_reference_to_stage
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    # useExtentsHint=False, NOT True: an authored extentsHint on an asset can
    # be stale/wrong and get read instead of the live posed geometry -- this
    # exact failure mode was confirmed 2026-09-22 for the "cabnit" prefab,
    # whose useExtentsHint=True-measured Y-footprint (1.41m) turned out to
    # exactly match its HEIGHT (1.41m per its sidecar .json), not its real
    # ~0.64m depth. Same root cause as the hallway geometry's own bbox issues
    # documented in [[jetracer-nav-isaac-gotchas]] -- always compute real
    # geometry, don't trust an asset's extentsHint.
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False)

    for i, spec in enumerate(specs):
        prim_path = f"/World/hallway_obstacles/obstacle_{i}"

        add_reference_to_stage(usd_path=spec.asset_path, prim_path=prim_path)

        for mesh_prim in Usd.PrimRange(stage.GetPrimAtPath(prim_path)):
            if not mesh_prim.IsA(UsdGeom.Mesh):
                continue
            UsdPhysics.CollisionAPI.Apply(mesh_prim)
            mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
            mesh_collision.CreateApproximationAttr("convexDecomposition")

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


# The real, live-measured height of the JetRacer's lidar sensor -- read from
# /leatherback_ROS/Rigid_Bodies/Chassis/Lidar_01's world position on the
# actual project stage, 2026-09-23. This is the real robot in use; an
# earlier estimate (~0.1225m) came from jetracer.urdf/jetracer.xacro at the
# repo root, which describe a DIFFERENT, unused robot model -- don't revive
# that number.
LIDAR_HEIGHT_M = 0.2850431556549647


def build_lidar_height_occupancy_grid(
    hallway_bounds: dict,
    obstacles: List[StaticObstacleSpec],
    lidar_height: float = LIDAR_HEIGHT_M,
    resolution: float = 0.05,
    scan_margin_m: float = 0.1,
):
    """Binary occupancy grid (same format/convention as
    build_ground_truth_occupancy_grid) representing what's ACTUALLY solid at
    `lidar_height`, not each obstacle's full top-to-bottom bounding box
    collapsed straight down. A table's open space between its four legs
    reads as free here, even though build_ground_truth_occupancy_grid marks
    its whole footprint as blocked -- the two grids answer different
    questions ("could anything about this obstacle possibly matter" vs.
    "does the lidar's own view actually see something blocking here").

    Requires each obstacle to already have REAL PhysX collision applied
    (spawn_static_obstacles does this) -- confirmed 2026-09-23 that a purely
    visual mesh with no collision reads as free everywhere, at every height,
    which would silently produce an all-clear grid instead of an error.

    Only scans within each obstacle's own measured footprint (padded by
    `scan_margin_m`), not the whole hallway -- open floor away from any
    obstacle is already known free without needing a physics query there,
    and scanning only ~14,900 cells' worth of open hallway floor for no
    reason would be needlessly slow. Uses
    omni.physx.get_physx_scene_query_interface().overlap_box_any -- Isaac
    Sim only, requires a live stage with Play active.
    """
    import carb
    import omni.physx

    x_min, x_max = hallway_bounds["x_min"], hallway_bounds["x_max"]
    y_min, y_max = hallway_bounds["y_min"], hallway_bounds["y_max"]
    width = int(math.ceil((x_max - x_min) / resolution)) + 1
    height = int(math.ceil((y_max - y_min) / resolution)) + 1
    grid = np.zeros((height, width), dtype=np.uint8)

    iface = omni.physx.get_physx_scene_query_interface()
    half_extent = carb.Float3(resolution / 2, resolution / 2, 0.02)
    rot = carb.Float4(0, 0, 0, 1)

    for obstacle in obstacles:
        if obstacle.footprint_min is None or obstacle.footprint_max is None:
            continue
        ox_lo = obstacle.footprint_min[0] - scan_margin_m
        oy_lo = obstacle.footprint_min[1] - scan_margin_m
        ox_hi = obstacle.footprint_max[0] + scan_margin_m
        oy_hi = obstacle.footprint_max[1] + scan_margin_m

        px_lo = max(0, int(math.floor((ox_lo - x_min) / resolution)))
        px_hi = min(width, int(math.ceil((ox_hi - x_min) / resolution)) + 1)
        py_lo = max(0, int(math.floor((oy_lo - y_min) / resolution)))
        py_hi = min(height, int(math.ceil((oy_hi - y_min) / resolution)) + 1)

        for py in range(py_lo, py_hi):
            wy = y_min + py * resolution
            for px in range(px_lo, px_hi):
                wx = x_min + px * resolution
                if iface.overlap_box_any(half_extent, carb.Float3(wx, wy, lidar_height), rot):
                    grid[py, px] = 1

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

    Each obstacle is approximated as a circle of its own real footprint
    radius (see _asset_footprint_radius) -- was a single fixed
    OBSTACLE_FOOTPRINT_RADIUS_M for every obstacle regardless of real size
    until 2026-09-22. Only obstacles whose x-position is within one footprint
    radius of `x` can block the corridor at this x-slice — everything else is
    too far away along the hallway to matter here.
    """
    y_min = hallway_bounds["y_min"]
    y_max = hallway_bounds["y_max"]

    blocked_intervals = []
    for obstacle in placed:
        ox, oy = obstacle.position[0], obstacle.position[1]
        radius = _asset_footprint_radius(obstacle.asset_path)
        if abs(ox - x) > radius:
            continue
        lo = max(y_min, oy - radius)
        hi = min(y_max, oy + radius)
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
