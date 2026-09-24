"""Dynamic (pedestrian) occupancy maps, built from recorded trajectory.json
data -- the moving-obstacle counterpart to obstacle_randomizer's static
ground_truth grid. Pure geometry, no Isaac Sim/ROS2 dependency (same split as
pedestrian_planner.py) -- this reads already-saved trajectory data, it
doesn't need a live stage.

Unlike the static ground truth (one fixed grid), a query here asks "where
will the pedestrian(s) be over the next N timesteps from time t" -- not just
their position at t alone. This is deliberate: scoring a candidate path
against only the pedestrian's *current* position ignores where they're
headed, which is exactly the "read a person's motion before it shows up as
a measurable velocity vector" gap plan.md identifies as the main thing worth
testing. Since trajectory.json already records each pedestrian's REAL walked
path (the oracle/ground-truth outcome, not a live prediction), building a
short future-horizon sequence from it is legitimate for training-label
generation the same way plan.md's oracle scorer already uses the full
ground-truth trajectory to label candidates -- it is NOT something a
deployed VLM/MLP arm would have access to at inference time, only the
labeling/scoring pipeline should consume this.
"""

import math
from typing import List, Optional, Tuple

import numpy as np

# Matches pedestrian_planner.PEDESTRIAN_BODY_RADIUS_M -- not imported directly
# to keep this module Isaac-Sim/ROS2-import-free and independently testable;
# keep these in sync if that constant changes.
DEFAULT_PEDESTRIAN_RADIUS_M = 0.25
DEFAULT_HORIZON_S = 2.0
DEFAULT_NUM_STEPS = 5
DEFAULT_RESOLUTION_M = 0.05


class GridMapInfo:
    """Duck-typed like pedestrian_planner's _MapInfo / obstacle_randomizer's
    _MapInfo -- origin + resolution, enough to convert between world (x, y)
    and grid (row, col)."""

    def __init__(self, origin_x: float, origin_y: float, resolution: float):
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.resolution = resolution

    def world_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        col = int(round((x - self.origin_x) / self.resolution))
        row = int(round((y - self.origin_y) / self.resolution))
        return row, col


def _interpolate_position(trajectory: List[dict], t: float) -> Optional[Tuple[float, float]]:
    """Linear interpolation between the two trajectory samples bracketing
    `t`. Clamps to the first/last sample outside the recorded range (holding
    position -- matches how a pedestrian sits still once its route finishes,
    see pedestrian_randomizer._drive_concurrently's own sampling behavior).
    Returns None for an empty trajectory."""
    if not trajectory:
        return None
    if t <= trajectory[0]["t"]:
        return trajectory[0]["x"], trajectory[0]["y"]
    if t >= trajectory[-1]["t"]:
        return trajectory[-1]["x"], trajectory[-1]["y"]

    for i in range(1, len(trajectory)):
        if trajectory[i]["t"] >= t:
            a, b = trajectory[i - 1], trajectory[i]
            span = b["t"] - a["t"]
            frac = (t - a["t"]) / span if span > 1e-9 else 0.0
            x = a["x"] + (b["x"] - a["x"]) * frac
            y = a["y"] + (b["y"] - a["y"]) * frac
            return x, y
    return trajectory[-1]["x"], trajectory[-1]["y"]  # unreachable, safety fallback


def _stamp_circle(grid: np.ndarray, cx: int, cy: int, radius_px: int) -> None:
    height, width = grid.shape
    x_lo, x_hi = max(0, cx - radius_px), min(width, cx + radius_px + 1)
    y_lo, y_hi = max(0, cy - radius_px), min(height, cy + radius_px + 1)
    for yy in range(y_lo, y_hi):
        for xx in range(x_lo, x_hi):
            if (xx - cx) ** 2 + (yy - cy) ** 2 <= radius_px ** 2:
                grid[yy, xx] = 1


def build_dynamic_occupancy_sequence(
    pedestrian_trajectories: List[List[dict]],
    query_t: float,
    hallway_bounds: dict,
    horizon_s: float = DEFAULT_HORIZON_S,
    num_steps: int = DEFAULT_NUM_STEPS,
    pedestrian_radius_m: float = DEFAULT_PEDESTRIAN_RADIUS_M,
    resolution: float = DEFAULT_RESOLUTION_M,
) -> Tuple[np.ndarray, List[float], GridMapInfo]:
    """Build one binary occupancy grid per future timestep from `query_t` to
    `query_t + horizon_s` (inclusive, `num_steps` evenly spaced samples,
    so step 0 IS query_t itself -- "now" is always included), each showing
    every pedestrian's real recorded position at that time (from
    `pedestrian_trajectories`, one trajectory list per pedestrian, as saved
    in trajectory.json) as an inflated circle. All pedestrians in a given
    timestep are unioned into that one grid, matching ground_truth.npy's
    same binary format so a caller can combine/compare them directly.

    Returns (grids, timestamps, map_info):
      grids: np.ndarray, shape (num_steps, height, width), dtype uint8
      timestamps: the `num_steps` absolute times each grid layer represents
      map_info: same origin/resolution convention as the static ground truth,
                for converting grid cells back to world coordinates
    """
    x_min, x_max = hallway_bounds["x_min"], hallway_bounds["x_max"]
    y_min, y_max = hallway_bounds["y_min"], hallway_bounds["y_max"]
    width = int(math.ceil((x_max - x_min) / resolution)) + 1
    height = int(math.ceil((y_max - y_min) / resolution)) + 1
    map_info = GridMapInfo(x_min, y_min, resolution)
    radius_px = max(1, int(round(pedestrian_radius_m / resolution)))

    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    timestamps = [
        query_t + (horizon_s * i / (num_steps - 1) if num_steps > 1 else 0.0)
        for i in range(num_steps)
    ]

    grids = np.zeros((num_steps, height, width), dtype=np.uint8)
    for step_idx, t in enumerate(timestamps):
        for trajectory in pedestrian_trajectories:
            pos = _interpolate_position(trajectory, t)
            if pos is None:
                continue
            row, col = map_info.world_to_pixel(pos[0], pos[1])
            _stamp_circle(grids[step_idx], col, row, radius_px)

    return grids, timestamps, map_info


def save_dynamic_occupancy_sequence(
    grids: np.ndarray, timestamps: List[float], map_info: GridMapInfo, output_path_prefix: str,
) -> None:
    """Save a build_dynamic_occupancy_sequence(...) result as
    <prefix>.npy (the (num_steps, height, width) array) + <prefix>.json
    (timestamps + the same resolution/origin/width/height metadata shape as
    ground_truth.json, so the same reader code can load either)."""
    import json

    np.save(f"{output_path_prefix}.npy", grids)
    with open(f"{output_path_prefix}.json", 'w') as f:
        json.dump({
            'resolution': map_info.resolution,
            'origin_x': map_info.origin_x,
            'origin_y': map_info.origin_y,
            'width': int(grids.shape[2]),
            'height': int(grids.shape[1]),
            'num_steps': int(grids.shape[0]),
            'timestamps': timestamps,
        }, f, indent=2)
