"""Render top-down PNG previews of one generated scene, straight from its
saved files -- no Isaac Sim needed. Two previews, answering two different
questions:

  preview_bbox.png         -- full pedestrian trajectories over the FULL
                               bounding-box ground truth (ground_truth.npy).
                               Pedestrians can't duck under a table, so this
                               is the map that matters for them -- look at
                               whether a path stays in the hallway, moves
                               smoothly, and doesn't cut through an obstacle.
  preview_lidar_height.png -- a single random (or chosen) timestamp's
                               pedestrian positions over the LIDAR-HEIGHT
                               ground truth (lidar_height_ground_truth.npy)
                               -- the map that matters for the car, which
                               CAN pass under a table. Marks where each
                               pedestrian actually was at that one instant,
                               not their whole path.

Usage:
    python3 visualize_scene.py /path/to/scenes_tester/scene_01
    python3 visualize_scene.py /path/to/scenes_tester/scene_01 --frame-t 12.4
    python3 visualize_scene.py /path/to/scenes_tester/scene_01 --seed 7

Standalone script, no ROS2/Isaac Sim imports -- only needs json/numpy/matplotlib,
so it runs in a plain Python environment (this repo's venv, a laptop, a CI
job, wherever), independent of whatever Isaac Sim session generated the data.
"""

import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Okabe-Ito qualitative palette -- colorblind-safe, fixed assignment order
# (never cycled/reassigned based on how many pedestrians are in a given
# scene, so pedestrian_0 is always the same color across every scene's plot).
PEDESTRIAN_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
OBSTACLE_COLOR = "#666666"
GRID_BLOCKED_COLOR = "#D9D9D9"


def _load_scene(scene_dir: str) -> dict:
    with open(os.path.join(scene_dir, "scenario.json")) as f:
        scenario = json.load(f)
    with open(os.path.join(scene_dir, "ground_truth.json")) as f:
        grid_meta = json.load(f)
    grid = np.load(os.path.join(scene_dir, "ground_truth.npy"))

    lidar_grid, lidar_meta = None, None
    lidar_npy = os.path.join(scene_dir, "lidar_height_ground_truth.npy")
    if os.path.exists(lidar_npy):
        lidar_grid = np.load(lidar_npy)
        with open(os.path.join(scene_dir, "lidar_height_ground_truth.json")) as f:
            lidar_meta = json.load(f)

    trajectories = None
    trajectory_path = os.path.join(scene_dir, "trajectory.json")
    if os.path.exists(trajectory_path):
        with open(trajectory_path) as f:
            trajectories = json.load(f)

    return {
        "scenario": scenario, "grid": grid, "grid_meta": grid_meta,
        "lidar_grid": lidar_grid, "lidar_meta": lidar_meta,
        "trajectories": trajectories,
    }


def _grid_extent(meta: dict):
    x0, y0 = meta["origin_x"], meta["origin_y"]
    res = meta["resolution"]
    return x0, y0, x0 + meta["width"] * res, y0 + meta["height"] * res


def _draw_hallway_and_obstacles(ax, scenario, x0, y0, x1, y1):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                            edgecolor="black", linewidth=1.5, zorder=3))
    for obstacle in scenario.get("static_obstacles", []):
        fmin, fmax = obstacle.get("footprint_min"), obstacle.get("footprint_max")
        if fmin is None or fmax is None:
            continue
        name = os.path.splitext(os.path.basename(obstacle["asset_path"]))[0]
        ax.add_patch(Rectangle(
            (fmin[0], fmin[1]), fmax[0] - fmin[0], fmax[1] - fmin[1],
            facecolor=OBSTACLE_COLOR, edgecolor="none", alpha=0.5, zorder=2,
        ))
        ax.annotate(name, (obstacle["position"][0], obstacle["position"][1]),
                    fontsize=6, ha="center", va="center", color="black", zorder=4)


def _interpolate_position(trajectory, t):
    """Same convention as dynamic_occupancy._interpolate_position -- kept as
    a local copy so this script stays import-free of the rest of the
    package and runnable standalone from anywhere."""
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
            return a["x"] + (b["x"] - a["x"]) * frac, a["y"] + (b["y"] - a["y"]) * frac
    return trajectory[-1]["x"], trajectory[-1]["y"]


def render_bbox_preview(scene_dir: str, data: dict, output_path: str) -> None:
    """Full pedestrian trajectories over the FULL bounding-box ground truth
    -- what matters for pedestrians, who can't duck under a table."""
    scenario, grid, meta = data["scenario"], data["grid"], data["grid_meta"]
    x0, y0, x1, y1 = _grid_extent(meta)

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.imshow(
        grid, extent=(x0, x1, y0, y1), origin="lower",
        cmap=plt.matplotlib.colors.ListedColormap(["white", GRID_BLOCKED_COLOR]),
        vmin=0, vmax=1, interpolation="nearest", zorder=0,
    )
    _draw_hallway_and_obstacles(ax, scenario, x0, y0, x1, y1)

    traj_by_id = {a["actor_id"]: a["samples"] for a in data["trajectories"]["actors"]} if data["trajectories"] else {}
    for i, ped_spec in enumerate(scenario.get("pedestrians", [])):
        color = PEDESTRIAN_COLORS[i % len(PEDESTRIAN_COLORS)]
        traj = traj_by_id.get(ped_spec["character_id"])
        if not traj:
            continue
        xs, ys = [p["x"] for p in traj], [p["y"] for p in traj]
        ax.plot(xs, ys, color=color, linewidth=2, zorder=5, label=ped_spec["character_id"], solid_capstyle="round")
        ax.plot(xs[0], ys[0], marker="o", color=color, markersize=8, markeredgecolor="white", markeredgewidth=1, zorder=6)
        ax.plot(xs[-1], ys[-1], marker="s", color=color, markersize=8, markeredgecolor="white", markeredgewidth=1, zorder=6)

    ax.set_xlim(x0 - 0.3, x1 + 0.3)
    ax.set_ylim(y0 - 0.5, y1 + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"{os.path.basename(scene_dir)} -- FULL BBOX (pedestrian view) -- seed {scenario.get('seed')} "
        f"({len(scenario.get('static_obstacles', []))} obstacles, {len(scenario.get('pedestrians', []))} pedestrians) "
        f"[circle=start, square=end]"
    )
    if traj_by_id:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_lidar_height_frame(scene_dir: str, data: dict, output_path: str, frame_t: float = None, rng: random.Random = None) -> float:
    """One random (or chosen) timestamp's pedestrian positions over the
    LIDAR-HEIGHT ground truth -- what matters for the car, which CAN pass
    under a table. Returns the timestamp actually used.
    """
    if data["lidar_grid"] is None:
        raise FileNotFoundError(
            f"No lidar_height_ground_truth.npy in {scene_dir} -- generate it via "
            "obstacle_randomizer.build_lidar_height_occupancy_grid (needs Isaac Sim, Play active)."
        )
    scenario, grid, meta = data["scenario"], data["lidar_grid"], data["lidar_meta"]
    x0, y0, x1, y1 = _grid_extent(meta)

    trajectories = data["trajectories"]["actors"] if data["trajectories"] else []
    if frame_t is None:
        rng = rng or random.Random()
        all_t = [s["t"] for p in trajectories for s in p["samples"]]
        frame_t = rng.uniform(min(all_t), max(all_t)) if all_t else 0.0

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="black", linewidth=1.5, zorder=3))

    # Fake lidar scan points, SLAM-map style: every blocked grid cell plotted
    # as its own small dot -- NOT the bounding box, and NOT reduced to one
    # centroid per object (tried that on 2026-09-23, looked wrong -- a
    # printer should read as a filled square of points, a table as 4
    # leg-shaped clusters, a chair as however many separate leg/pedestal
    # pieces its real collision geometry has, not one dot standing in for
    # the whole shape). This is dense enough that real shapes are
    # recognizable, sparse enough (one point per grid cell, not a filled
    # raster) to still look like a point-cloud scan rather than a solid block.
    ys_idx, xs_idx = np.nonzero(grid)
    if len(xs_idx):
        world_x = x0 + xs_idx * meta["resolution"]
        world_y = y0 + ys_idx * meta["resolution"]
        ax.scatter(world_x, world_y, s=3, c="black", marker=".", zorder=4, label=f"scan points ({len(xs_idx)})")

    # Pedestrians as two "leg columns" (a real lidar at this height sees legs,
    # not a torso) -- small points offset from the trajectory's recorded
    # position, perpendicular to the recorded facing direction (yaw, radians
    # -- matches training.common.trajectory_io.TrajectorySample's convention),
    # not a real per-leg geometry query (trajectory.json doesn't record
    # that) -- an approximation, same spirit as the rest of this "fake scan"
    # preview.
    LEG_OFFSET_M = 0.08
    for i, ped in enumerate(trajectories):
        color = PEDESTRIAN_COLORS[i % len(PEDESTRIAN_COLORS)]
        pos = _interpolate_position(ped["samples"], frame_t)
        if pos is None:
            continue
        yaw = min(ped["samples"], key=lambda s: abs(s["t"] - frame_t)).get("yaw") or 0.0
        perp = yaw + np.pi / 2.0
        dx, dy = np.cos(perp) * LEG_OFFSET_M, np.sin(perp) * LEG_OFFSET_M
        leg_xs = [pos[0] - dx, pos[0] + dx]
        leg_ys = [pos[1] - dy, pos[1] + dy]
        ax.scatter(leg_xs, leg_ys, s=60, color=color, marker="o",
                   edgecolors="white", linewidths=1, zorder=6, label=ped["actor_id"])

    ax.set_xlim(x0 - 0.3, x1 + 0.3)
    ax.set_ylim(y0 - 0.5, y1 + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"{os.path.basename(scene_dir)} -- LIDAR HEIGHT fake scan (car view) -- t={frame_t:.2f}s -- "
        f"seed {scenario.get('seed')}"
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return frame_t


def render(scene_dir: str, out_bbox: str = None, out_lidar: str = None, frame_t: float = None, seed: int = None) -> dict:
    """Render both previews. Returns {"bbox": path, "lidar_height": path or None, "frame_t": float or None}."""
    data = _load_scene(scene_dir)
    out_bbox = out_bbox or os.path.join(scene_dir, "preview_bbox.png")
    render_bbox_preview(scene_dir, data, out_bbox)
    result = {"bbox": out_bbox, "lidar_height": None, "frame_t": None}

    if data["lidar_grid"] is not None:
        out_lidar = out_lidar or os.path.join(scene_dir, "preview_lidar_height.png")
        rng = random.Random(seed) if seed is not None else None
        used_t = render_lidar_height_frame(scene_dir, data, out_lidar, frame_t=frame_t, rng=rng)
        result["lidar_height"] = out_lidar
        result["frame_t"] = used_t

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_dir", help="Path to a scene_NN/ subfolder")
    parser.add_argument("--out-bbox", default=None, help="Output path for the bbox preview (default: <scene_dir>/preview_bbox.png)")
    parser.add_argument("--out-lidar", default=None, help="Output path for the lidar-height frame preview (default: <scene_dir>/preview_lidar_height.png)")
    parser.add_argument("--frame-t", type=float, default=None, help="Timestamp to use for the lidar-height frame (default: random)")
    parser.add_argument("--seed", type=int, default=None, help="Seed the random frame-time pick, for reproducible previews")
    args = parser.parse_args()
    result = render(args.scene_dir, args.out_bbox, args.out_lidar, args.frame_t, args.seed)
    print(f"Saved {result['bbox']}")
    if result["lidar_height"]:
        print(f"Saved {result['lidar_height']} (frame t={result['frame_t']:.2f}s)")
    else:
        print("No lidar_height_ground_truth.npy found -- skipped that preview")


if __name__ == "__main__":
    main()
