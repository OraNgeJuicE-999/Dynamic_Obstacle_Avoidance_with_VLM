"""Dataset assembly: orchestrates decision_sampler -> candidate_path_generator
-> oracle_scorer (and, where available, camera_capture) across every scenario
to build the shared Dataset every arms/train_*.py script trains from.

Pedestrian trajectories are read via trajectory_io -- pass in whichever
source is available (trajectory_recorder.trajectory_from_commands' analytic
stand-in today, or a real recorded trajectory once that's wired up). This
module doesn't care which, since both produce the same ActorTrajectory shape.

Run directly (`python -m training.data_pipeline.dataset_builder`) to build a
dataset from every scene_NN.json / scene_NN_trajectory.json pair already on
disk in scenes/.
"""

import glob
import json
import os
from typing import Callable, List, Optional

from training.common.dataset_schema import Dataset, TrainingExample, save_dataset
from training.common.trajectory_io import SceneTrajectories, load_trajectories
from training.data_pipeline.candidate_path_generator import generate_candidates
from training.data_pipeline.decision_sampler import sample_decision_points
from training.data_pipeline.oracle_scorer import score_all


def build_dataset_for_scene(
    scene_index: int,
    hallway_bounds: dict,
    static_obstacle_footprints: List[dict],
    trajectories: SceneTrajectories,
    goal_xy: tuple,
    capture_camera_frame: Optional[Callable] = None,  # e.g. camera_capture.capture, once implemented
) -> List[TrainingExample]:
    # ALL pedestrians in the scene, not just the first one found -- a real
    # scene can (and usually does) have more than one (see
    # scene_randomizer's spawn_pedestrians, which drives every pedestrian in
    # a take concurrently). Fixed 2026-09-23: this used to grab only
    # `next(a for a in trajectories.actors if a.actor_id.startswith("pedestrian"))`,
    # silently ignoring every pedestrian after the first.
    pedestrians = [a for a in trajectories.actors if a.actor_id.startswith("pedestrian")]
    if not pedestrians:
        return []

    decisions = sample_decision_points(
        scene_index, hallway_bounds, trajectories, goal_xy, static_obstacle_footprints=static_obstacle_footprints,
    )

    examples = []
    for decision in decisions:
        candidates = generate_candidates(decision, hallway_bounds, static_obstacle_footprints)
        if not candidates:
            continue
        labels = score_all(candidates, decision, static_obstacle_footprints, pedestrians)
        image_path = capture_camera_frame(decision) if capture_camera_frame else None
        examples.append(
            TrainingExample(decision=decision, candidates=candidates, labels=labels, camera_image_path=image_path)
        )

    return examples


def build_dataset(
    scenes_dir: str,
    hallway_bounds: dict,
    goal_xy: tuple,
    output_path: str,
    capture_camera_frame: Optional[Callable] = None,
) -> None:
    """Build the full dataset across every scene_NN/ subfolder found in
    `scenes_dir` that has both a scenario.json and a trajectory.json
    (scenes without a trajectory file yet -- e.g. before pedestrians have
    been driven for them -- are skipped, not errored on).

    Layout fixed 2026-09-23: this used to look for flat
    scene_NN.json/scene_NN_trajectory.json files directly in `scenes_dir` --
    the real scene_randomizer output (obstacle_randomizer.py +
    pedestrian_randomizer.py) is per-scene subfolders
    (scene_NN/scenario.json, scene_NN/trajectory.json, plus ground_truth
    and lidar_height_ground_truth grids and a scene.usd snapshot), so none
    of that data was ever actually found by this function before.
    """
    dataset = Dataset()

    for scenario_path in sorted(glob.glob(os.path.join(scenes_dir, "scene_*", "scenario.json"))):
        scene_dir = os.path.dirname(scenario_path)
        basename = os.path.basename(scene_dir)  # "scene_NN"
        scene_index = int(basename.split("_")[1])

        with open(scenario_path) as f:
            scenario = json.load(f)
        static_obstacle_footprints = [
            o for o in scenario["static_obstacles"] if o.get("footprint_min") is not None
        ]

        trajectory_path = os.path.join(scene_dir, "trajectory.json")
        if not os.path.exists(trajectory_path):
            continue
        trajectories = load_trajectories(trajectory_path)

        examples = build_dataset_for_scene(
            scene_index, hallway_bounds, static_obstacle_footprints, trajectories, goal_xy, capture_camera_frame,
        )
        dataset.examples.extend(examples)

    save_dataset(dataset, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes-dir", default="/home/itri/jetracer_nav/scenes")
    parser.add_argument("--output", default="/home/itri/jetracer_nav/training_data/dataset.json")
    parser.add_argument("--goal-x", type=float, default=7.0)
    parser.add_argument("--goal-y", type=float, default=0.0)
    args = parser.parse_args()

    # y_min/y_max corrected 2026-09-23 to match the REAL measured inner wall
    # (was -1.16/1.16, the outer wall reference -- see
    # scene_randomizer_node.py's hallway_bounds param for the full
    # dense-raycast measurement this now matches).
    HALLWAY_BOUNDS = {"x_min": -7.93, "x_max": 7.93, "y_min": -0.98, "y_max": 0.94}
    build_dataset(args.scenes_dir, HALLWAY_BOUNDS, (args.goal_x, args.goal_y), args.output)
    print(f"Dataset written to {args.output}")
