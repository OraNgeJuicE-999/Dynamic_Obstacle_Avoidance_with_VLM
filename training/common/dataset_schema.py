"""Shared schema for the assembled training dataset.

Every stage in data_pipeline/ produces or consumes these shapes, and every
arms/train_*.py script reads the final TrainingExample rows through this
module instead of redefining its own format -- so a schema change only ever
has to happen in one place.
"""

from dataclasses import asdict, dataclass, field
import json
import os
from typing import List, Optional, Tuple


@dataclass
class DecisionPoint:
    """One instant during a take at which the car has to pick a path."""

    scene_index: int
    t: float
    car_x: float
    car_y: float
    car_yaw: float
    goal_x: float
    goal_y: float


@dataclass
class CandidatePath:
    candidate_id: str
    waypoints: List[Tuple[float, float]]  # world-space, A*-smoothed
    # True for the random-subgoal / perturbation negatives plan.md specifies
    # for training -- not something a real deployment would ever consider,
    # kept flagged so eval code can exclude them when reporting what the
    # model would actually choose from.
    is_negative_example: bool = False


@dataclass
class OracleLabel:
    candidate_id: str
    collision_free: bool
    min_pedestrian_clearance_m: float
    path_complexity: float  # path length / straight-line distance (1.0 = straight)
    goal_progress: float  # fraction of remaining straight-line distance to goal closed, in [0, 1]
    score: float  # combined weighted score -- the actual training target


@dataclass
class TrainingExample:
    decision: DecisionPoint
    candidates: List[CandidatePath]
    labels: List[OracleLabel]  # same order/length as candidates
    camera_image_path: Optional[str] = None  # filled in for VLM/YOLO arms; None for lidar-only


@dataclass
class Dataset:
    examples: List[TrainingExample] = field(default_factory=list)


def save_dataset(dataset: Dataset, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(asdict(dataset), f, indent=2)


def load_dataset(input_path: str) -> Dataset:
    with open(input_path) as f:
        data = json.load(f)

    examples = []
    for ex in data["examples"]:
        examples.append(
            TrainingExample(
                decision=DecisionPoint(**ex["decision"]),
                candidates=[
                    CandidatePath(
                        candidate_id=c["candidate_id"],
                        waypoints=[tuple(wp) for wp in c["waypoints"]],
                        is_negative_example=c.get("is_negative_example", False),
                    )
                    for c in ex["candidates"]
                ],
                labels=[OracleLabel(**label) for label in ex["labels"]],
                camera_image_path=ex.get("camera_image_path"),
            )
        )

    return Dataset(examples=examples)
