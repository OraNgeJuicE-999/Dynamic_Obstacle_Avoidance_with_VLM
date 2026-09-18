"""Scenario data structures + JSON persistence.

A "scenario" is everything needed to reproduce one scene instance: the RNG seed
that produced it, the sampled static obstacle placements, and the sampled
pedestrian paths. Downstream (A* candidate-path generation, and each of the
VLM/MLP/world-model scorers) all key off this same JSON file so every model
sees an identical scenario.
"""

from dataclasses import asdict, dataclass, field
import os
import json
import time
from typing import List, Optional


@dataclass
class StaticObstacleSpec:
    """One static prop placed in the hallway."""

    asset_path: str          # .glb/.usd source, e.g. "hallway_assets/static_obstacles/chair.glb"
    position: List[float]    # [x, y, z] in world/stage units
    yaw_deg: float
    scale: float = 1.0
    prim_path: Optional[str] = None  # filled in once spawned on the USD stage
    # World-space [x, y] footprint corners, filled in once spawned — computed
    # from the prim's actual USD bounding box (already accounts for yaw_deg
    # since it's measured after the prim is placed/rotated on the stage), not
    # a uniform-radius approximation. See obstacle_randomizer.spawn_static_obstacles.
    footprint_min: Optional[List[float]] = None
    footprint_max: Optional[List[float]] = None


@dataclass
class PedestrianPathSpec:
    """One pedestrian's motion for a single take.

    `heading` is one of "straight" | "diagonal_left" | "diagonal_right" — the
    hallway's long axis is the reference direction. `turn_event` / `stop_event`
    are optional and mutually exclusive: a pedestrian either goes straight for
    the whole take, makes exactly one turn partway, or walks then stops and
    stays in place. Keeping this to "at most one event" is what keeps generated
    routes from looking erratic.
    """

    asset_id: str             # character/asset identifier for omni.anim.people
    start_position: List[float]  # [x, y, z]
    heading: str
    speed: float              # m/s
    turn_event: Optional[dict] = None   # e.g. {"at_fraction": 0.5, "new_heading": "diagonal_left"}
    stop_event: Optional[dict] = None   # e.g. {"at_fraction": 0.6, "duration_s": 3.0}
    character_id: Optional[str] = None  # filled in once spawned

@dataclass
class CarStartPoseSpec:
    name: str
    x: float
    y: float
    yaw_deg: float

@dataclass
class Scenario:
    seed: int
    static_obstacles: List[StaticObstacleSpec] = field(default_factory=list)
    pedestrians: List[PedestrianPathSpec] = field(default_factory=list)
    car_start_pose: Optional[CarStartPoseSpec] = None
    created_at: float = field(default_factory=time.time)


def new_seed() -> int:
    """Return a fresh RNG seed for a new scenario.

    TODO: decide the seed source (time-based vs. an incrementing counter read
    from a run manifest) — time-based is simplest but harder to re-derive by
    hand if you want to name/re-request a specific past scenario.
    """
    return time.time_ns() % (2 ** 32)


def save_scenario(scenario: Scenario, output_path: str) -> None:
    """Write `scenario` to `output_path` as JSON.

    TODO: implement with json.dump(asdict(scenario), ...). Decide the naming
    scheme for output files (e.g. scenario_<seed>_<take_idx>.json) so full
    rerolls and same-layout "takes" are distinguishable on disk.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(asdict(scenario), f, indent=2)


def load_scenario(input_path: str) -> Scenario:
    """Read a Scenario back from JSON (for replaying/inspecting a past run).

    TODO: implement with json.load(...) + reconstructing the dataclasses.
    """
    with open(input_path) as f: 
        data = json.load(f)

    data['static_obstacles'] = [StaticObstacleSpec(**d) for d in data['static_obstacles']]
    data['pedestrians'] = [PedestrianPathSpec(**d) for d in data['pedestrians']]
    if data['car_start_pose'] is not None:
        data['car_start_pose'] = CarStartPoseSpec(**data['car_start_pose'])

    return Scenario(**data)
