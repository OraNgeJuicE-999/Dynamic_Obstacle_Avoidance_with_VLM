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
    prim_path: Optional[str] = None     # USD prim path, filled in once spawned (see StaticObstacleSpec)
    # Actual walked path over time, filled in once driven -- one dict per
    # sample: {"t": seconds since this pedestrian's route started, "x", "y",
    # "yaw" (RADIANS -- matches training.common.trajectory_io.TrajectorySample's
    # convention; this was "yaw_deg" until 2026-09-23, which silently broke
    # loading as degrees-read-as-radians). This is what gets converted into
    # a training.common.trajectory_io.SceneTrajectories and written to
    # trajectory.json for dataset_builder.py -- see
    # save_pedestrian_trajectories and pedestrian_randomizer._drive_concurrently.
    trajectory: Optional[List[dict]] = None

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


def save_pedestrian_trajectories(pedestrians: List[PedestrianPathSpec], scene_index: int, output_path: str) -> None:
    """Write each pedestrian's actual walked path over time to `output_path`
    as a training.common.trajectory_io.SceneTrajectories JSON -- what
    dataset_builder.py's load_trajectories() reads for oracle_scorer's "is
    the car near the pedestrian at time t" checks. Call this only after the
    pedestrians have actually been driven (spec.trajectory populated by
    pedestrian_randomizer.spawn_pedestrians), alongside save_scenario and
    obstacle_randomizer.build_ground_truth_occupancy_grid's outputs for the
    same scene.

    Deliberately imports and reuses training's own schema/save function
    (added 2026-09-23) rather than writing a similar-but-not-identical shape
    from scratch -- the whole reason there was a mismatch before
    ({"pedestrians": [{"character_id", "asset_id", "trajectory"}]} vs. the
    real SceneTrajectories{scene_index, actors: [{actor_id, samples}]}) was
    two independently-hand-rolled schemas drifting apart. `training/` has no
    Isaac Sim/ROS2 dependency of its own (pure dataclasses + json), so this
    import is safe from this package.
    """
    import sys
    _repo_root = "/home/itri/jetracer_nav"
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from training.common.trajectory_io import ActorTrajectory, SceneTrajectories, TrajectorySample, save_trajectories

    actors = [
        ActorTrajectory(
            actor_id=ped.character_id,
            samples=[TrajectorySample(t=s["t"], x=s["x"], y=s["y"], yaw=s["yaw"]) for s in (ped.trajectory or [])],
        )
        for ped in pedestrians
    ]
    save_trajectories(SceneTrajectories(scene_index=scene_index, actors=actors), output_path)


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
