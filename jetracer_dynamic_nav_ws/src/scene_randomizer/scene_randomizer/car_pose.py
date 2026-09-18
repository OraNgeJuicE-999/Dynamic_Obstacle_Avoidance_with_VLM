"""JetRacer start-pose selection.

Unlike obstacles/pedestrians, start poses are NOT sampled freely — they're a
small fixed set read from config/start_poses.yaml (see that file's header for
why). This module just picks one per scenario and places the car there.

`apply_car_start_pose` depends on Isaac Sim's `omni.*` APIs, same caveat as
obstacle_randomizer.py / pedestrian_randomizer.py: only importable inside
Isaac Sim's own Python process.
"""

import math
import random
from typing import List

import numpy as np
import yaml
from tf_transformations import quaternion_from_euler

from scene_randomizer.scenario_io import CarStartPoseSpec

# TODO: verify this against the actual USD stage hierarchy once the JetRacer
# is loaded in the hallway scene — this is a placeholder, not a confirmed path.
JETRACER_PRIM_PATH = "/World/jetracer"


def load_start_poses(yaml_path: str) -> List[dict]:
    """Load the candidate start poses from `yaml_path`."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    start_poses = data.get("start_poses", [])
    if not start_poses:
        raise ValueError(f"No start_poses found in {yaml_path}")

    return start_poses


def sample_start_pose(rng: random.Random, start_poses: List[dict]) -> CarStartPoseSpec:
    """Pick one start pose for this scenario.

    Uniform random choice for now. NOTE: if you want every pose guaranteed to
    get covered across a batch of runs instead of leaving it to chance,
    swap this for a round-robin index tracked by the caller instead of `rng`.
    """
    pose_dict = rng.choice(start_poses)
    return CarStartPoseSpec(**pose_dict)


def apply_car_start_pose(pose: CarStartPoseSpec) -> None:
    """Move the JetRacer articulation in Isaac Sim to `pose`.

    NOTE: Articulation's exact API (in particular whether it needs
    `.initialize()` called first, and the joint-velocity-zeroing calls) varies
    across Isaac Sim versions — verify against yours.
    """
    from omni.isaac.core.articulations import Articulation

    articulation = Articulation(JETRACER_PRIM_PATH)

    # tf_transformations returns (x, y, z, w); Isaac Sim wants (w, x, y, z).
    qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, math.radians(pose.yaw_deg))
    orientation = np.array([qw, qx, qy, qz])

    articulation.set_world_pose(
        position=np.array([pose.x, pose.y, 0.0]),
        orientation=orientation,
    )

    # Zero out velocity/joint state so the car doesn't carry momentum over
    # from wherever it was before this reset.
    articulation.set_linear_velocity(np.zeros(3))
    articulation.set_angular_velocity(np.zeros(3))
    articulation.set_joint_velocities(np.zeros(articulation.num_dof))
