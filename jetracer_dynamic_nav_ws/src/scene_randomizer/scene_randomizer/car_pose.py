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

# Verified 2026-09-24 directly against the live hallway stage: /leatherback_ROS
# (not /World/jetracer, which was never a real prim) is the actual car -- its
# UsdPhysics.ArticulationRootAPI sits right on this prim, and
# SingleArticulation("/leatherback_ROS").initialize() successfully reports 26
# DOF (full suspension + wheel articulation). set_world_pose() was confirmed
# to actually move it (and restore correctly) in a live test.
JETRACER_PRIM_PATH = "/leatherback_ROS"


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

    Isaac Sim 6.0's articulation API lives under `isaacsim.core.prims`, not
    the older `omni.isaac.core.articulations` (which doesn't exist in this
    install -- confirmed via ModuleNotFoundError). `SingleArticulation` (the
    singular-prim wrapper, as opposed to the batched `Articulation`) matches
    this function's one-prim-at-a-time usage and its API
    (get/set_world_pose, get/set_linear_velocity, etc.) directly, no reshaping
    needed. `.initialize()` IS required first -- confirmed live: num_dof is 0
    until then.
    """
    from isaacsim.core.prims import SingleArticulation

    articulation = SingleArticulation(JETRACER_PRIM_PATH)
    articulation.initialize()

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
