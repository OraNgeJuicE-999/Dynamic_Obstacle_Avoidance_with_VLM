"""Actor trajectory schema + JSON persistence.

A "trajectory" is one actor's (pedestrian or car) recorded pose over time
during a single take: a list of (t, x, y, yaw) samples. This is the shared
shape both the real Isaac Sim recorder (trajectory_recorder.record_live_trajectory,
not yet wired up -- see that module) and the analytic stand-in
(trajectory_recorder.trajectory_from_commands, usable today against
pedestrian_planner.plan_pedestrian_commands' output) produce -- so everything
downstream (decision_sampler, candidate_path_generator, oracle_scorer) only
ever needs to know this one shape, regardless of which source built it.
"""

from dataclasses import asdict, dataclass, field
import json
import os
from typing import List


@dataclass
class TrajectorySample:
    t: float
    x: float
    y: float
    yaw: float


@dataclass
class ActorTrajectory:
    actor_id: str  # e.g. "pedestrian_0", "car"
    samples: List[TrajectorySample] = field(default_factory=list)

    def pose_at(self, t: float) -> TrajectorySample:
        """Nearest-neighbor pose lookup -- fine for the sampling density
        decision_sampler/oracle_scorer use (dt on the order of the control
        loop rate); switch to linear interpolation between the two bracketing
        samples if that density ever gets coarse enough to matter."""
        if not self.samples:
            raise ValueError(f"'{self.actor_id}' trajectory has no samples")
        return min(self.samples, key=lambda s: abs(s.t - t))

    def duration(self) -> float:
        return self.samples[-1].t if self.samples else 0.0


@dataclass
class SceneTrajectories:
    """All actors' trajectories for one take."""

    scene_index: int
    actors: List[ActorTrajectory] = field(default_factory=list)

    def get(self, actor_id: str) -> ActorTrajectory:
        for actor in self.actors:
            if actor.actor_id == actor_id:
                return actor
        raise KeyError(f"No trajectory recorded for actor '{actor_id}'")


def save_trajectories(trajectories: SceneTrajectories, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(asdict(trajectories), f, indent=2)


def load_trajectories(input_path: str) -> SceneTrajectories:
    with open(input_path) as f:
        data = json.load(f)

    data["actors"] = [
        ActorTrajectory(
            actor_id=a["actor_id"],
            samples=[TrajectorySample(**s) for s in a["samples"]],
        )
        for a in data["actors"]
    ]
    return SceneTrajectories(**data)
