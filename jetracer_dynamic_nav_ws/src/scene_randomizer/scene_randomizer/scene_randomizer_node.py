"""ROS2 node exposing scene (re)generation as services.

IMPORTANT — where this runs: obstacle_randomizer.py and pedestrian_randomizer.py
import Isaac Sim's `omni.*` APIs (Replicator's stage utils, omni.anim.people),
which only exist inside Isaac Sim's own Python process. That means this node
cannot be launched the normal way (`ros2 run scene_randomizer scene_randomizer_node`
from a plain ROS2 shell) — it has to run inside Isaac Sim, either:
  - as a standalone script executed via Isaac Sim's bundled interpreter
    (`<isaac_sim_root>/python.sh .../scene_randomizer_node.py`), or
  - loaded as a Kit extension enabled from Isaac Sim's Extension Manager,
so that `rclpy` (via Isaac Sim's ROS2 Bridge extension) and `omni.*` are both
importable in the same process. Sourcing this package's colcon install space
first (`source install/setup.bash`) makes `scene_randomizer` importable there.

Two services, matching the "regenerate vs. take" distinction from the project
plan:
  - /regenerate_scene: full reroll — new seed, new static layout, new
    pedestrian paths.
  - /new_take: keep the current static layout, only reroll pedestrian paths
    (a new "take" of the same scene).
"""

import json
import os
import random
import re

import numpy as np
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from scene_randomizer.car_pose import apply_car_start_pose, load_start_poses, sample_start_pose
from scene_randomizer.obstacle_randomizer import (
    build_ground_truth_occupancy_grid,
    build_lidar_height_occupancy_grid,
    clear_static_obstacles,
    load_obstacle_prefab_library,
    sample_obstacle_positions,
    spawn_static_obstacles,
)
from scene_randomizer.pedestrian_randomizer import (
    clear_pedestrians,
    sample_pedestrian_paths,
    spawn_pedestrians,
)
from scene_randomizer.scenario_io import Scenario, new_seed, save_pedestrian_trajectories, save_scenario


class SceneRandomizerNode(Node):

    def __init__(self):
      super().__init__('scene_randomizer')

      self.declare_parameter('num_static_obstacles', 8)
      self.declare_parameter('num_pedestrians', 3)
      self.declare_parameter('assets_dir', '/home/itri/jetracer_nav/hallway_assets/static_obstacles')
      self.declare_parameter('output_dir', '/home/itri/jetracer_nav/scenes')
      self.declare_parameter('start_poses_yaml', '')
      # y_min/y_max corrected 2026-09-23 via direct raycast against the real
      # wall collision geometry (was -1.16/1.16, the *outer* wall reference --
      # objects/pedestrians could be placed up to ~17cm INTO the actual wall).
      # A first pass (5 x-samples, 1 height) found 0.988/-1.026, but a much
      # denser follow-up scan (51 x-positions x 6 heights, 0.05m-1.2m off the
      # floor) found the TRUE worst case tighter still: y=0.9758 (near-floor,
      # x=-6.6 -- likely a baseboard/kickplate) and y=-1.0137 (higher up,
      # x=-5.1, a different height than the +Y tightest point -- the wall
      # profile is NOT a simple flat vertical surface). 0.94/-0.98 leaves
      # ~3.5cm margin below that measured worst case on both sides.
      # x_max=7.93 was already correct (real wall measured at 7.933). x_min
      # has NO measurable wall at all -- raycasts found nothing within 5m even
      # from x=-6 -- the -X end is a genuinely open doorway/architectural
      # transition, not a solid wall, so it's left as the existing authored
      # value rather than "corrected" against something that doesn't exist.
      self.declare_parameter(
          'hallway_bounds',
          [-7.93, 7.93, -0.98, 0.94],  # x_min, x_max, y_min, y_max
      )
      self.declare_parameter('character_pool', [''])

      self.asset_dir = self.get_parameter('assets_dir').value
      self.num_ped = self.get_parameter('num_pedestrians').value
      self.num_static_obstacles = self.get_parameter('num_static_obstacles').value
      self.output_dir = self.get_parameter('output_dir').value
      self.start_pose_yaml = self.get_parameter('start_poses_yaml').value
      self.character_pool = self.get_parameter('character_pool').value
      self._rng = random.Random()
      self._current_scenario: Scenario = None
      self._prefab_library = []
      self._start_poses = []

      # Continue numbering from existing scene_NN/ subfolders rather than
      # overwriting scene_01 every time this node restarts. Each scene gets
      # its own subfolder (scenario.json, ground_truth.npy/.json,
      # trajectory.json, and a scene.usd snapshot) rather than flat
      # scene_NN_*.* files, so everything about one scene lives together.
      self._scene_index = 0
      if os.path.isdir(self.output_dir):
          for name in os.listdir(self.output_dir):
              match = re.fullmatch(r"scene_(\d+)", name)
              if match and os.path.isdir(os.path.join(self.output_dir, name)):
                  self._scene_index = max(self._scene_index, int(match.group(1)))

      self.create_service(Trigger, 'regenerate_scene', self._on_regenerate_scene)
      self.create_service(Trigger, 'new_take', self._on_new_take)

      self.get_logger().info('scene_randomizer node ready.')

    def _hallway_bounds(self) -> dict:
      x_min, x_max, y_min, y_max = self.get_parameter('hallway_bounds').value
      return {'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}

    def _scene_dir(self, scene_index: int) -> str:
      """Per-scene subfolder -- everything about one scene (scenario.json,
      ground_truth.npy/.json, trajectory.json, scene.usd) lives together here
      rather than as flat scene_NN_*.* files in self.output_dir."""
      path = os.path.join(self.output_dir, f"scene_{scene_index:02d}")
      os.makedirs(path, exist_ok=True)
      return path

    def _save_ground_truth_grid(self, obstacles, scene_index: int) -> None:
      """Save the real-footprint obstacle occupancy grid into this scene's subfolder.

      Runs automatically on every full regeneration — obstacles must already be
      spawned (spawn_static_obstacles measures each one's real bounding box).
      """
      grid, map_info = build_ground_truth_occupancy_grid(self._hallway_bounds(), obstacles)
      base_path = os.path.join(self._scene_dir(scene_index), "ground_truth")
      np.save(f"{base_path}.npy", grid)
      with open(f"{base_path}.json", 'w') as f:
          json.dump({
              'resolution': map_info.resolution,
              'origin_x': map_info.origin.position.x,
              'origin_y': map_info.origin.position.y,
              'width': int(grid.shape[1]),
              'height': int(grid.shape[0]),
          }, f, indent=2)

    def _save_lidar_height_grid(self, obstacles, scene_index: int) -> None:
      """Save the lidar-height-sliced occupancy grid into this scene's
      subfolder, alongside (not instead of) the full-bbox ground_truth grid
      -- the two answer different questions (see
      build_lidar_height_occupancy_grid's docstring): this one correctly
      shows the real open space under e.g. a table, between its legs, which
      ground_truth.npy marks fully blocked. Requires obstacles to already
      have real PhysX collision (spawn_static_obstacles applies it) and
      Play to be active.
      """
      grid, map_info = build_lidar_height_occupancy_grid(self._hallway_bounds(), obstacles)
      base_path = os.path.join(self._scene_dir(scene_index), "lidar_height_ground_truth")
      np.save(f"{base_path}.npy", grid)
      with open(f"{base_path}.json", 'w') as f:
          json.dump({
              'resolution': map_info.resolution,
              'origin_x': map_info.origin.position.x,
              'origin_y': map_info.origin.position.y,
              'width': int(grid.shape[1]),
              'height': int(grid.shape[0]),
          }, f, indent=2)

    def _save_trajectories(self, pedestrians, scene_index: int) -> None:
      """Save each pedestrian's actual walked path over time into this scene's
      subfolder -- pedestrians must already be driven (spawn_pedestrians
      populates spec.trajectory as it drives them)."""
      output_path = os.path.join(self._scene_dir(scene_index), "trajectory.json")
      save_pedestrian_trajectories(pedestrians, scene_index, output_path)

    def _save_scene_usd(self, scene_dir: str) -> None:
      """Export the live stage's root layer as scene.usd in this scene's
      subfolder -- a self-contained, reopenable copy of hallway_scene.usd
      showing exactly this scene's obstacle + pedestrian layout (at their
      final resting position, since this runs after spawn_pedestrians
      finishes driving them -- the walk itself isn't baked into USD state,
      see trajectory.json for that).

      Deliberately uses Usd.Stage.GetRootLayer().Export(...), NOT
      omni.usd's save_as_stage() -- the latter would make the newly-saved
      copy the new "currently open" stage going forward, silently diverging
      every later regeneration away from the master hallway_scene.usd.
      Export() just writes a copy without touching what's open. Best-effort:
      doesn't fail scene generation if the export itself has trouble.
      """
      import omni.usd
      try:
          stage = omni.usd.get_context().get_stage()
          stage.GetRootLayer().Export(os.path.join(scene_dir, "scene.usd"))
      except Exception as exc:
          self.get_logger().warn(f"Failed to export scene.usd snapshot to {scene_dir}: {exc}")

    def _on_regenerate_scene(self, request, response):
      """Full reroll: new seed, new statics, new pedestrians."""
      try:
          if self._current_scenario is not None:
              clear_static_obstacles(self._current_scenario.static_obstacles)
              clear_pedestrians(self._current_scenario.pedestrians)

          seed = new_seed()
          self._rng.seed(seed)
          self._prefab_library = load_obstacle_prefab_library(assets_dir=self.asset_dir)
          self._start_poses = load_start_poses(self.start_pose_yaml)

          pose = sample_start_pose(self._rng, self._start_poses)
          apply_car_start_pose(pose)

          obstacles = sample_obstacle_positions(
              self.num_static_obstacles, self._prefab_library, self._rng, self._hallway_bounds())
          ob = spawn_static_obstacles(obstacles)

          pedestrians = sample_pedestrian_paths(
              self.num_ped, self.character_pool, self._rng, self._hallway_bounds())
          ped = spawn_pedestrians(pedestrians, self._hallway_bounds(), obstacles=ob)

          self._current_scenario = Scenario(
              seed=seed,
              static_obstacles=ob,
              pedestrians=ped,
              car_start_pose=pose,
          )
          self._scene_index += 1
          scene_dir = self._scene_dir(self._scene_index)
          save_scenario(self._current_scenario, os.path.join(scene_dir, "scenario.json"))
          self._save_ground_truth_grid(ob, self._scene_index)
          self._save_lidar_height_grid(ob, self._scene_index)
          self._save_trajectories(ped, self._scene_index)
          self._save_scene_usd(scene_dir)

          response.success = True
          response.message = f"Regenerated scene with seed {seed}"
      except Exception as exc:
          self.get_logger().error(f"regenerate_scene failed: {exc}")
          response.success = False
          response.message = str(exc)

      return response

    def _on_new_take(self, request, response):
        """Reroll pedestrians only, keeping the current static layout."""
        if self._current_scenario is None:
            return self._on_regenerate_scene(request, response)

        try:
            clear_pedestrians(self._current_scenario.pedestrians)
            pedestrians = sample_pedestrian_paths(
                self.num_ped, self.character_pool, self._rng, self._hallway_bounds())
            self._current_scenario.pedestrians = spawn_pedestrians(
                pedestrians, self._hallway_bounds(), obstacles=self._current_scenario.static_obstacles)

            self._scene_index += 1
            scene_dir = self._scene_dir(self._scene_index)
            save_scenario(self._current_scenario, os.path.join(scene_dir, "scenario.json"))
            # Static layout is unchanged from the previous take, but each
            # scene_NN/ subfolder should be self-contained -- re-write the
            # (identical) ground truth grid here too rather than only in the
            # earlier scene's folder.
            self._save_ground_truth_grid(self._current_scenario.static_obstacles, self._scene_index)
            self._save_lidar_height_grid(self._current_scenario.static_obstacles, self._scene_index)
            self._save_trajectories(self._current_scenario.pedestrians, self._scene_index)
            self._save_scene_usd(scene_dir)

            response.success = True
            response.message = "New take: respawned pedestrians."
        except Exception as exc:
            self.get_logger().error(f"new_take failed: {exc}")
            response.success = False
            response.message = str(exc)

        return response


def main(args=None):
    rclpy.init(args=args)
    node = SceneRandomizerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
