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
from scene_randomizer.scenario_io import Scenario, new_seed, save_scenario


class SceneRandomizerNode(Node):

    def __init__(self):
      super().__init__('scene_randomizer')

      self.declare_parameter('num_static_obstacles', 8)
      self.declare_parameter('num_pedestrians', 3)
      self.declare_parameter('assets_dir', '/home/itri/jetracer_nav/hallway_assets/static_obstacles')
      self.declare_parameter('output_dir', '/home/itri/jetracer_nav/scenes')
      self.declare_parameter('start_poses_yaml', '')
      self.declare_parameter(
          'hallway_bounds',
          [-7.93, 7.93, -1.16, 1.16],  # x_min, x_max, y_min, y_max
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

      # Continue numbering from existing scene_NN.json files rather than
      # overwriting scene_01 every time this node restarts.
      self._scene_index = 0
      if os.path.isdir(self.output_dir):
          for name in os.listdir(self.output_dir):
              match = re.fullmatch(r"scene_(\d+)\.json", name)
              if match:
                  self._scene_index = max(self._scene_index, int(match.group(1)))

      self.create_service(Trigger, 'regenerate_scene', self._on_regenerate_scene)
      self.create_service(Trigger, 'new_take', self._on_new_take)

      self.get_logger().info('scene_randomizer node ready.')

    def _hallway_bounds(self) -> dict:
      x_min, x_max, y_min, y_max = self.get_parameter('hallway_bounds').value
      return {'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}

    def _save_ground_truth_grid(self, obstacles, scene_index: int) -> None:
      """Save the real-footprint obstacle occupancy grid alongside scene_<idx>.json.

      Runs automatically on every full regeneration — obstacles must already be
      spawned (spawn_static_obstacles measures each one's real bounding box).
      """
      grid, map_info = build_ground_truth_occupancy_grid(self._hallway_bounds(), obstacles)
      base_path = os.path.join(self.output_dir, f"scene_{scene_index:02d}_ground_truth")
      np.save(f"{base_path}.npy", grid)
      with open(f"{base_path}.json", 'w') as f:
          json.dump({
              'resolution': map_info.resolution,
              'origin_x': map_info.origin.position.x,
              'origin_y': map_info.origin.position.y,
              'width': int(grid.shape[1]),
              'height': int(grid.shape[0]),
          }, f, indent=2)

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
          output_path = os.path.join(self.output_dir, f"scene_{self._scene_index:02d}.json")
          save_scenario(self._current_scenario, output_path)
          self._save_ground_truth_grid(ob, self._scene_index)

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
            output_path = os.path.join(self.output_dir, f"scene_{self._scene_index:02d}.json")
            save_scenario(self._current_scenario, output_path)

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
