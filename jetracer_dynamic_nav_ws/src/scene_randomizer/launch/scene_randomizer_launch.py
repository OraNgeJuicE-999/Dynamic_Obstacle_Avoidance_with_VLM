import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    default_start_poses_yaml = os.path.join(
        get_package_share_directory('scene_randomizer'), 'config', 'start_poses.yaml')

    return LaunchDescription([
        Node(
            package='scene_randomizer',
            executable='scene_randomizer_node',
            name='scene_randomizer',
            output='screen',
            parameters=[
                # TODO: point these at real values once the hallway/assets are set.
                {'num_static_obstacles': 8},
                {'num_pedestrians': 3},
                # Repo-root hallway_assets/static_obstacles/, not shipped via colcon
                # install since these are large binary props, not package code —
                # point this at wherever you've checked out the repo.
                {'assets_dir': '/home/itri/jetracer_nav/hallway_assets/static_obstacles'},
                {'output_dir': '/home/itri/jetracer_nav/scenes'},
                {'start_poses_yaml': default_start_poses_yaml},
                {'hallway_bounds': [-7.93, 7.93, -1.16, 1.16]},
                # TODO: fill in with real omni.anim.people character ids once known.
                {'character_pool': ['']},
            ],
        ),
    ])
