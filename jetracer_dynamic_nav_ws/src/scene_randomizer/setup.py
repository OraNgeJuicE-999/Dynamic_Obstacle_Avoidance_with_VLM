import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'scene_randomizer'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='itri',
    maintainer_email='itri@todo.todo',
    description='Procedural hallway scene randomization (statics + pedestrians) for Isaac Sim',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'scene_randomizer_node = scene_randomizer.scene_randomizer_node:main',
        ],
    },
)
