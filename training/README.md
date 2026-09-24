# training/

Offline dataset generation + model training for the VLM/MLP/world-model
path-selection comparison (see [`../plan.md`](../plan.md) for the full
research design, [`../README.md`](../README.md) for the project overview).
Separate from the ROS2 `jetracer_dynamic_nav_ws` on purpose -- nothing here
needs colcon, it's plain Python + PyTorch, run as `python -m
training.<subpackage>.<module>` from the repo root.

## Status

| File | Status |
|---|---|
| `common/trajectory_io.py` | Done |
| `common/dataset_schema.py` | Done |
| `data_pipeline/trajectory_recorder.py` | `trajectory_from_commands` done (analytic stand-in); `record_live_trajectory` stubbed, blocked on the pedestrian-animation work in the main repo settling |
| `data_pipeline/decision_sampler.py` | **Template** — signatures + docstrings only, being written by hand |
| `data_pipeline/candidate_path_generator.py` | **Template** — signatures + docstrings only, being written by hand (docstrings carry the two gotchas already found by smoke-testing a first pass against real `jetracer_navigation.a_star` — worth reading before reimplementing) |
| `data_pipeline/oracle_scorer.py` | **Template** — signatures + docstrings only, being written by hand |
| `data_pipeline/camera_capture.py` | Stubbed, blocked on a confirmed JetRacer camera prim path in the hallway scene |
| `data_pipeline/dataset_builder.py` | Done (orchestrates the above — will error at runtime until the three templates above are filled in, imports still resolve fine) |
| `arms/train_vlm_teacher.py` | **Template** — signatures + docstrings only, being written by hand |
| `arms/train_vlm_student.py` | **Template** — signatures + docstrings only, being written by hand |
| `arms/train_yolo.py`, `train_heading_head.py`, `train_mlp_scorer.py` | Stubbed, deprioritized behind the VLM arm |
| `arms/train.py` | Done (thin `--arm` dispatcher) |
| `../evaluate.py` | Stubbed, needs a trained checkpoint to evaluate first |

## Try it today, without Isaac Sim

Once `candidate_path_generator.py` + `oracle_scorer.py` + `decision_sampler.py`
are filled in, they're fully testable against `pedestrian_planner.py`'s
*planned* route (via `trajectory_recorder.trajectory_from_commands`), with
no Isaac Sim dependency:

```python
from scene_randomizer.pedestrian_planner import plan_pedestrian_commands, sample_pedestrian_path
from training.data_pipeline.trajectory_recorder import trajectory_from_commands
from training.common.trajectory_io import SceneTrajectories
from training.data_pipeline.dataset_builder import build_dataset_for_scene

spec = sample_pedestrian_path(rng, hallway_bounds, character_id="pedestrian_0")
commands = plan_pedestrian_commands(spec, obstacles, hallway_bounds)
trajectories = SceneTrajectories(scene_index=0, actors=[trajectory_from_commands("pedestrian_0", commands)])

examples = build_dataset_for_scene(0, hallway_bounds, obstacle_footprints, trajectories, goal_xy=(7.0, 0.0))
```

Swap `trajectory_from_commands` for the real recorded trajectory once
`record_live_trajectory` is implemented -- nothing downstream needs to
change, since both produce the same `ActorTrajectory` shape.
