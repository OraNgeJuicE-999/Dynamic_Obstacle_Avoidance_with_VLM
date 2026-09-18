# Dynamic Obstacle Avoidance with VLM

## What this is

A research testbed for a single question: **under tight compute constraints, is a VLM,
an MLP, or a world model the better way to pick a driving/navigation path around a
moving pedestrian?**

None of the compared approaches gets to generate its own candidate paths — that would
muddy what's actually being compared. Instead every approach is handed the exact same
set of candidate paths, produced once by shared, non-learned infrastructure (A* over a
locally-sensed map), inside a shared hallway scene with a moving pedestrian. Each
approach's only job is to rank/score that identical candidate set and pick the best one.
The metric that matters is **performance per unit of compute** — every run is meant to
log both task performance (collision-freeness, path complexity, progress toward goal)
and per-decision cost (latency, memory, on the actual edge hardware), not just which
approach "wins."

The comparison has grown into a five-arm ablation (VLM, MLP+YOLO, YOLO-based naive,
lidar-only naive, and a deferred world-model arm), plus a scored recovery behavior for
when the fixed-camera JetRacer runs out of safe candidates in its current view. The full
reasoning behind that design — why five arms instead of three, how oracle training
labels are generated without ever leaking ground truth into candidate generation itself,
the VLM distillation setup, etc. — is written up in [`plan.md`](plan.md). This README
only covers what's actually built and runnable today.

## Current status

This repo is the **scene/simulation infrastructure** the comparison runs inside of, not
the VLM/MLP/world-model arms themselves (those live elsewhere / aren't started yet).
What's implemented so far, and what isn't:

| Piece | Status |
|---|---|
| Hallway scene (geometry, obstacle prop library) | Done — `hallway_assets/` |
| Static obstacle scatter + placement validation | Done — `obstacle_randomizer.py` |
| Ground-truth obstacle occupancy grid (for A*) | Done — real measured bounding boxes, not approximated |
| Pedestrian route planning (A*, one turn/stop event) | Done — `pedestrian_planner.py` |
| Pedestrian spawning + walk/idle animation | In progress — see below |
| JetRacer start-pose placement | Partial — `car_pose.py`, prim path/Articulation API unverified |
| ROS2 service layer (`/regenerate_scene`, `/new_take`) | Done — `scene_randomizer_node.py` |
| VLM / MLP / world-model arms themselves | Not started (design only, see `plan.md`) |

**Pedestrian animation** was the hard part: Isaac Sim 6.0.0's NavMesh baking is
confirmed broken in this environment (bakes hang regardless of volume size), and manual
skeletal retargeting onto the stock `Isaac/People/Characters` rig from a mismatched
81-joint clip rig doesn't work either. The solution that does work is
`CreateBehaviorAgentCommand` + `IBehaviorAgent` (`omni.anim.behavior.core`) —
`test_pedestrian_spawn.py` at the repo root spawns a character wired to Isaac's
`HumanMotionLibrary`, reads the live obstacle layout off the stage, plans a route with
`pedestrian_planner.py`, and drives the character along it with real walk-cycle
animation via `agent.move_along()`/`agent.idle()`. It's still being debugged end-to-end
in the real hallway scene (most recently: the character has intermittently ended up
above the floor/roof partway through a run, root-caused as far as spawn-time placement
but not yet fully resolved through a complete idle→turn→idle cycle) — treat it as the
current integration point, not a finished feature. `pedestrian_randomizer.py`'s
`spawn_pedestrians`/`move_character_along_commands` still need the same fix ported in
once the test script is fully confirmed working.

## Repository layout

```
plan.md                      Full research design doc (the "why" behind everything below)
test_pedestrian_spawn.py     Active pedestrian spawn/animate test (see Current status)
hallway_assets/               Hallway scene + obstacle prop assets
  hallway.usd, hallway_scene.usd, hallway.glb, hallway.json   Scene geometry
  static_obstacles/           .glb/.usd prop library, scanned at runtime by scene_randomizer
  textures/                   Texture assets referenced by the USD/glb scene
jetracer_dynamic_nav_ws/     ROS2 (Jazzy) colcon workspace
  src/jetracer_navigation/     Hand-written A* global planner + DWA local controller —
                                the same non-learned A* this project reuses as the shared
                                candidate-path generator described above
  src/scene_randomizer/        Hallway scene randomization + ROS2 service layer (this project)
```

### `jetracer_dynamic_nav_ws/src/scene_randomizer/`

Each Isaac Sim-dependent module is split into a pure-Python planning half (unit
testable, no `omni.*` imports) and an Isaac Sim-dependent spawning/execution half:

| File | Purpose |
|---|---|
| `scenario_io.py` | `Scenario`/`StaticObstacleSpec`/`PedestrianPathSpec`/`CarStartPoseSpec` dataclasses + JSON save/load. One scenario file is everything needed to reproduce a scene instance — seed, obstacle placements (incl. real measured footprints), pedestrian paths, car start pose. This is what every VLM/MLP/world-model arm keys off so they all see identical scenarios. |
| `obstacle_randomizer.py` | Samples non-overlapping obstacle placements inside the hallway bounds (reject-and-resample against wall clearance, spacing, and minimum corridor width), spawns them on the USD stage, measures each one's real world-space bounding box, and rasterizes those into a ground-truth occupancy grid for A*. |
| `pedestrian_planner.py` | Pure-geometry pedestrian route planning: samples a start point/heading/speed and one optional turn-or-stop event, then A*-routes each leg around the known obstacle layout (reusing `jetracer_navigation`'s grid A*, not Isaac's NavMesh). |
| `pedestrian_randomizer.py` | Isaac Sim-dependent spawning/driving of pedestrian characters. Character instantiation is still being migrated to the Behavior Agent API (see Current status / `test_pedestrian_spawn.py`); `move_character_along_commands` is an older direct-mover fallback that predates that API being found. |
| `car_pose.py` | Loads a small fixed set of candidate JetRacer start poses from `config/start_poses.yaml` (not randomized — meant to cover a representative spread of approaches) and applies one to the stage. |
| `scene_randomizer_node.py` | ROS2 node exposing `/regenerate_scene` (full reroll: new seed, new statics, new pedestrians) and `/new_take` (keep the static layout, only reroll pedestrians) as `std_srvs/Trigger` services. Must run inside Isaac Sim's own Python process (its `omni.*` APIs aren't pip-installable into a plain ROS2 environment) — see the node module's docstring for how to launch it there. |

### `jetracer_dynamic_nav_ws/src/jetracer_navigation/`

The frozen A*+DWA navigation stack (originally built for a different office-scene
robot-navigation project; reused here purely as the non-learned candidate-path
generator). `a_star.py`/`primitives.py`/`utils.py` are what `pedestrian_planner.py` and
(eventually) the candidate-path generator import directly; `dwa.py`,
`dwa_controller.py`, `a_star_planner.py`, and the launch files are the original
standalone ROS2 nodes, not currently wired into this project's pipeline.

## Prerequisites

- Isaac Sim 6.0.0, with `omni.anim.behavior.core` and `omni.anim.retarget.core` enabled
- ROS2 Jazzy (for `scene_randomizer_node`'s service layer)
- `numpy`, `pyyaml`, `tf_transformations`

## Running the scene randomizer

```bash
cd jetracer_dynamic_nav_ws
colcon build --packages-select jetracer_navigation scene_randomizer
source install/setup.bash
```

`scene_randomizer_node` has to run inside Isaac Sim's own Python process — see that
module's docstring for the two ways to do that (standalone script via Isaac Sim's
bundled interpreter, or as a loaded Kit extension). Once running, with the hallway scene
open and Play active:

```bash
ros2 service call /regenerate_scene std_srvs/srv/Trigger
ros2 service call /new_take std_srvs/srv/Trigger
```

## Trying the pedestrian spawn/animate test

With `hallway_assets/hallway_scene.usd` open in Isaac Sim and Play active, paste
`test_pedestrian_spawn.py` into Window → Script Editor and run it (Ctrl+Enter). It reads
the live obstacle layout, plans an obstacle-aware route, and spawns a character that
walks it with real walk-cycle animation, stopping/turning partway through per the
planned route. See the file's own docstring and Current status above for where this
currently stands.
