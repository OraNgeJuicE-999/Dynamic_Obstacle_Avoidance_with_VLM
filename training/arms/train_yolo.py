"""Fine-tune COCO-pretrained YOLO on sim-rendered pedestrians, for the
MLP+YOLO and YOLO-based arms (plan.md).

Labels are nearly free here: Isaac Sim has ground-truth bounding boxes for
the spawned pedestrian character for the taking, so this is a matter of
capturing (frame, bbox) pairs during a take rather than any manual
annotation.

NOT YET IMPLEMENTED -- deprioritized behind candidate_path_generator.py /
oracle_scorer.py / the VLM arm (see training/README.md for current focus).
Blocked on the same two things as camera_capture.py and the VLM arm:
  1. camera_capture.py needs a confirmed JetRacer camera prim path
  2. ground-truth bounding box capture needs the pedestrian spawn/drive
     interface finalized (see trajectory_recorder.py's same caveat) --
     Isaac Sim's UsdSkel/BBoxCache tooling already used elsewhere in this
     project (see obstacle_randomizer.spawn_static_obstacles) can likely
     give the pedestrian's own world-space bounding box the same way, once
     there's a confirmed prim path to query it from.
"""

raise NotImplementedError("train_yolo.py: blocked on camera_capture.py + pedestrian bbox capture, see module docstring")
