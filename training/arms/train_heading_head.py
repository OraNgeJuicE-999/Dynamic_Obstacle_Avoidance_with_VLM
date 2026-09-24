"""Train the MLP arm's heading-prediction head: bounding-box motion between
frames -> predicted pedestrian movement direction (plan.md).

Trained on this project's own sim data (from train_yolo.py's detections)
plus JRDB (a real-world pedestrian dataset) for generalization beyond the
simulated hallway.

NOT YET IMPLEMENTED -- deprioritized behind candidate_path_generator.py /
oracle_scorer.py / the VLM arm (see training/README.md for current focus).
Blocked on train_yolo.py producing real bounding-box sequences to train
against, plus deciding how JRDB gets ingested/formatted alongside them.
"""

raise NotImplementedError("train_heading_head.py: blocked on train_yolo.py + JRDB ingestion, see module docstring")
