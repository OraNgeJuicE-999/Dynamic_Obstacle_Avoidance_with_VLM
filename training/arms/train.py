"""Thin CLI dispatcher across the per-arm training scripts.

Each arm's actual training logic differs enough (different model types,
different loss shapes -- see plan.md's per-arm specifics) that it isn't one
shared training loop; this just saves remembering five different module
paths. Equivalent to calling the target script directly with the same flags,
e.g. `--arm vlm_teacher --dataset x.json` == `python -m
training.arms.train_vlm_teacher --dataset x.json`.

Usage:
    python -m training.arms.train --arm vlm_teacher --dataset training_data/dataset.json
"""

import argparse
import sys

ARMS = {
    "yolo": "training.arms.train_yolo",
    "heading": "training.arms.train_heading_head",
    "mlp": "training.arms.train_mlp_scorer",
    "vlm_teacher": "training.arms.train_vlm_teacher",
    "vlm_student": "training.arms.train_vlm_student",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    args, remaining = parser.parse_known_args()

    import importlib

    module = importlib.import_module(ARMS[args.arm])
    sys.argv = [ARMS[args.arm]] + remaining
    module.main()


if __name__ == "__main__":
    main()
