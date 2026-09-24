"""Train the MLP arm's candidate scorer: one head per oracle-label category
(collision-freeness, complexity, goal-progress, etc.) rather than one
monolithic scorer, combined per oracle_scorer.SCORE_WEIGHTS (plan.md: "one
MLP head per category ... combined per the shared weighting scheme").

Unlike the VLM arm, this doesn't need a camera image at all for its
candidate-scoring features (distance to obstacles, path curvature, progress
toward goal are all derivable from dataset_schema.CandidatePath +
DecisionPoint directly) -- it only needs YOLO+heading-head output as an
*additional* input feature representing "predicted pedestrian position,"
same as the VLM arm's own semantic reasoning stands in for the same thing.

NOT YET IMPLEMENTED -- deprioritized behind candidate_path_generator.py /
oracle_scorer.py / the VLM arm (see training/README.md for current focus).
The candidate-geometry features this needs are already fully available from
today's dataset_schema.Dataset (no blocker there); what's blocked is the
pedestrian-heading-prediction input feature, which needs train_heading_head.py
first. A version using ground-truth pedestrian position instead of a
predicted one could plausibly be built and trained today as an "oracle
upper bound" ablation, if that's useful before train_heading_head.py is
ready -- worth discussing before starting this one.
"""

raise NotImplementedError("train_mlp_scorer.py: blocked on train_heading_head.py, see module docstring")
