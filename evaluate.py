"""Run the actual VLM vs. MLP vs. world-model comparison (plan.md's "Running
the comparison"): feed every trained arm the same scenarios/candidate sets,
log task performance (collision-freeness, path complexity, progress toward
goal) AND per-decision cost (real-time latency, memory/VRAM footprint) on
the target hardware -- the metric that matters is performance per unit of
compute, not raw performance, so both need to be logged for every arm on
every decision.

Deliberately NOT part of training/ -- this doesn't train anything, it's the
downstream comparison harness that consumes whatever arms/train_*.py
scripts have already produced.

NOT YET IMPLEMENTED -- there's nothing to evaluate until at least one arm
(train_mlp_scorer.py or train_vlm_student.py) has a trained checkpoint.
Shape it's expected to take once that exists: load each arm's checkpoint,
run it over a held-out set of training.common.dataset_schema.DecisionPoint
+ CandidatePath sets (excluding is_negative_example candidates -- see that
field's docstring, a real deployment never sees those), compare each arm's
chosen candidate_id against training.data_pipeline.oracle_scorer's ranking
on the SAME held-out set, and separately log wall-clock latency + peak
memory per decision.
"""

raise NotImplementedError("evaluate.py: needs at least one trained arm checkpoint first, see module docstring")
