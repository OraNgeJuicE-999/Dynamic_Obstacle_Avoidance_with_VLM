# Research Design: VLM vs. MLP vs. World Model Path Selection

This document captures the design decisions for the core research question this repo
feeds into (see [[README.md]] "Project context" for the short version). It's the result
of a design review that worked through the open forks in the original framing —
recorded here so the reasoning behind each choice isn't lost.

## Core question

Under tight compute constraints, is a VLM, an MLP, or a world model the better way to
pick a driving/navigation path? None of the compared approaches generates its own
candidate paths — that would muddy the comparison. All approaches are handed the exact
same candidate set, produced once by shared, non-learned infrastructure, inside a
hallway test scene with a moving pedestrian. Each approach's only job is to rank/pick
the best candidate.

The metric that matters is **performance per unit of compute**, not raw performance —
so every run logs both task performance (collision-freeness, path complexity, progress
toward goal) and per-decision cost (real-time latency, memory footprint, on the actual
edge hardware).

## The five-arm ablation

Scope was deliberately widened from a straight three-way (VLM/MLP/world model)
comparison into a 2x2-style ablation, because the two extra arms are nearly free — they
reuse components the other arms already require — and they isolate two different
questions that a straight three-way comparison would conflate:

| Arm | Sensing | Prediction | Isolates |
|---|---|---|---|
| **VLM** | camera (VQA) | learned, semantic (2-stage, distilled from a 7B teacher) | full semantic reasoning |
| **MLP+YOLO** | camera (YOLO) | learned (MLP head) | learned prediction, non-semantic |
| **YOLO-based** | camera (YOLO) | naive (constant-velocity) | "does vision help even without learned prediction" |
| **Pure Algorithm** | lidar only | naive (constant-velocity) | the true no-vision, no-learning floor |
| **World Model** | camera | deferred (out of scope for now — no time budget for it currently) | future work |

`YOLO-based` and `Pure Algorithm` cost ~nothing extra: `YOLO-based` reuses the exact
same fine-tuned YOLO weights as `MLP+YOLO` (only the downstream prediction step
differs — naive math vs. a learned head), and `Pure Algorithm` needs no training at all
(lidar clustering + constant-velocity extrapolation, fully deterministic).

**Pure Algorithm is expected to sometimes win, and that's fine.** If it ties VLM/MLP in
simple reactive scenarios but loses specifically in scenarios that require reading
pedestrian intent *before* it shows up as measurable motion, that's a *more* credible
result than VLM/MLP dominating everywhere — uniform dominance would suggest a leaky
comparison, not a real finding.

### Where VLM's value is actually expected to show up

Since goal-progress, path complexity, and static-obstacle collision-freeness are all
fully derivable from lidar + geometry alone (no vision needed), the real
architecture-under-test is narrower than "path selection" — it's specifically: *turning
"a person is there, here's where they're heading" into a collision-risk adjustment.*
Hypothesized advantages for vision/semantic reasoning over pure geometry, roughly
strongest to weakest:

1. **Pre-motion intent from posture** — body orientation/gaze/weight-shift predicts
   movement before it shows up as a measurable velocity vector (which is what lidar
   tracking is limited to).
2. **Disambiguating what an obstacle is** — a person vs. a similarly-sized static prop
   should get different clearance treatment; geometry alone can't tell them apart.
3. **Choke-point context** — what's visually beyond a narrow gap (open room vs. another
   tight corridor), not just its width. Not testable in the current single-hallway
   scene; flagged as a future scene-diversity extension (see Scope below).
4. **Generalization to novel scenarios** outside the training distribution.

"Sees farther than lidar" was considered and rejected as a hypothesis — in an indoor
hallway, lidar range already covers what the camera covers, so it isn't a real
differentiator here.

## Candidate path generation

- **Generated locally**, not globally: A* runs against the local per-scan map, biased
  toward a general direction rather than solving straight to the goal (greedy, not
  optimal) — the goal may be outside what's currently sensed.
- To search *toward* the goal direction beyond the sensed range, a fake map extension is
  used, but only for **walls** — taken from the pre-built hallway floorplan (privileged,
  since static architecture is a reasonable thing for a deployed robot to know a priori,
  e.g. a loaded floorplan). **Obstacles (static props, the pedestrian) are never
  privileged in this extension** — those stay unknown until actually sensed. This is why
  candidates can turn out to be dead ends or run into obstacles the generator didn't
  know about — that's expected and corrected on the next scan, not a bug.
- Candidate generation is **pedestrian-agnostic** even during training-dataset
  construction — ground truth is never baked into what candidates exist, only into how
  they're scored (see Training data below). Baking ground truth into candidate shaping
  would make every generated candidate trivially collision-free by construction, which
  would leave the collision-freeness training label with no negative examples to learn
  from, and would create a train/inference mismatch (inference-time generation has no
  ground truth to avoid the pedestrian with).
- Pedestrian prediction (from whichever arm is running) feeds into **scoring only**,
  never back into regenerating candidates. This keeps the candidate set identical across
  arms in a given scenario, which is what makes the comparison an apples-to-apples test
  of decision-making rather than a test of whose obstacle prediction was more accurate.

## Training data / oracle labeling

- One shared, deterministic **oracle scoring script** generates training labels for
  every arm — same weighted multi-category rubric (collision-freeness, path complexity,
  goal-progress, "is this the best of the candidate set"), same underlying ground truth,
  same label format. **MLP and VLM train against the exact same answer key** — this is
  what makes it valid to attribute a later performance gap to architecture rather than
  label quality or differing notions of "best."
  - If the weighted-sum rubric ever produces a "best" candidate that doesn't actually
    look best, the fix is to refine the rubric (better weights/categories) — not to give
    VLM a separate, un-labelable notion of quality that MLP doesn't share.
  - Category weights are not yet decided; that's a tuning detail to settle empirically,
    not a design fork.
- Ground truth (true pedestrian trajectory, run through A*) is used to build the label
  for each candidate — but never to shape which candidates exist (see above).
- **Negative example generation is fully scriptable**: sample random intermediate
  waypoints/subgoals at varying distance/angle to get a spread of "very different, very
  bad" candidates, plus small perturbations of the best-scoring candidate (nudging
  waypoints near the start, or near the end) to get harder near-miss negatives. No
  manual labeling required — the oracle script scores whatever the generator produces.
- Dataset generation only needs to run once per scenario (candidates + labels), then
  every arm trains against that same fixed dataset.

## VLM specifics

- Two-stage single-model pipeline, both stages done by the deployed VLM itself (not
  split across a separate perception model and a separate non-VLM scorer, which would
  make "VLM compute cost" impossible to attribute cleanly):
  1. VQA: is there a person on screen, and if so, predict their movement.
  2. Candidate selection: choose the best candidate path given that context.
- **Distillation**: fine-tune a larger teacher (7B Qwen) on the oracle labels first, so
  it's a strong, task-competent teacher. Then train the deployable student (3B) against
  **both** the oracle label and the fine-tuned 7B's output (combined loss), not
  distillation alone — pure distillation has no error-correction if the teacher itself
  has blind spots. Evaluation always scores against the shared oracle rubric / real
  outcomes, never against "agreement with the 7B."

## MLP specifics

- Pedestrian detection: YOLO (COCO-pretrained, fine-tuned on Isaac Sim renders of the
  simulated pedestrians — auto-generated labels are nearly free since the sim has ground
  truth bounding boxes for free).
- Heading prediction: a learned head trained on bounding-box motion between frames, own
  collected data + JRDB (real-world pedestrian dataset) for the behavior-prediction
  head specifically.
- Scoring: **one MLP head per category** (collision-freeness, complexity, goal-progress,
  etc.) rather than one monolithic scorer, combined per the shared weighting scheme.
- Known confound, logged rather than corrected for now: MLP's detector benefits from
  JRDB's real-world pretraining, which the VLM arm doesn't get an equivalent of. Treated
  as "each arm uses its best realistic off-the-shelf pretraining," same as a real
  deployment would — not worth controlling for unless a result later hinges on it.

## Recovery / look-around behavior

Not a separate blind fallback — it's the same decision loop with a repositioning step
inserted. JetRacer's Ackermann steering can't spin in place, and the camera is rigidly
chassis-mounted (no pan-tilt), so:

1. When every candidate in the current forward view is unsafe/unavailable, take the
   best still-uncommitted lidar-suggested direction as a **pseudo-goal** (not
   auto-trusted).
2. Execute a small scripted creep maneuver (forward/back, since no in-place turn is
   possible) to bring that direction into the camera's frame.
3. Run the normal VLM/MLP decision pipeline on the new frame. If approved, proceed
   normally; if not, the lidar direction is marked refused and the loop tries the next
   candidate.

Open edge case to remember when implementing: bound the retry count (or add a
stop-and-wait fallback) so this can't creep indefinitely if every direction keeps
getting vetoed.

## Compute cost metric

**Real-time wall-clock latency per decision** on target hardware is the headline number
(captures the practical cost of e.g. a VLM's autoregressive decoding in a way raw FLOPs
would hide), with **peak memory/VRAM** logged alongside it as a separate pass/fail
constraint. The metric that actually matters for "smoothness" is consistency (e.g.
p95/p99 latency against the control loop's required decision rate), not just mean
latency — worth being concrete about this once real numbers exist.

## Scope decisions

- **World model: deferred**, not part of the near-term deliverable — no time budget for
  it right now, revisit later.
- **Scene stays simple for now**: one hallway, obstacle scatter + one pedestrian's path.
  Extending the scene generator to vary what's behind doorways/junctions (needed to
  actually test the "choke-point context" hypothesis) is a known, deliberately deferred
  follow-up — not worth the scope creep before the core VLM/MLP loop works end-to-end.

## Open / not yet decided

- Category weights for the oracle scoring rubric (empirical, to tune later).
- Retry/timeout bound for the look-around recovery loop.
- Whether/how to later add a scene-diversity axis (doorways to visually distinct rooms)
  to test the choke-point hypothesis.
