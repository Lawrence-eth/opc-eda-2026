# FloorSet winning plan

**Status:** active strategic plan, updated 2026-07-11. v32 has cleared the
pre-beta promotion tournament; v31 remains the verified rollback floor. The objective is first
place on the hidden final set.  Every intermediate metric exists only to make
that outcome more likely.

## 1. Ground truth we must plan around

- Beta submission is **2026-07-24 (GMT+8)** and final submission is
  **2026-08-21 (GMT+8)**.  Beta is the first authoritative hidden-distribution
  and field-runtime signal; the final result alone determines awards.
- The official FAQ promises evaluation results for beta and confirms beta is
  advisory, but it does not yet specify per-case fields, ranking detail, or a
  resubmission limit. Plan for aggregate-only feedback; use richer detail only
  if the organizers actually provide it.
- v32 is the promoted pre-beta candidate: public RF=1 score **1.615379**,
  100/100 feasible, and paired runtime-adjusted improvement over v31 at field
  medians 0.25/0.5/1/2/3s. It reuses an already-computed strong-pin first pass,
  so it adds no dissection solve. Across all five clean folds it improves
  **1.778134 → 1.767565**; across all five raw folds it improves **1.848364 →
  1.834588**, with 1,050/1,050 feasible and every fold delta negative. Sealed
  fold-4 deltas are -0.010418 clean and -0.012594 raw.
- v31 remains reproducible and submittable as the rollback: RF=1 score **1.616638**, 100/100 public
  feasibility, exact AMD64 wrapper parity, and score **1.798552** on the
  tracked 105-case MIB-clean heavy holdout. On the frozen five-fold
  input-compatible heavy stratum it is 525/525 feasible with pooled score
  **1.778134**. On the separate label-blind raw-offset stratum it is also
  525/525 feasible with pooled score **1.848364**; supplied golden layouts
  themselves violate MIB in 519/525 raw cases, so that score is a paired
  robustness baseline rather than a clean optimization target.
- Public validation is overfit: its score is materially better than unseen
  training layouts.  Public improvement alone is never promotion evidence.
- Cases with 100–120 blocks contribute about 80% of total score; 116–120 alone
  contribute about 34%.  Heavy layouts are the primary optimization target,
  while every size remains a feasibility and runtime obligation.
- The current residual is led by HPWL, then soft violations and area.  Current
  n>=100 averages are hg/ag/V = **0.560/0.154/0.085**.
- Exact global optimization for a 120-block mixed topology/shape problem is not
  credible inside the contest runtime.  Exact methods remain valuable as
  offline teachers and selectively gated polishers.
- The intended data-driven path has real measured headroom.  On 105 unseen
  heavy layouts, golden coordinate ordering passed through a constraint-safe
  dissection pass improves **70/105** cases and score
  **1.798552 -> 1.745767**, with every candidate feasible.  A first input-only
  ridge model already reduces held-out vertical-order inversions from roughly
  27% to 7%; a learned hybrid candidate plus an **offline golden-baseline
  oracle** improves the same heavy holdout to **1.788169**, selecting 29 true
  wins and no official losses.  This proves candidate headroom, not a
  deployable selection policy.  The submitted solver cannot see the hidden
  golden HPWL/area denominators, and its current self-normalized selector can
  rank tradeoffs differently.
- Training MIB annotations are corrupted in many layouts (official issue #12).
  MIB-dependent losses use input-compatibility masks, and every result is
  reported separately on clean/input-compatible and unfiltered hash-offset
  regimes. Coordinate/tree supervision may use broader data only with a
  masked-MIB ablation and source-disjoint evaluation.
- Reverse-engineering the dataset generator is explicitly disqualifying.  We
  will use the supplied inputs/labels, the official template, and published
  floorplanning methods; we will not inspect or imitate generator internals.

## 2. Definition of a winning candidate

There is no honest fixed score that proves first place before the beta
leaderboard.  We therefore optimize a Pareto frontier and promote only evidence
that predicts hidden performance.

### Non-negotiable gates

1. **Hard feasibility:** 100% on public, every held-out fold, adversarial unit
   cases, source fuzz, and packaged-binary fuzz.  One high-weight failure can
   erase an entire research gain.
2. **Generalization:** paired improvement on the input-selected,
   source-file-disjoint heavy development folds, confirmation on calibration,
   and one final sealed audit. The pseudo-test 95% confidence interval of the
   weighted delta must favor the candidate; no single fold may regress by more
   than 0.005 without a diagnosed distribution reason.
3. **Runtime-adjusted value:** beat the incumbent at assumed field medians
   1s, 2s, and 3s using paired per-case timings.  Keep a separate 0.5s stress
   result.  After beta, replace assumptions with the measured field signal.
4. **No validation-ID logic:** gates may use input features, incumbent metrics,
   or learned predictions, never public case identity or a block-count singleton
   chosen solely because it wins one public layout.
5. **Reproducibility:** deterministic weights, manifests, source revision,
   result JSON, AMD64 archive checksum, exact wrapper parity, and a documented
   rollback point.
6. **Selection fidelity:** compare every deployable selector with the offline
   golden-baseline oracle on held-out candidate pools. Report weighted regret,
   false-accept cost, missed-win cost, and precision/recall. Never promote an
   oracle-only gain as though it were an inference-time result.
7. **Tail risk:** report score-weighted false-accept regret, worst regression,
   5% regression CVaR, and the probability of beating v31 on pseudo-tests that
   draw exactly one case per block count. A count of wins cannot hide one large,
   high-weight loss. The sealed audit has no post-hoc exception.

### Stretch targets

- **Beta:** heavy cross-validation mean <=1.75, public RF=1 <=1.55, 100%
  feasibility, and a package that stays on the favorable runtime frontier.
- **Final:** beat the best beta score with margin on both hidden quality and
  runtime.  Internally pursue heavy mean <=1.60 and V<=0.05; these are research
  targets, not claims that they guarantee first place.

## 3. Target solver architecture

```text
inputs + hard targets
        |
        v
constraint hypergraph featurizer
(blocks, nets, pins, MIB/cluster hypernodes, boundary, fixed/preplaced anchors)
        |
        +-----------------------------+
        |                             |
        v                             v
learned topology model          candidate-value/runtime model
(x/y rank, row/band, aspect,    (which decoders/polishers are worth paying for)
 precedence/tree hints)
        |
        v
constraint-by-construction decoders
  A. learned dissection (fast, guaranteed geometry)
  B. learned B*-tree/sequence-pair shape-curve decoder (higher ceiling)
        |
        v
input-calibrated Pareto/abstaining selector
        |
        v
selective fixed-topology LP/median polish + safe soft repair
        |
        v
v31 fallback and final hard-feasibility gate
```

The model proposes topology; it never owns hard legality. The decoder preserves
areas, preplaced/fixed dimensions and locations, and non-overlap by construction.
It jointly plans the soft cluster, MIB, and boundary requirements, but those are
measured objectives rather than guarantees in the current engine. This avoids
letting a generic learned-coordinate repair destroy the topology signal while
remaining honest about residual soft violations.

## 4. Research tracks and how they challenge each other

### Track A — evaluation science (first dependency)

- Hash-partition by source `.th` file, not individual layout, to prevent
  near-configuration leakage.
- Maintain five heavy folds (n=100..120) and three all-size folds. Stratify by n
  exactly and measure/rebalance preplaced/fixed count, boundary count,
  connectivity density, and MIB/cluster profile rather than merely recording
  those covariates.
- Freeze roles before tuning: folds 0–2 are development, fold 3 is selector
  calibration, and fold 4 is sealed until the beta freeze. Repeatedly consulting
  all five would turn them into another public set. Add an input-covariate OOD
  panel and newly sealed confirmations for repeated hypothesis campaigns.
- Store immutable sample manifests.  Public validation is report-only.
- Score raw quality, exact soft attribution, paired runtime-adjusted totals,
  per-stratum deltas, selector regret/tail risk, and selector precision/recall.
  Bootstrap pseudo-tests with exactly one draw per block count, matching hidden
  weighting, in addition to source-cluster uncertainty.
- Keep both the input-compatible quality stratum and an unfiltered,
  label-blind hash-offset robustness stratum. Mask impossible/corrupt MIB
  contributions for attribution; never use a corrupt MIB target to tune a
  shape policy or call the offset-0-heavy clean stratum representative.
- Require multi-seed stability and fresh confirmation during development, not
  only during final ablation week.

**Kill condition:** no model or decoder work is promoted until these folds can
replay v31 deterministically.

**Current state:** satisfied for the incumbent under fail-closed schema 3.
The clean stratum has 525 cases / 522 sources and v31 fold scores 1.797574 /
1.736764 / 1.749308 / 1.810148 / 1.796878. The raw stratum has 525 distinct
sources, all 112 offsets represented, and scores 1.848571 / 1.826033 /
1.823104 / 1.861499 / 1.882611. Every case binds base input, derived hard
targets, scoring labels, pinned dataset/evaluator, and harness identity.

### Track B — teachers and upper bounds

- Use supplied golden coordinates/B*-trees as legitimate supervised labels.
- Use fixed-topology sparse LP and weighted-median polishing offline to label
  movable-component optima and candidate value.  The broad LP already improves
  public 1.616638 -> 1.609674 but is too slow to deploy unchanged.
- Generate counterfactual decoder candidates across width, aspect, row, and
  topology choices; retain exact per-case winners and failure reasons.
- Use those pools to label both topology models and a hidden-baseline/value
  estimator. Candidate quality and the ability to select it are separate
  learning problems.
- Solve small/medium subproblems exactly or with long-budget search to create
  better topology/shape teachers.  Never incorporate generator-specific code.

**Purpose:** teach a fast policy and quantify decoder ceiling.  A teacher is not
shipped merely because it improves raw score.

Before choosing model complexity, run an **oracle ladder** through every decoder:
golden x rank, y/row assignment, dimensions/aspects, obstacle/frame relations,
MIB/cluster shape plans, and golden versus predicted B*-tree topology in
controlled combinations. A structured decoder must first approach golden output
when given golden structure; otherwise improve the decoder before the predictor.

**First ladder result (fold 0):** the ordinary six-config two-pass control
envelope reaches 1.757074 from v31's 1.797574. Relative to that control, golden
direct-x rank adds -0.013490, direct-y -0.010887, combined ranks -0.016685, and
exact golden prev -0.019333. Making every block rigid at its golden dimensions
wins 0/105 and scores above 3 without fallback. The beta model should therefore
target within-row x first, while the final track fixes row/topology/shape
decoding; forcing labels into an incompatible decoder is not progress.
The later learned-v3 ladder found consistent fixed-configuration x-order
headroom on folds 0–3, but its proxy normalized against the saved final v31
layout rather than the primary production pool. It is therefore a diagnostic,
not promotion evidence. Deployment must be tested end-to-end as an explicit
final `[v31_final, candidate]` gate with the same first-candidate reference.

### Track C — beta-capable learned ordering

- Start from a streamed, input-only feature set: normalized area, net/pin
  degrees, pin centroids, message-passed graph coordinates, hard-target data,
  and constraint bits/hypernodes.
- Production schema additions must include actual MIB/cluster group identities,
  group sizes and messages, preplaced dimensions/obstacle geometry, weighted pin
  medians/quantiles/spread, fixed-neighbor geometry, and multi-hop landmarks.
  Membership booleans alone do not implement the planned hypergraph.
- Train x/y rank and row/band heads.  Use reflection/rotation augmentation with
  correctly transformed pins, dimensions, and boundary codes.
- Enforce block-permutation equivariance and reflection/rotation/scale/
  translation metamorphic tests. Prefer pairwise/relational tree losses because
  valid B*-tree encodings are not unique; train/listwise-fine-tune against
  decoded contest outcomes rather than treating coordinate MAE as the goal.
- Distill the best model to small fixed coefficient/MLP weights implemented in
  stdlib or native code; extract features once per case.
- Replace a demonstrably redundant portfolio candidate instead of adding work.
  Preserve the current primary candidates and first-candidate fallback as
  safety.
- Replacement is conditional: when the scheduler/chooser abstains, execute the
  exact incumbent pass (or make the learned pass reproduce its default order).
  A plausible low-confidence prediction must not remove v31's unique win.
- Feed learned x as the within-row prior and learned y as unit/row precedence;
  aggregate scores over cluster/MIB hypernodes.

**Promotion gate:** improve at least 0.01 on the multi-fold heavy mean with the
**deployable** selector, accept candidates at >=98% precision, keep weighted
selection regret below 10% of oracle headroom, and add no meaningful median-1
runtime. The current ridge result meets the topology-signal gate only; it has
not met the selection or runtime-neutral integration gates.

### Track D — structured high-ceiling model

- Train a compact message-passing/attention model with multi-task heads:
  normalized x/y rank, log aspect, frame/row assignment, pairwise precedence,
  and B*-tree parent/side candidates.
- Losses: coordinate/rank, pairwise order, tree edge/side, aspect, boundary,
  MIB equality, cluster connectivity, plus the official differentiable cost
  proxy.  Mask corrupted labels per task.
- Compare deterministic GNN, autoregressive tree prediction, and a small graph
  diffusion/flow model.  The contest statement reports strong internal
  diffusion results, so diffusion is a real high-ceiling branch, but it must
  justify training/inference cost against the simpler structured model.
- Keep coordinate diffusion/flow with legality-aware projection or a
  constraint-preserving denoiser as a distinct branch. It is not equivalent to
  the rejected naïve independent-rectangle predictor; kill it only after an
  oracle projection-retention experiment.
- Export inference to a tiny pure-Python/native runtime or compile the model;
  do not casually bundle/import full torch per test case.

**Kill gates:** a model must beat the distilled rank baseline on three folds;
raw coordinate accuracy without decoded contest-score improvement is failure.
If local compute is the limiter rather than learning quality, secure a GPU
training run without changing the evaluation contract.

### Track E — constraint-safe decoders

1. **Dissection v3 (beta path):** learned row/band assignment, predicted x/y
   priors, shared cluster/MIB hypernodes, exact areas, obstacle-aware cuts, and
   global shared MIB aspect choices.
2. **Structured topology decoder (final path):** learned B*-tree or sequence
   pair, shape curves for soft/MIB units, explicit boundary frame, fixed and
   preplaced obstacle relations, followed by compaction.
3. Keep both on the Pareto frontier.  The dissection decoder is the guaranteed
   fallback; the topology decoder must prove 100% feasibility before quality
   matters.

**Challenge:** golden-order dissection still scores about 1.746 on the current
heavy holdout, so learning alone cannot reach the theoretical floor.  The
decoder must also improve shape, obstacle use, MIB, and cluster structure.

### Track F — adaptive search and runtime engineering

- First audit the current self-normalized selector on all held-out candidate
  pools. Because hidden golden HPWL/area baselines set the true HPWL-vs-area
  exchange rate and clamp, self-normalization does not in general preserve the
  official ranking.
- The v31 primary-pool audit now bounds the existing problem: on 420
  development/calibration cases it matches the pinned-official offline oracle
  412 times, makes zero false accepts, and has only 0.000202 weighted regret.
  Do not spend the
  campaign rebuilding what already works; re-audit because a learned candidate
  changes the candidate distribution and can expose different tradeoffs.
- Train input-derived golden-baseline/value models to predict official cost
  delta, uncertainty, and runtime before executing or accepting a candidate.
  Compare direct value prediction with predicted HPWL/area baselines.
- Always accept Pareto-dominant candidates; use calibrated prediction for
  HPWL/area/soft tradeoffs; abstain to the incumbent when uncertainty is high.
  Calibrate thresholds exclusively on source-file-disjoint training folds.
- Run LP/topology polish only where predicted runtime-adjusted value is
  positive.  Rejected candidate time still counts, so selector safety alone is
  insufficient.
- Prune redundant two-pass width candidates; fund learned/teacher-guided passes
  by replacement.  Cache all graph features and adjacency once.
- Profile a native C++/Rust hot path and per-case multiprocessing on the stated
  48-core host.  Use concurrency only after target-host startup and wall-time
  measurements prove it helps.
- Maintain fast, balanced, and quality variants until beta reveals the actual
  runtime median.
- Evaluate per-design median curves m(n)—constant, linear, quadratic, and
  accelerator-startup-shaped—with stress medians 0.1/0.2/0.3/0.5s as well as
  1/2/3s. Secure native Ice-Lake-like x86 timing; QEMU is a correctness check,
  not runtime evidence.
- Make a native C++/Rust feature/dissection/scoring path a first-class final
  track. Speed purchases additional topology starts, not just a smaller runtime
  factor, so begin parity work before beta if it does not block the beta model.

## 5. Why tempting alternatives are not the plan

| Temptation | Why it loses | Proper role |
|---|---|---|
| Tune more public feature pockets | Proven public overfit; hidden risk | Only if a multi-fold input-computable signal transfers |
| Broad exact/LP optimization | Raw wins, median-1 runtime loss | Offline teacher or highly selective polish |
| Predict independent raw rectangles then generic-legalize | Overlap/constraint repair destroys learned topology | Predict structure, or use legality-aware diffusion/projection and measure signal retention |
| Bundle a large torch model | Per-case process/import cost can erase quality | Distill/compile compact inference |
| Keep adding best-of candidates | Rejected work is still timed | Replace/prune, then use a value model |
| Force every MIB/cluster/boundary locally | Prior probes trade away much more HPWL/area | Joint shape/topology planning with calibrated selection |
| Reproduce generator behavior | Explicitly disqualifying | Supervised learning from supplied labels only |
| Exploit evaluator bugs | Fragile, non-algorithmic, organizer-risky | Report privately; submit spec-compliant output |
| Assume v31 already wins | No leaderboard evidence | Preserve it only as the floor |

## 6. Calendar and decision points

### Beta campaign

- **Jul 10–11:** strategic reset; fold manifests; candidate attribution;
  feature cache; teacher/oracle corpus.
- **Jul 12–15:** rank/GNN training; learned dissection integration; portfolio
  replacement; three-fold tournament.
- **Jul 16–18:** decoder v3 shape/MIB/cluster work; value gate; paired runtime
  profiling.
- **Jul 19:** beta algorithm freeze.  Choose from the Pareto frontier using
  hidden-risk, not public score.
- **Jul 15–18:** build a preliminary AMD64 learned package and add learned
  modules/weights to source-identity and clean-clone gates.
- **Jul 20–22:** sealed fold, 5,000+ source fuzz, target-native binary fuzz,
  AMD64 rebuild/parity, clean-clone reproduction, upload rehearsal.
- **Jul 23:** checksum freeze and beta upload buffer.
- **Jul 24:** beta deadline (GMT+8).

### Final campaign

- **Immediately after beta:** treat hidden quality and runtime as Bayesian
  evidence; diagnose by size/constraint strata if detailed results exist.
- **Jul 25–Aug 10:** structured GNN/diffusion and topology decoder; stronger
  teachers; target-host parallel/native optimization.
- **Aug 11–15:** final ablations, multi-seed/file-disjoint tournaments, adversarial
  feasibility, selector calibration, final Pareto decision.
- **Aug 16:** final algorithm freeze.
- **Aug 17–20:** rebuild, exact wrapper parity, large binary fuzz, clean-host
  rehearsal, re-download verification.
- **Aug 21:** final deadline (GMT+8).

## 7. Immediate execution queue

1. Finish heavy-fold candidate attribution and audit deployable-selector regret
   against the offline oracle; verify which current portfolio pass can be
   replaced without losing hidden wins.
2. Add file-disjoint fold-manifest and paired, source-clustered tournament
   tooling to the repository.
3. Turn the measured ridge/MLP and hidden-baseline/value experiments into a
   reproducible streamed training pipeline with input-only inference.
4. Integrate one learned hybrid candidate by replacement, cache features once,
   and run public + five-fold quality and paired-runtime gates.
5. In parallel, build the structured topology/aspect teacher dataset and the
   oracle decoder ladder before the first compact message-passing model.
6. Replay robust historical incumbents and leave-one-candidate/feature-pocket
   ablations on development/calibration folds; measure unique weighted value,
   selector regret, and tail loss before assuming v31 is the best hidden floor.
7. Verify beta submission-count limits and exactly which quality/runtime detail
   the leaderboard reveals; do not plan a Bayesian update around unavailable
   observations.
8. Promote and push only a fully gated checkpoint; otherwise retain v31 and
   iterate.

## 8. Repository and operational discipline

- `main` always remains buildable and submittable.  Research uses unique
  worktrees/branches and `/tmp` artifacts until promotion.
- Every promoted result includes code, model/weight provenance, fold manifests,
  raw results, timing controls, diagnostics, package hash, and wrapper parity.
- Every failed hypothesis is logged with enough evidence to prevent repetition.
- A historical “dead end” binds only the tested representation, objective, and
  decoder. A materially different learned/min-cut/constraint-preserving method
  may reopen the underlying idea with an explicit reason and new gate.
- Monitor the official problem page, Q&A, FloorSet commits/issues, and upload
  instructions daily through final submission.
- Organizer communication that can affect other teams or rules is drafted in
  the repo and sent only with operator approval.

This plan is deliberately falsifiable.  “Perfect” means it keeps challenging
its own assumptions with hidden-like evidence and reallocates effort when a
track fails; it does not mean pretending that first place can be certified
before the final leaderboard.
