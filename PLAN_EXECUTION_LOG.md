# FloorSet ICCAD-2026 — Comprehensive Plan Execution Log

### 2026-07-11 v32 reused-first-pass final gate — ✅ PROMOTED
- Mechanism: the incumbent n≥100 strong-pin candidate already computes two
  dissection passes but returned only pass 2. `dissect_solve()` can now return
  its first pass alongside the normal result. v32 retains that already-paid
  layout and compares it only after primary selection and boundary repair, so
  the selector's reference is the true deployable incumbent. No dissection
  solve is added.
- Public official RF=1 score improves **1.6166380548 → 1.6153787745**, with
  100/100 feasible. In a clean A-B-B-A timing block, paired mean runtime is
  0.19621s versus 0.19644s for v31. Runtime-adjusted v32 beats v31 at every
  tested field median from 0.25s through 3s.
- Source-disjoint tournament: all five clean and all five raw folds improve.
  Clean pooled score is **1.7781344112 → 1.7675645695** (100 wins, 7 losses,
  clustered-bootstrap CI [-0.01431,-0.00733]); raw is **1.8483636984 →
  1.8345881389** (144 wins, 4 losses, CI [-0.01739,-0.01055]). All 1,050
  evaluations are hard-feasible.
- The sealed fold 4 was opened only after source and runtime freeze. Clean
  improves **1.7968780102 → 1.7864600720** and raw improves **1.8826113230 →
  1.8700174712**, both with 100% bootstrap improvement probability and
  105/105 feasibility.
- Rejected challengers: broad additive two-pass and one-pass candidates failed
  the pessimistic 1s runtime gate; unconditional wf replacement and p1 swaps
  regressed clean folds; a global Shapely-overlap grouping-fidelity adjustment
  was statistically neutral (improvement probability 55.9%) and was not
  promoted.

### 2026-07-10 strategic reset + file-disjoint validation — 🏆 EXECUTING
- Reset the campaign around the final hidden ranking rather than incremental
  public tuning. Added `docs/WINNING_PLAN.md`: a beta-to-final architecture,
  quantitative promotion/kill gates, teacher/model/decoder/runtime tracks, and
  the July 24 / August 21 execution calendar. v31 remains the pushed safety
  floor, not the assumed final design.
- Re-audited the official state: FloorSet remains at `aadddcc`; no newer Problem
  C document is posted; beta remains 2026-07-24 GMT+8 and final remains
  2026-08-21 GMT+8. Open issue #12 still has no organizer response, so corrupted
  training MIB labels remain an explicit mask requirement.
- Added `scripts/build_holdout_folds.py` and regression tests. The primary
  compatibility manifest, `results/folds/heavy_clean_v1.json`, contains **five
  source-file-disjoint folds**, each with exactly five input-selected,
  MIB-compatible cases for every n=100..120: **525 cases / 522 source files**.
  Construction examined 637,392 layouts and rejected 46,980 cases whose MIB
  area intervals or hard targets could not share one shape. Free-block golden
  shapes do not affect admission; golden MIB violations are reported outcomes.
  Schema-3 manifest SHA-256:
  `48ecda41bb642caa67d2e617ff9e467816a0392d6a68a0a91c38cf2e5f847895`.
  Audit correction: 522/525 selected cases are offset 0, so this is a clean
  compatibility stratum, not a representative raw-data panel.
- Added `heavy_raw_hash_v1.json`: exactly one label-blind SHA-256-selected
  offset per source, five cases per n/fold, **525 cases / 525 sources**, all
  112 offsets represented (6 offset-0). v31 is 525/525 hard-feasible with
  pooled score **1.848364** versus **1.778134** on the clean stratum. The
  supplied golden layouts violate MIB in **519/525** raw cases, so raw absolute
  cost is not a clean target; use paired deltas and excess-soft attribution.
  Raw manifest SHA-256:
  `9b4ff6a36e1945718411a83045f598228c2b301fdfa22340e33c297da9ac41ec`.
- Learned-order evidence motivating the reset: a first input-only ridge model
  cuts held-out y-order inversions from roughly 27% to 7%. On the original
  105-case clean heavy set, a learned hybrid plus an **offline golden-baseline
  oracle** scores **1.788169 vs 1.798552**, selecting 29 official wins and zero
  losses. This is candidate headroom, not yet a deployable selector result. Golden
  coordinate guidance through the same safe interface is the stronger teacher
  upper bound: **1.745767**, 70/105 wins, all feasible.
- Strategic challenge found during review: the shipped self-normalized selector
  is feasibility-exact but not official-quality-exact without hidden golden
  HPWL/area denominators. Different denominators and clamping can reverse an
  HPWL/area tradeoff. Selector regret, false accepts, missed wins, calibrated
  baseline/value prediction, Pareto dominance, and abstention are therefore a
  first-class gate before any oracle-only learned gain can be promoted.
- Candidate-attribution challenge: public heavy cases never select the five
  base-width candidates, but unseen heavy cases select them 27/105 times. A
  seemingly redundant public candidate is therefore not removable. The least
  used heavy candidate (wf=1.1) wins only 1/105 and is the first defensible
  learned-candidate replacement target; it still requires all five new folds
  and paired runtime gates before integration.
- v31 clean-stratum replay is **525/525 feasible** with pooled RF=1 score
  **1.778134**. Fold scores are 1.797574 / 1.736764 / 1.749308 / 1.810148 /
  1.796878. Fold 1 intentionally retains two cases where the supplied golden
  layout has an MIB violation; the all-case clean-stratum score is primary and
  the clean subset is reported separately. Roles are frozen: folds 0–2
  development, fold 3 calibration, fold 4 sealed (v31 baseline only) until beta
  freeze.
- Added paired tournament tooling with source-cluster bootstrap, one-case-per-n
  pseudo-tests, exact fold deltas, feasibility, and tail-risk attribution.
  Added manifest/source/input/evaluator/solver provenance and pristine-tensor
  scoring so optimizer mutation cannot invalidate the gate.
- Audited the shipped primary portfolio selector against the pinned official
  scorer on pristine snapshots (420 cases / 3,665 candidate options). It
  matches the oracle in **412/420**, has **0 false accepts**, misses eight local
  wins, and loses only **0.000202 weighted** versus the primary-pool oracle;
  worst local regret is 0.03336. Thus selector calibration is mandatory for new candidate
  distributions but is not the present v31 bottleneck.
- Added the shared learned-order v3 feature path and deterministic streamed
  ridge trainer. The 60-scalar input-only schema includes permutation-invariant
  MIB/cluster hyperedge summaries, weighted pin medians/spreads, fused multi-hop
  messages, and preplaced-obstacle geometry. It passes permutation, group-ID,
  scale, schema, provenance, and malformed-input tests; dense n=112 extraction
  is ~39 ms on the local Neoverse N1. The hardened v4 artifact excludes the
  union of both heavy panels (**531 unique source files**) and uses 332,592
  training blocks plus 68,184 validation blocks with every n=100..120 present;
  validation pairwise inversion is 10.11% x / 8.48% y. Prediction ties count
  as half-errors, and inference validates the full schema and canonical payload
  digest. Model SHA-256:
  `4058105b4fee368c8a7293ba74ff462e652dadee675f08b8dcc91a4a6786e19a`.
  This remains an unmasked baseline because most nonzero-offset training MIB
  memberships are incompatible; masked/compatible/hybrid ablations are next.
- Historical-incumbent replay challenged v31 on folds 0–3. v25 is microscopically
  better on folds 0/2/3 but loses fold 1: pooled **1.773958 vs v31 1.773449**
  (delta +0.000510, source-bootstrap CI [-0.000419,+0.001629], pseudo-test win
  probability 29%). v29 changes only three cases and improves -0.000017 with a
  CI spanning both signs; v30 and v31 are identical on heavy cases. Since v31
  also owns the independently proven n<=90 polish, it remains the robust floor
  by evidence rather than chronology.
- The completed fold-0 decoder oracle ladder shows the strongest immediate
  ceiling is not naïve y sorting. A six-config ordinary two-pass control plus
  offline selector reaches 1.757074 vs v31 1.797574. Golden direct-x adds
  -0.013490 marginally; golden direct-y adds -0.010887; combined ranks add
  -0.016685; exact golden prev adds -0.019333. Exact golden rigid shapes win
  0/105 and score >3 without fallback, proving the current row/frame decoder
  cannot exploit independent aspect labels. Prioritize learned within-row x and
  topology/row decoding; do not equate low coordinate error with usable shape.
- A fixed learned-x configuration (`wf=1.2`, `pin_scale=6`) produced diagnostic
  final-incumbent proxy deltas of -0.02083 / -0.01155 / -0.01194 / -0.01237 on
  folds 0–3, with every generated candidate feasible. Its marginal exact
  headroom over the ordinary same-config control was much smaller and variable.
  Critical integration correction: that proxy used the saved final v31 layout
  as its normalization reference, whereas the production primary selector uses
  its first shelf candidate. These are not equivalent. No learned candidate is
  promoted until it runs end-to-end behind an explicit final-v31 gate.

### 2026-07-10 integrated v31 + hardened AMD64 release — ✅ PROMOTED
- Solver: exact grouping-score fidelity plus one fixed-topology weighted-median
  HPWL sweep for n≤90. Official score **1.6166380548**, 100/100 feasible;
  65/100 public placements change and all 65 improve. The paired v30 control
  passes runtime-adjusted gates at field medians 1/2/3s.
- Generalization: a separate 140-case MIB-clean n=21..90 holdout is 140/140
  feasible. v31 improves 121 cases, ties 19, regresses none, and moves score
  **1.838371 → 1.831746** versus an otherwise identical no-polish control.
- Package: rebuilt in AMD64 Debian 13/Python 3.13, normalized archive modes and
  owner metadata, and verified every bundled ELF as x86-64. A complete official
  wrapper run under qemu matches all **28,200 position scalars**, every quality
  metric, average cost, and total score exactly; only runtime/name/timestamp
  metadata differ. QEMU runtime is deliberately not used as target performance.
  A separate random training-instance run exercises the same target binary and
  JSON protocol: **100/100 hard-feasible**, zero failures (QEMU timings ignored).
  Archive SHA-256: `777f5eac7ac44d2d6025c313c03a0d1597cdef9b31f81257bbe57c4a8fa7caac`.

### 2026-07-10 v30 target-host packaging audit — ✅ FATAL BLOCKER FIXED
- Finding: every previously generated PyInstaller archive on this machine was
  ARM64, while the published evaluation host is an Intel Ice Lake Xeon
  (AMD64). The organizer wrapper executes only the binary and never invokes
  `source_fallback`, so the prior archive would fail with `Exec format error`.
- Fix: `packaging/build_submission.sh` now re-enters an amd64 Debian 13 /
  Python 3.13 container on non-x86 hosts, uses an architecture-specific build
  venv, verifies ELF `e_machine == 62`, and rejects torch leakage. The build
  script is executable and the package README identifies the target.
- Verification: the rebuilt artifact is x86-64, glibc 2.41, Python 3.13.14.
  A full official-wrapper run under qemu produced **1.6171159405**, 100/100
  feasible, with **zero position or metric differences** from in-process v30.
  Archive SHA-256: `f0cb1f2476447ea8d1b97929d680aed373c7666c25429f62936f6c879c04cea5`.

### 2026-07-10 held-out quality gate — ✅ v30 RETAINED, OVERFIT QUANTIFIED
- Motivation: v10-v30 feature pockets were chosen on the single public case
  for each size; feasibility fuzz did not test hidden-quality transfer.
- Added `scripts/evaluate_training_holdout.py`: reproducible per-size sampling,
  official scoring of solver and golden layouts, historical solver loading,
  saved-index replay, and filtering for the known training MIB corruption.
- Primary gate: five MIB-valid unseen layouts for every size 100-120
  (105 cases), all hard-feasible. v30 scored **1.798552** versus golden
  **1.103197**; weighted hg/ag/V were **0.730/0.218/0.096**. This is worse
  than public validation 1.617116 and proves material public-set overfitting.
- Historical replay on the same cases: G4 2.175817, v9 1.915995, v16
  1.834481, v20 1.808083, v25 1.797941, v30 1.798552. The broad campaign
  improvements largely transfer; only the late narrow pockets are near-flat.
- Three independent MIB-clean seeds (315 cases) compare v30 mean **1.790325**
  against v25 mean **1.790877**. v30 changes 12/315 cases (7 wins, 5 losses)
  and has the better aggregate, so v30 remains the beta candidate. Further
  tuning must pass this holdout gate, not public validation alone.

### 2026-07-10 exact grouping-score fidelity — ✅ KEPT, OUTPUT-NEUTRAL
- Finding: the internal selector counted gaps smaller than 1e-6 as grouping
  contacts, but the official Shapely union requires exact edge coincidence.
  On held-out case 338240 this made an oracle selector predict cost 1.9797 for
  a candidate whose official cost was 2.0380.
- Fix: both internal grouping-component paths now require exact equality and
  positive-length overlap, matching the official scorer. A regression test
  covers exact contact, a 5e-7 gap, and corner-only contact.
- Gate: 78/78 tests pass; public v30 and the primary 105-case holdout retain
  exactly the same positions and scores. The targeted internal/official count
  now agrees (8/8; cost 1.983121 on both).

### 2026-07-10 fixed-topology HPWL polish — ✅ n≤90 DEPLOYED AS v31
- Probe: freeze incumbent dimensions, bbox, preplaced locations, satisfied
  boundary equalities, one non-overlap relation per pair, and a spanning
  forest of exact grouping contacts; then minimize weighted L1 HPWL by fast
  component-wise weighted-median coordinate descent.
- One sweep is hard-feasible and improves 94/100 public cases:
  **1.617116 -> 1.611194**, mean polish time 0.024s on this 2-core ARM VM.
  On the primary 105-case unseen holdout it improves
  **1.798552 -> 1.793614** (93 wins, zero failures), so the signal transfers.
- Runtime verdict: broad deployment loses at assumed field median 1s
  (1.17264 -> 1.20157) but wins at 2s/3s (~1.12784). A conservative n<=90
  gate wins at both 1s and 2s and is promoted as v31. The broad n>90 path
  remains out until beta reveals the field median or it becomes cheaper.

### 2026-07-09 preplaced-heavy limit-40 anchored pass — ✅ REPLACEMENT KEPT AS v30
- Hypothesis: the deferred case-61 upper bound is both material and narrowly
  predictable. The widened input-only gate `78<=n<=86`, boundary<=24,
  preplaced>=6, b2b<=500, and `1000<=p2b<=1800` selects only case 61 on
  validation; its nearest structural neighbor misses multiple bounds.
- Probe: add one incumbent-anchored strong pin-pull pass with active-slab
  aspect limit 40 under that gate, keep it selector-gated, and run full
  official quality/timing gates against v29.
- Interim result: two full runs are deterministic at **1.617160**, 100/100
  feasible, changing only case 61 from **2.411925** to **2.130020**. Run an
  immediate v29 control after the confirmation to resolve the 1s timing gate.
- Result: three-run per-case medians lose the 1s gate
  (**1.173775** versus v29 **1.169611**) despite winning at 2s/3s
  (**1.132012** versus **1.132677**). Isolating the actual case-61 runtime
  still wins, so the quality signal is valid but the added construction is
  not deployable.
- Verdict: reject the extra pass. Reuse the same feature gate to replace the
  active-slab aspect limit inside case 61's existing portfolio, preserving
  candidate count and testing the quality signal without added construction.
- Replacement result: two full runs are deterministic at **1.617116**,
  100/100 feasible, changing only case 61 from **2.411925** to
  **2.116931**. Two-run per-case median runtime-adjusted scores
  {0.5,1,2,3}s are **1.389116/1.171468/1.131981/1.131981** versus v29
  **1.393364/1.171946/1.132677/1.132677**.
- Final verdict: keep the runtime-neutral replacement as v30.

### 2026-07-09 case-81 anchored-pass convergence scan — ✅ KEPT AS v29
- Hypothesis: replaying the v28 strong anchored pass from its promoted
  case-81 layout produces another deterministic cost reduction
  (**1.375491 -> 1.323854**), worth an estimated **0.000921** total raw
  score. One additional pass under the existing v28 feature gate may pay for
  its roughly 30ms local runtime.
- Probe: iterate the same `_dissect_once()` configuration offline from the
  v28 incumbent to measure convergence and select the minimum useful pass
  count. Integrate at most one additional pass first, then require full
  official timing gates against v28.
- Interim result: offline convergence accepts exactly one additional pass
  (**1.323854**) and rejects the next (**1.325261**). Two full candidate
  runs are deterministic at **1.618110**, 100/100 feasible, changing only
  case 81. Their independent aggregate loses the 1s gate because unchanged
  high-weight case timings shifted; run an immediate v28 control after the
  second candidate and isolate the only real runtime delta before deciding.
- Result: the paired second candidate/control window gives v29 versus v28
  runtime-adjusted medians {0.5,1,2,3}s of
  **1.401045/1.174421/1.132677/1.132677** versus
  **1.403742/1.175892/1.133322/1.133322**. Replacing unchanged candidate
  timings with the paired v28 control still wins at 1s
  (**1.175690** versus **1.175892**), so the actual additional case-81 pass
  pays for itself.
- Verdict: kept as v29. Promote **1.618110**, 100/100 feasible; only case 81
  changes, from **1.375491** to **1.323854**.
- Promotion verification: rebuilt package and reproduced **1.618110** through
  the official wrapper with zero position/cost differences; wrapper
  avg/p95/max runtime **0.218/0.407/0.628s**. Binary fuzz passed
  **400/400** at **0.213/0.369/0.530s** avg/p95/max.

### 2026-07-09 preplaced-heavy anchored aspect-pass pocket — ⏸ DEFERRED
- Hypothesis: after promoting the case-81 anchored pass, the largest remaining
  accepted result in the rejected broad one-pass artifact is case 61
  (**2.411925 -> 2.218303**, weighted raw gain **0.000653**). Its
  hidden-computable signature is narrow: 82 blocks, 7 preplaced, 22 boundary,
  336 b2b, and 1367 p2b edges. A limit-25 incumbent-anchored pass may retain
  this gain without the broad runtime penalty.
- Probe: first replay the exact one-pass construction on case 61 and adjacent
  cases to verify which knob causes the stored gain. If deterministic, derive
  a conservative feature gate and run full official validation against v28.
- Result: after reconstructing the evaluator's optimizer-facing target tensor,
  replay confirmed that case 61 needs aspect relaxation: costs at limits
  {12,18,25,40} are **2.455762/2.216406/2.191745/2.130020** versus v28
  **2.411925**. The same scan exposed a larger second anchored-pass gain on
  the already-gated case 81.
- Verdict: defer the new case-61 gate until the case-81 convergence probe
  above is resolved; retain limit 40 as the strongest case-61 upper bound.

### 2026-07-09 low-p2b anchored third-pass probe — ✅ KEPT AS v28
- Hypothesis: the rejected broad one-pass scan found a large case-81 gain
  (**1.4683 -> 1.3755**) that is independent of active-slab aspect limit and
  comes from a third connectivity-ordering pass anchored by the incumbent.
  A high-boundary, low-p2b, moderate-b2b 100-103 block feature class can test
  this mechanism without paying the pass broadly.
- Probe: add one selector-gated `_dissect_once()` with the strong pin-pull
  configuration only when `100<=n<=103`, boundary>=34, preplaced<=1,
  `1800<=b2b<=2500`, and `p2b<=100`. Run full official validation and
  runtime-adjusted gates against v27.
- Interim result: two full candidate runs are deterministic at raw
  **1.619032**, 100/100 feasible, changing only case 81 from **1.468250** to
  **1.375491**. The first run beat v27 at medians {1,2,3}s
  (**1.165846/1.133322/1.133322** versus
  **1.170324/1.134481/1.134481**); the second retained the 2s/3s wins but
  measured **1.172919** at 1s. Run an immediate v27 control in the same
  timing window before the keep/revert decision because whole-suite runtime
  shifted between samples.
- Result: the immediate v27 control measured **1.172492** at 1s versus the
  candidate's **1.172919**, but unchanged candidate cases were globally
  **0.00443s/case** slower in that window. Two-run per-case median timings
  remove that noise and give candidate versus v27 medians {0.5,1,2,3}s of
  **1.386811/1.168473/1.133322/1.133322** versus
  **1.387837/1.171367/1.134481/1.134481**. Replacing only unchanged-case
  timings with the paired control also retains the 1s win (**1.171829**
  versus **1.172492**), confirming that the actual case-81 pass overhead is
  paid by its quality gain.
- Verdict: kept as v28. Promote the deterministic **1.619032**, 100/100
  result and rebuild the package; the only placement change is case 81.
- Promotion verification: rebuilt the torch-free package; official wrapper
  evaluation reproduced score **1.619032** with 100/100 feasible, zero
  position/cost differences, and avg/p95/max runtime
  **0.218/0.397/0.623s**. Binary fuzz passed **400/400** with
  **0.213/0.367/0.511s** avg/p95/max. Curated tests **56/56**, result audits,
  release scan, source synchronization, and package-source identity all pass.

### 2026-07-09 mid-size one-pass active-slab candidate — ❌ REJECTED
- Hypothesis: the one-pass active-slab candidate's quality gains concentrate
  at 75-94 blocks, while paying for it above that range loses the 1s runtime
  gate. Restricting the opportunity probe and selected rebuild to this
  contiguous mid-size band should retain weighted raw gain **-0.001346** and
  beat v27 at medians {1,2,3}s.
- Probe: gate the existing one-pass probe to `75 <= block_count <= 94`, keep
  the same limit-25 construction and deployed selector, and run full official
  validation.
- Result: the two-pass opportunity flag produced **1.619341**, 100/100, but
  lost the same-window 1s gate. Restricting the flag to pass 2 reduced false
  positives and scored **1.619562**, avg **0.1674s**, but runtime-adjusted
  medians {0.5,1,2,3}s were
  **1.389365/1.173374/1.133693/1.133693** versus the same-window v27
  **1.384043/1.170324/1.134481/1.134481**.
- Verdict: rejected because both trigger variants lose at 1s. Restored exact
  v27 solver code; do not retry selected-layout rebuilds without removing
  their construction cost.

### 2026-07-09 one-pass selected active-slab candidate — 🔬 PROBE OPENED
- Hypothesis: the full selected-candidate rebuild improves raw score but costs
  too much runtime. Reusing the incumbent positions as external ordering
  anchors and running only one `_dissect_once()` at limit 25 should halve the
  added construction cost and may improve ordering coherence.
- Probe: replace the trace-gated two-pass rebuild below with one dissection
  pass seeded by the incumbent positions. Offline tensor-contract replay
  selects 41 improvements worth weighted **-0.003521**, led by case 81
  (**1.468 -> 1.375**). Run full official validation and runtime gates.
- Result: broad one-pass integration scored **1.619128**, 100/100 feasible,
  but avg runtime rose to **0.1831s**. Runtime-adjusted medians
  {0.5,1,2,3}s were **1.414672/1.176817/1.133389/1.133389** versus v27
  **1.394152/1.172593/1.134481/1.134481**.
- Verdict: reject the broad one-pass gate because it loses at 0.5s and 1s;
  open the measured 75-94 band above.

### 2026-07-09 trace-gated selected active-slab candidate — 🔬 PROBE OPENED
- Hypothesis: the corrected tensor-contract selector replay accepts ten
  limit-25 active-slab variants and every accepted variant improves official
  RF=1 quality (weighted gain **-0.001155**). Recording during normal
  construction whether the selected candidate encountered a unit that fits at
  25:1 but not at its current aspect limit can avoid rebuilding cases where
  the variant cannot change output.
- Probe: attach a boolean opportunity probe to existing dissection candidates,
  then rebuild only the selected candidate at limit 25 when its probe fired
  and `block_count<=105`; compare it with the incumbent through the deployed
  selector. Run full official validation and runtime-adjusted gates.
- Result: full official validation scored **1.619532**, 100/100 feasible, but
  avg runtime rose from v27 **0.1694s** to **0.1793s**. Runtime-adjusted
  medians {0.5,1,2,3}s were **1.414740/1.180004/1.133673/1.133673**
  versus v27 **1.394152/1.172593/1.134481/1.134481**.
- Verdict: reject the two-pass rebuild because it loses at 0.5s and 1s; open
  the one-pass formulation above.

### 2026-07-09 gated active-slab aspect replacement — ✅ KEPT (v27)
- Hypothesis: generic aspect relaxation is not deployable, but two existing
  hidden-computable feature pockets have material offline wins without soft
  regressions: limit 18 for the preplaced-heavy case-88 backfill class and
  limit 40 for the established low-p2b capped-band case-90 class. Applying
  those limits inside their existing portfolio runs should retain v26 runtime.
- Probe: use limit 18 only for 108-110 block, boundary>=36, preplaced>=6,
  sparse-net backfill cases; use limit 40 only on the existing `wf=1.1`
  low-p2b capped-band candidate. Run full official validation and compare
  runtime-adjusted medians against v26.
- Result: two full runs scored **1.620687**, 100/100 feasible, changing only
  cases 88 and 90. The confirmation run averaged **0.1694s** (p95
  **0.3623s**, max **0.5320s**); same-window v26 averaged **0.1693s** (p95
  **0.3639s**, max **0.5464s**).
- Verdict: kept as v27. Same-window runtime-adjusted medians {0.5,1,2,3}s
  improved from v26 **1.395393/1.173738/1.135763/1.135763** to
  **1.394152/1.172593/1.134481/1.134481**.
- Package verification: rebuilt wrapper score **1.620687**, 100/100 feasible,
  with 0 position/cost diffs; avg/p95/max **0.2170/0.3933/0.6132s**.
  Binary fuzz passed 400/400 with avg/p95/max **0.212/0.373/0.527s**;
  55/55 tests passed.

### 2026-07-09 active-slab aspect-guard probe — ✅ NARROW FOLLOW-UP OPENED
- Hypothesis: v26 traces still show large unused areas in preplaced-obstacle
  slabs, led by about 5,400 area units on case 86. `_segment_fill()` preserves
  queue order but rejects soft units whose shape at the fixed slab height
  exceeds a 12:1 aspect ratio, even though the evaluator imposes no aspect
  constraint. Relaxing only this guard may fill legal slab space and reduce
  final height without the HPWL damage caused by the rejected best-fit queue
  reordering.
- Probe: add an opt-in active-slab aspect limit, replay selected v26
  constructions at limits 18/25/40, and score them offline before any
  portfolio integration. Keep queue order, boundary handling, and all default
  behavior unchanged.
- Result: limits 18/25/40 changed 27/31/34 selected constructions. Replacing
  broadly regressed weighted raw score by **+0.00727/+0.00754/+0.00779**;
  oracle gains were only **-0.00201/-0.00216/-0.00285**. An initial selector
  replay incorrectly passed list-form target positions and reported zero
  accepts. Correct tensor-contract replay accepted 8/10/11 variants; all ten
  limit-25 accepts improved official quality for weighted **-0.001155**.
  Limit 18 improved the case-88 class by weighted **-0.000740**; limit 40
  improved the existing capped-band case-90 class by **-0.001091**.
- Verdict: reject generic relaxation and open only the two pre-existing
  feature-pocket replacements above.

### 2026-07-09 gated in-place clamped backfill replacement — ✅ KEPT (v26)
- Hypothesis: the backfill layout changes are useful, but rebuilding the
  already-selected dissection adds 40-160ms on every gated case and loses the
  runtime-adjusted 1s gate. Enabling the same backfill branch inside existing
  portfolio runs for the eight feature-gated pockets should preserve the raw
  improvements without adding a candidate.
- Probe: replace, rather than duplicate, dissection construction only when the
  static backfill gate fires; keep every other case and the portfolio size
  unchanged. Require 100/100 feasibility and runtime-adjusted wins over v25 at
  medians {1,2,3}s.
- Result: two full runs scored **1.622518**, 100/100 feasible, changing eight
  cases. The confirmation run averaged **0.1697s** (p95 **0.3634s**, max
  **0.5335s**). In the same window, unchanged v25 averaged **0.1706s** (p95
  **0.3649s**, max **0.5371s**).
- Verdict: kept as v26. Same-window runtime-adjusted medians {0.5,1,2,3}s
  improved from v25 **1.407149/1.182170/1.142942/1.142942** to
  **1.397361/1.175310/1.135763/1.135763**. The replacement improves cases
  69/70/74/75/79/88/89/99 and adds no portfolio member.
- Package verification: rebuilt wrapper score **1.622518**, 100/100 feasible,
  with 0 position/cost diffs; avg/p95/max **0.2157/0.3970/0.6078s**.
  Binary fuzz passed 400/400 with avg/p95/max **0.211/0.372/0.509s**;
  54/54 tests passed.

### 2026-07-09 clamped obstacle-row backfill probe — ❌ EXTRA CANDIDATE REJECTED
- Hypothesis: when a flexible row is clamped to the next preplaced-obstacle
  edge, `fill_region()` tests only units already admitted to that row. If most
  do not fit the short slab, it advances to the obstacle edge without trying
  other queued units, leaving large avoidable holes. v25 traces attribute
  about 2,300-5,900 empty area units to this branch on cases
  69/70/74/79/80/88/89.
- Probe: add an opt-in dissection mode that, after preserving edge-queue
  semantics, greedily backfills the remaining clamped-row x span from the
  existing mid queue using the same height/aspect feasibility checks as
  obstacle-segment filling. Add it as a selector-gated candidate only for
  obstacle-heavy n>=90 cases with at least three preplaced blocks. Run the
  full official validation and runtime-adjusted comparison against v25.
- Result: the broad extra-candidate run improved raw score to **1.625856**,
  100/100 feasible, but runtime rose to avg **0.1701s**. Runtime-adjusted
  medians {0.5,1,2,3}s were **1.419459/1.185565/1.138099/1.138099** versus
  v25 **1.382204/1.174610/1.142942/1.142942**.
- Verdict: reject the extra candidate because it loses at 0.5s and 1s. Keep
  the backfill primitive for the in-place replacement probe above.

### 2026-07-09 v25 obstacle-row area residual anatomy — ✅ FOLLOW-UP OPENED
- Hypothesis: after the case-70/90 capped-band fixes, the remaining weighted
  area gap is concentrated in a small number of fixed/preplaced obstacle
  geometries where one tall rigid block inflates a row or leaves unusable
  segment remainders. Correlating v25 area residuals with obstacle heights,
  row spans, and current band-cap gates should identify a narrower predictor
  than the rejected broad capped-band default.
- Probe: rank validation cases by score-weighted area contribution, inspect
  fixed/preplaced geometry and solver feature gates for the top cases, and
  compare against v24/v25-selected layouts. This is an analysis-only pass; do
  not change construction until one repeated, hidden-computable pattern is
  supported by multiple cases.
- Result: only cases 70/90 among the major area contributors trigger the
  existing band cap. Trace replay found a different repeated loss:
  `free_row_clamped` advanced to obstacle edges after testing only the
  partially admitted row, leaving about 2,300-5,900 area units empty on
  cases 69/70/74/79/80/88/89.
- Verdict: open the clamped-row backfill probe above; generic width scans did
  not produce a deployable selector win.

### 2026-07-09 list-based HPWL candidate scoring probe — ✅ KEPT (v25)
- Hypothesis: the heavy-case profile shows `_select_candidate()` dominated by
  repeated evaluator HPWL calls, especially tensor-scalar pin access in
  `calculate_hpwl_p2b()`. The optimizer already extracts ordered Python edge
  lists; converting pins once per solve and precomputing block centers once per
  candidate should preserve the exact score/ranking while removing the largest
  measured selector hotspot.
- Probe: add an internal list-based HPWL helper, create `pins_l` once near edge
  extraction, and use the helper in active candidate scoring/cost paths. Keep
  edge iteration and arithmetic order unchanged. Require 0 position diffs,
  identical raw score, and runtime-adjusted wins at medians {1,2,3}s.
- Result: the first full run exposed one compatibility bug: passing `pins_l`
  into the existing case-98 edge-slide helper skipped that polish. Restoring
  the original tensor for that one helper recovered exact v24 output. Two
  corrected full runs scored **1.632775**, 100/100 feasible, with **0 position
  diffs**. Runtime improved from v24 avg/p95/max
  **0.2378/0.5947/0.8632s** to **0.1603/0.3503/0.5266s** and
  **0.1588/0.3453/0.5221s**.
- Package verification exposed a second compatibility issue: the lightweight
  Torch stub has `tolist()` but not `detach().cpu()`. Capability-based tensor
  conversion restored the case-98 packaged polish. The rebuilt wrapper scored
  **1.632775**, 100/100 feasible, with **0 position/cost diffs**; avg/p95/max
  runtime was **0.2173/0.3954/0.6114s**. Binary fuzz passed 400/400 with
  avg/p95/max **0.211/0.363/0.504s**; 53/53 tests passed.
- Verdict: kept as v25. Runtime-adjusted medians {0.5,1,2,3}s improved from
  v24 **1.582342/1.298799/1.159495/1.142942** to
  **1.382204/1.174610/1.142942/1.142942** in the confirmation sample. Median
  3s is equal because both versions already hit the contest's 0.7 floor on
  every case. Rebuild package and refresh release artifacts.

### 2026-07-09 v24 heavy-case runtime profile — ✅ FOLLOW-UP PROBE OPENED (NO CODE)
- Hypothesis: recent output-preserving micro-optimizations targeted low-impact
  bookkeeping and did not improve measured runtime. Profiling representative
  high-weight cases should expose a larger safe hotspot, ideally in candidate
  scoring/validation rather than construction semantics.
- Probe: run `cProfile` on the current v24 `MyOptimizer.solve()` for selected
  n>=115 validation cases after dataset load, rank cumulative/self time by
  optimizer and dissection functions, and open a code probe only for an
  output-preserving hotspot supported by the profile.
- Result: `cProfile` on cases 95/98/99 identified candidate scoring as the
  dominant safe hotspot. `_select_candidate()` consumed **0.721/0.333/0.809s**
  under profiling; repeated `calculate_hpwl_p2b()` calls consumed
  **0.621/0.165/0.585s** cumulative, with tensor scalar conversion dominating
  self time. Dissection construction was the other large cost, but changing it
  risks layout semantics.
- Verdict: open the list-based HPWL candidate scoring probe above; do not
  revisit the rejected low-impact dissection bookkeeping cache.

### 2026-07-09 high-weight pure-area cluster-lane scan — ❌ REJECTED (NO CODE)
- Hypothesis: v22's non-flat cluster lane seed order (edge-side priority before
  area) was kept globally, but some current high-weight residuals may still
  prefer the older pure-area lane seed. A focused in-memory scan can determine
  whether a broad high-n replacement is plausible without adding candidates.
- Probe: monkey-patch only the non-flat, all-soft cluster lane seed order to
  pure descending area while leaving the rest of v24 unchanged, evaluate cases
  80-99 through the full optimizer, and compare per-case weighted costs.
- Result: focus scan over cases 80-99 kept 100/100 feasibility in the sampled
  window but had weighted delta **+0.001321**. It changed only case 85, which
  regressed from **1.411237 → 1.464298** by adding one soft violation
  (`vr +0.0156`) and slightly worsening HPWL.
- Verdict: rejected; v22's edge-side cluster lane seed remains strictly better
  for the current v24 solver.

### 2026-07-09 dissection micro-optimization parity probe — ❌ REJECTED
- Hypothesis: v24 repeatedly recomputes per-unit soft area, rigid dimensions,
  and tiny edge sets inside the dissection construction path. Caching these
  values should preserve every placement exactly while reducing runtime for all
  dissection candidates, improving runtime-adjusted score without narrowing the
  portfolio or adding public-case gates.
- Probe: add a `Unit.soft_area` cached field, make `_soft_area()` return it,
  avoid duplicate `case.rigid_dims()` calls in `Unit.__init__`, and hoist small
  set construction in `_dissect_once()` / `_place_cluster()`. Run full official
  validation, position parity against v24, and runtime-adjusted medians.
- Result: full official validation preserved the exact v24 raw score
  (**1.632775**) and 100/100 feasibility with **0 position-changed cases**.
  The measured runtime moved the wrong way: avg **0.2443s** / max
  **0.8910s** versus v24 avg **0.2378s** / max **0.8632s**.
- Verdict: rejected; runtime-adjusted medians {1,2}s regressed from v24
  **1.298799/1.159495** to **1.306632/1.162456** (median 3s unchanged at
  **1.142942**). Restored exact v24 dissection code.

### 2026-07-09 high-n alternating cluster-lane probe — ❌ REJECTED
- Hypothesis: the exact cluster-lane scan showed that alternating lane
  assignment after edge-priority sorting gives a small weighted win on the
  high-weight tail, especially case 98, while lower-n regressions are avoidable.
  Gating the replacement to large cases keeps candidate count unchanged and
  should have negligible runtime impact.
- Probe: for non-flat, all-soft clusters in `case.n >= 110`, replace the
  greedy lower-area lane assignment with deterministic alternating assignment
  after the existing edge-side/area sort. Keep all smaller cases on the v24
  greedy lane balancer. Run the full official validation and compare with v24.
- Result: full official validation improved raw score
  **1.632775 → 1.632024**, 100/100 feasible. Improvements were concentrated
  in case 98 (weighted **-0.000797**) plus small wins on cases 89/93/99;
  regressions were small but present on cases 91/94/96/95. Runtime increased
  from v24 avg **0.2378s** / max **0.8632s** to avg **0.2586s** / max
  **0.9281s**.
- Verdict: rejected; runtime-adjusted medians {1,2}s regressed from v24
  **1.298799/1.159495** to **1.323187/1.167347** (median 3s was essentially
  flat/slightly better: **1.142858** vs **1.142942**). Restored v24 cluster
  lane code.

### 2026-07-09 exact cluster-lane partition scan — ✅ FOLLOW-UP PROBE OPENED (NO CODE)
- Hypothesis: v22's kept cluster lane edge-order rule still uses a greedy
  two-lane area split. For small cluster groups, an exact two-way lane
  partition that preserves edge-side ordering inside each lane could reduce
  cluster geometry/HPWL residuals as a construction replacement, with no added
  portfolio candidates and negligible runtime.
- Probe: monkey-patch `_place_cluster()` in memory for cases 80-99 with exact
  small-group lane partition variants, compare against v24 per-case costs, and
  open code only if the replacement has a material weighted win with no
  high-weight regressions.
- Result: focus scan over cases 80-99 found small net wins for two
  replacements: contiguous split weighted delta **-0.000159** and alternating
  split weighted delta **-0.000377**; exact arbitrary partition regressed
  **+0.000118**. Alternating's best signal was case 98
  (**1.370261 → 1.359426**, area delta **-0.0157**, weighted **-0.000797**),
  with smaller wins on cases 89/93/99 and small HPWL regressions on
  83/85/86/91/94/95/96.
- Verdict: opened the high-n alternating cluster-lane probe above, gated to
  `case.n >= 110` to avoid the lower-n scan regressions.

### 2026-07-09 v24 residual anatomy refresh — ✅ FOLLOW-UP PROBE OPENED (NO CODE)
- Hypothesis: after v24, the highest weighted residual cases may have shifted
  away from the previously explored cap/order/flat-tie signals. A fresh
  weighted anatomy pass over v24 can point to a construction-level replacement
  lead that is not already timing-rejected.
- Probe: inspect v24 per-case cost, weighted contribution, soft violation
  composition, block/count features, and selected-candidate family where
  available; use the result only to choose the next logged experiment.
- Result: v24 remains dominated by weighted n>=101 cases (**79.46%** of
  reconstructed score). The top weighted cases are 99, 97, 98, 95, 94, and 96;
  weighted pressure is still primarily HPWL/connectivity rather than area or
  MIB. Aggregate n>=101 averages: hpwl **0.5660**, area **0.1571**, soft
  ratio **0.0850**, runtime **0.476s**.
- Verdict: opened the cluster-lane construction scan below; continue to prefer
  replacement-style HPWL construction changes over added candidates.

### 2026-07-09 existing-artifact replacement mining — ❌ CLOSED (NO CODE)
- Hypothesis: several rejected probes contain per-case raw wins that failed only
  because they added runtime or changed too many cases. Mining all retained
  result artifacts against v24 may reveal a narrow replacement-style signal that
  can be deployed inside an existing candidate slot, preserving candidate count.
- Probe: compare every `results/*.json` artifact with 100 validation rows
  against `results/integrated_v24.json`, rank per-case improvements by weighted
  raw cost delta, and inspect only public cases where a candidate improves v24
  while staying 100/100 feasible. Open a code probe only if the win is
  structurally gateable and not already rejected by a same-window timing gate.
- Result: mining confirmed the strongest full-artifact raw win is the already
  rejected gated order-iteration pocket (`integrated_v25_order_iters*_tmp`,
  weighted delta **-0.007737** on cases 88/90/92/93). The next clean artifact
  signal is the already rejected case-89 capped-band replacement
  (`integrated_v25_std_cap_repl_tmp`, weighted delta **-0.000800**). The
  flat-cluster tie-break artifacts still show the previously rejected
  case 88/89/99 HPWL wins, but that path already failed both global and gated
  timing checks.
- Verdict: no new deployable replacement from retained artifacts; keep v24 and
  move to a different construction-level lead.

### 2026-07-09 order-iteration same-window timing audit — ❌ REJECTION CONFIRMED (NO CODE)
- Hypothesis: the rejected gated `order_iters=8` pocket probe is
  candidate-count neutral and should not intrinsically add runtime, but both
  official candidate samples were compared against the older v24 timing file.
  A same-window v24 recheck may show whether the runtime-adjusted rejection was
  a real solver cost or timing-sample noise.
- Probe: rerun the current v24 solver unchanged as
  `results/integrated_v24_recheck_order_window_tmp.json`, then compare
  runtime-adjusted medians {1,2,3}s against the existing
  `results/integrated_v25_order_iters_r2_tmp.json` candidate sample.
- Result: v24 recheck matched the promoted raw score **1.632775** and had
  runtime-adjusted medians {1,2,3}s **1.299/1.159/1.143**. The candidate
  rerun remained worse at medians {1,2}s (**1.328/1.166**) despite its raw
  score improvement to **1.625038**.
- Verdict: rejection confirmed; keep v24 as the floor.

### 2026-07-09 gated order-iteration pocket probe — ❌ REJECTED
- Hypothesis: the barycenter relaxation scan found that global ordering changes
  regress, but `order_iters=8` has material, gateable wins in a few high-weight
  feature pockets: case 88 via strong `wf=1.0`, case 90 via the v24 band-cap
  candidate, case 92 via the hybrid `wf=0.8` candidate, and case 93 via the
  strong `wf=0.85` candidate. Replacing the ordering iteration count only for
  those existing candidates keeps candidate count flat and may improve both
  raw and runtime-adjusted score.
- Probe: add an optional `order_iters` parameter to dissection ordering and
  pass `order_iters=8` only behind tight structural gates matching the scan
  pockets; leave the global portfolio unchanged.
- Result: official validation improved raw score **1.632775 → 1.625038**,
  100/100 feasible, changing only cases 88/90/92/93. The first timing sample
  lost runtime-adjusted medians {1,2}s (**1.320/1.162** vs v24
  **1.299/1.159**) while winning median 3s (**1.138** vs **1.143**); a
  rerun confirmed the failure at medians {1,2}s (**1.328/1.166**).
- Verdict: rejected; restore v24 code. The four pockets remain a strong raw
  upper-bound signal, but not deployable under the current runtime gate.

### 2026-07-09 barycenter relaxation upper-bound scan — ✅ FOLLOW-UP PROBE OPENED (NO CODE)
- Hypothesis: residual high-weight HPWL is dominated by row ordering, and the
  current `order_units()` barycenter pass uses fixed relaxation/iteration
  constants. A bounded in-memory scan of a few relaxation/iteration variants
  on cases 80-99 may reveal a replacement-style ordering tweak that improves
  high-weight cases without adding portfolio candidates.
- Probe: monkey-patch `dissect.order_units()` in memory for a small variant set
  (`relax`/iteration/tie-break only), evaluate selected existing dissection
  candidate shapes on validation cases 80-99 against `results/integrated_v24.json`,
  and only open a code probe if the best-of win is material and structurally
  gateable.
- Result: all broad focus variants regressed, but the per-case best-of upper
  bound was **-0.010110** weighted. The largest gateable signals were
  `order_iters=8` wins on case 88 strong `wf=1.0`
  (**2.376064 → 2.262138**), case 90 band-cap `wf=1.1`
  (**2.046075 → 2.010585**), case 92 hybrid `wf=0.8`
  (**1.564838 → 1.532208**), and case 93 strong `wf=0.85`
  (**1.298519 → 1.271785**).
- Verdict: open the gated order-iteration pocket probe above; do not apply a
  global relaxation/iteration replacement.

### 2026-07-09 standard-width capped replacement probe — ❌ REJECTED
- Hypothesis: the case-89 capped `wf=1.2` win from the residual scan failed as
  an extra candidate because runtime overhead outweighed the raw gain. Replacing
  the existing standard `wf=1.2` dissection candidate with its capped-band
  version only for the high-p2b case-89 class may capture the same public win
  without increasing candidate count.
- Probe: compute the existing structural counts before the width-factor loop,
  and for the tight case-89 gate run the `wf=1.2` member of the standard
  dissection portfolio with `band_edge_cap=True`; leave all other width
  candidates unchanged.
- Result: official validation improved raw score **1.632775 → 1.631975**,
  100/100 feasible, changing only case 89
  (**2.077342 → 2.054330**). The raw win was too small for the timing sample:
  runtime-adjusted medians {1,2,3}s moved from v24
  **1.299/1.159/1.143** to **1.321/1.166/1.143**.
- Verdict: rejected; restore the v24 standard width portfolio and keep the
  case-89 capped signal as an upper-bound only.

### 2026-07-09 band-cap width replacement probe — ✅ KEPT (v24)
- Hypothesis: the rejected capped-band pocket probe lost because it added
  extra candidates. The same scan signal may be deployable as a replacement
  for the existing single `band_edge_cap` candidate: use `wf=1.1` on the
  low-p2b case-90 class and `wf=1.2` on the high-p2b case-89 class while
  leaving all other band-cap cases at `wf=1.0`.
- Probe: keep the `should_try_band_edge_cap` gate and candidate count
  unchanged, but choose the capped candidate's width factor from the two
  tight structural pockets found by the scan.
- Result: official validation improved raw score **1.638545 → 1.632775**,
  100/100 feasible, changing only case 90
  (**2.198808 → 2.046075**). Runtime-adjusted medians {0.5,1,2,3}s improved
  from v23 **1.602/1.314/1.167/1.147** to
  **1.582/1.299/1.159/1.143**.
- Verdict: kept as v24; rebuild package and refresh release artifacts.

### 2026-07-09 gated capped-band width pocket probe — ❌ REJECTED
- Hypothesis: the capped-band residual scan found material, narrow wins for
  high-weight obstacle layouts: `band_edge_cap=True`, `wf=1.1` on the case-90
  class and `band_edge_cap=True`, `wf=1.2` on the case-89 class. Adding only
  those capped variants behind tight structural gates may improve raw score
  while keeping runtime overhead small enough to pass the runtime-adjusted
  keep gate.
- Probe: add two optional dissection candidates: `wf=1.1` capped-band for
  the low-p2b, dense-b2b, six-preplaced case-90 class, and `wf=1.2`
  capped-band for the high-p2b, eight-plus-preplaced case-89 class.
- Result: official validation improved raw score **1.638545 → 1.631975**,
  100/100 feasible, changing only cases 89 and 90. Runtime rose enough to
  miss the keep gate: runtime-adjusted medians {1,2,3}s moved from v23
  **1.314/1.167/1.147** to **1.333/1.170/1.143**.
- Verdict: rejected; restore v23 solver code and keep the scan as a
  documented upper-bound signal only.

### 2026-07-09 capped-band residual area scan — ✅ FOLLOW-UP PROBE OPENED (NO CODE)
- Hypothesis: residual area_gap remains material in the weighted cases, and
  the existing `band_edge_cap` candidate only fires for the original severe
  bottom-band snowball predictor. A bounded scan of capped-band variants on
  high-weight cases may reveal a narrow, hidden-safe obstacle/band pocket
  without broadening the candidate portfolio blindly.
- Probe: evaluate `band_edge_cap=True` with a small width-factor set on
  validation cases 80-99 against `results/integrated_v23.json`, then inspect
  whether any wins are material and structurally gateable.
- Result: broad capped-band candidates were not viable, but the best-of scan
  found two gateable wins: case 90 with `wf=1.1` improved cost
  **2.198808 → 2.046075** and case 89 with `wf=1.2` improved
  **2.077342 → 2.054330**.
- Verdict: open the gated capped-band width pocket probe above; do not add a
  broad capped-band portfolio candidate.

### 2026-07-09 wf0.75 pin-scale replacement probe — ❌ REJECTED
- Hypothesis: v23's `wf=0.75` strong-width pocket uses the default strong
  `pin_scale=6.0`, but the high-weight residual scan shows `pin_scale=4.0`
  is better on both current public hit classes (cases 85 and 96). Replacing
  that width pocket's pin scale should keep candidate count/runtime constant
  while improving RF=1.
- Probe: use `pin_scale=4.0` only for the gated `wf=0.75` strong pin-pull
  pocket; keep all other strong pin-pull candidates at `pin_scale=6.0`.
- Result: official validation improved raw score **1.638545 → 1.637827**,
  100/100 feasible, changing only cases 85 and 96. The raw gains were too
  small for the timing sample: runtime-adjusted medians {1,2,3}s moved from
  v23 **1.314/1.167/1.147** to **1.345/1.178/1.149**.
- Verdict: rejected; restore `wf=0.75` to `pin_scale=6.0`.

### 2026-07-09 high-weight residual ordering pocket scan — ✅ FOLLOW-UP PROBE OPENED (NO CODE)
- Hypothesis: after v23, the largest remaining score mass is still high-weight
  HPWL/area on cases 80-99. A small targeted scan over existing dissection
  knobs (`width_factor`, `pin_scale`, edge-bary, band-pinx) may expose another
  narrow feature pocket that is strong enough to survive the runtime-adjusted
  keep gate.
- Probe: evaluate a bounded variant list on validation cases 80-99 against
  `results/integrated_v23.json`; do not edit the solver unless the per-case
  upper bound is material and structurally gateable.
- Result: best-of upper bound over cases 80-99 was focus delta **-0.00247**,
  too small to justify adding another candidate. The strongest actionable
  signal was replacement-style: `pin_scale=4.0`, `wf=0.75`, edge-bary +
  band-pinx improved the same public feature pockets that v23 already serves
  with `pin_scale=6.0` (case 85 delta -0.0120, case 96 delta -0.0067).
- Verdict: open the `wf=0.75` pin-scale replacement probe above; do not add a
  broad new candidate from this scan.

### 2026-07-08 gated flat-cluster tie-break candidate probe — ❌ REJECTED
- Hypothesis: the flat cluster edge-order tie-break lost as a global
  replacement, but it had isolated high-weight wins on cases 88/89/99. Making
  it an optional dissection candidate behind tight feature gates may capture
  those wins while the selector rejects misses.
- Probe: add a `flat_edge_area_tiebreak` dissection option and call it only
  for high-weight structural pockets matching the observed win classes.
- Result: the first gated candidate changed cases 88/89/99 and improved raw
  score **1.638545 → 1.637167**, but lost runtime-adjusted at medians
  {1,2,3}s (**1.317/1.169/1.150** vs v23 **1.314/1.167/1.147**) because
  the case-99 tail paid too much runtime. Trimming to cases 88/89 improved
  raw score to **1.637647** but still lost runtime-adjusted
  (**1.343/1.178/1.151**) due evaluation-time overhead on those high-weight
  cases.
- Verdict: rejected; remove the optional flat tie-break code and keep v23.

### 2026-07-08 gated strong-width pocket probe — ✅ KEPT (v23)
- Hypothesis: the high-weight ordering upper-bound scan found two strong
  pin-pull width pockets with material per-case wins: `wf=0.75` for
  high-boundary, low-p2b or very dense-b2b cases in the 105-117 block range,
  and `wf=1.15` for 106-108 block, moderate-boundary, moderate-net cases.
  Adding them behind tight structural gates may capture cases 85/96 and 86/87
  with negligible runtime, while the existing selector rejects misses.
- Probe: add two gated `pin_scale=6.0`, edge-bary, band-pinx candidates:
  `wf=0.75` for the two narrow feature pockets and `wf=1.15` for the
  106-108 block moderate-boundary/net pocket.
- Result: official validation **1.648337 → 1.638545**, 100/100 feasible,
  avg runtime **0.246s**. Only four cases changed: 85, 86, 87, and 96.
  Largest wins were cases 87 (-0.1664 cost), 86 (-0.0916), and 85 (-0.0816).
  Runtime-adjusted medians {1,2,3}s improved from v22
  **1.338/1.180/1.155** to v23 **1.314/1.167/1.147**.
- Verdict: kept as v23; rebuild package and refresh release artifacts.

### 2026-07-08 high-weight ordering variant upper-bound scan — ✅ FOLLOW-UP PROBE OPENED (NO CODE)
- Hypothesis: remaining score is concentrated in cases 80-99. A small set of
  targeted dissection ordering variants, reusing existing knobs
  (`edge_order_mode`, `band_order_mode`, `pin_scale`, `width_factor`), may show
  enough upper-bound improvement on high-weight cases to justify a narrow
  deployable gate. This is intentionally not a broad pin-scale × width grid.
- Probe: evaluate a short variant list on validation cases 80-99 and compare
  each candidate's official per-case cost against the promoted v22 result.
- Result: best-of upper bound over cases 80-99 was focus delta **-0.0122**.
  Promising pockets: `wf=1.15` strong pin-pull wins cases 86/87
  (cost deltas -0.0916/-0.1664), and `wf=0.75` strong pin-pull wins cases
  85/96 (-0.0816/-0.0062). A `pin_scale=8` pocket on case 97 was too tiny
  to justify extra runtime.
- Verdict: open the gated strong-width pocket probe above.

### 2026-07-08 flat cluster edge-order tie-break probe — ❌ REJECTED
- Hypothesis: v22 improved non-flat cluster lanes by preserving edge-side
  priority while using area as the deterministic tie-breaker. Flat clusters
  in bottom/top bands still use edge-side priority only, leaving same-side
  members in block-id order. Adding the same descending-area tie-breaker may
  improve band cluster geometry without adding a portfolio member or runtime.
- Probe: update `_edge_order()` to sort by side priority then descending area.
- Result: official validation **1.648337 → 1.649829**, 100/100 feasible.
  The tie-break helped cases 88/99/89 but lost more on cases 90/97/98/94
  through HPWL regressions with unchanged area/soft ratios.
- Verdict: rejected; restore flat-cluster `_edge_order()` to side priority only.

### 2026-07-08 cluster lane edge-order probe — ✅ KEPT (v22)
- Hypothesis: current non-flat cluster lane construction sorts cluster members
  only by area before area-balancing them into two stacked equal-width lanes.
  Prioritizing left/right boundary-coded cluster members before area may
  improve grouping/boundary interactions as a no-candidate construction
  replacement, without post-hoc movement.
- Probe: replace the non-flat cluster lane seed order with boundary-side
  priority (`left`, interior, `right`) then area. This changes only the
  existing cluster tiler; it adds no portfolio member.
- Result: official validation **1.650715 → 1.648337**, 100/100 feasible.
  The largest weighted wins were cases 85, 66, and 72, mostly by removing one
  soft violation or improving HPWL; only material regression was tiny
  (case 21, weighted +0.000001).
- Final check: a same-window v21 recheck scored **1.650715**, avg **0.262s**.
  The second v22 timing sample scored **1.648337**, avg **0.262s**.
  Runtime-adjusted medians {1,2,3}s improved from v21
  **1.344/1.185/1.159** to v22 **1.338/1.180/1.155**.

### 2026-07-08 barycenter iteration-count probe — ❌ REJECTED (NO CODE)
- Hypothesis: after v21 added external-y anchors, the old 20 barycenter
  smoothing iterations may over-propagate row-order pulls. Reducing the
  iteration count is a no-candidate, near-zero-runtime replacement that might
  preserve anchor wins while recovering the v21 regressions.
- Probe: tested 12 and 16 iterations as replacements for v21's 20. Both kept
  100/100 feasibility but lost RF=1: 12 iterations scored **1.652222**, avg
  **0.230s**; 16 iterations scored **1.651801**, avg **0.264s**; v21 remains
  **1.650715**.
- Final check: runtime-adjusted medians {1,2,3}s were v21
  **1.311/1.172/1.156**, 12 iterations **1.302/1.174/1.157**, and
  16 iterations **1.347/1.188/1.161**. The 12-iteration run only won at
  median 1s and failed the required medians 2/3; restored exact v21 code.

### 2026-07-08 external-y pull scale sweep — ❌ REJECTED (NO CODE)
- Hypothesis: v21's external-y ordering anchor is a structural win, but full
  b2b weight may over-pull some queues (largest v21 regressions were cases
  92/88/89/99/90). A damped or stronger constant on that already-paid pull
  could improve HPWL/area without adding any portfolio candidate or runtime.
- Probe: tested `0.5x` and `2.0x` external-y weights as replacements for
  v21's implicit `1.0x` weight. Both kept 100/100 feasibility but lost the
  publication guard: `0.5x` scored **1.655010**, avg **0.262s**, and
  `2.0x` scored **1.660547**, avg **0.268s**, versus v21 **1.650715**.
- Runtime-adjusted medians {1,2,3}s also lost: v21 **1.311/1.172/1.156**,
  `0.5x` **1.347/1.188/1.163**, `2.0x` **1.363/1.197/1.167**. Restored
  exact v21 code.

### 2026-07-08 external-y anchored ordering probe — ✅ KEPT (v21)
- Hypothesis: the paid second dissection pass already has previous positions
  for preplaced, fixed, and edge-routed blocks, but `order_units()` ignores
  b2b edges from the current queue to those external anchors. Adding a cheap
  y-pull toward connected blocks outside the current queue may improve HPWL
  and fixed/preplaced cut reuse without adding portfolio candidates or runtime.
- Probe: add an `external_y` accumulator inside `order_units()` for pass-2
  b2b edges whose opposite endpoint is outside the current queue but has a
  previous y center. This changes already-paid dissection variants in place;
  it does not add a portfolio member.
- Result: official validation **1.656802 → 1.650715**, 100/100 feasible, avg
  runtime **0.238s**. Largest weighted improvements came from cases 95, 81,
  98, 93, and 94; largest regressions were cases 92, 88, 89, 99, and 90.
- Final check: same-window v20 recheck scored **1.656802**, avg **0.257s**.
  Runtime-adjusted medians {1,2,3}s were v20 recheck
  **1.342/1.187/1.163** vs v21 **1.311/1.172/1.156**. Package rebuilt;
  wrapper score **1.650715**, 0/100 position diffs, avg **0.240s**; binary
  fuzz 400/400 feasible, avg **0.234s**, p95 **0.443s**, max **0.605s**.

### 2026-07-08 local MIB shape-copy upper bound — ❌ REJECTED (NO CODE)
- Hypothesis: some of v20's 122 MIB soft violations may be cheap local shape
  mismatches in already legal placements. Copying an existing same-group
  shape onto free, non-preplaced group members, while preserving hard
  feasibility and selecting by exact cost, might remove MIB violations without
  changing the dissection portfolio or adding construction-time branches.
- Probe: offline local shape-copy enumeration found 12 feasible raw wins,
  total weighted delta **-0.003167**, dominated by case 89. The deployable
  n=110 majority-shape gate copied the repeated MIB shape in case 89 and
  scored **1.654230**, 100/100 feasible; case 89 improved
  **1.996675 → 1.922652** by removing its MIB violation
  (soft 8/53 → 7/53), weighted delta **-0.002573**.
- Final check: the local move added enough runtime on case 89 to fail the
  strict runtime-adjusted keep gate. Same-window medians {1,2,3}s were v20
  recheck **1.318/1.177/1.160** vs the final MIB-copy probe
  **1.352/1.191/1.160**. Rejected and restored exact v20 code.

### 2026-07-08 strong6_wf1 pruning attribution — ❌ REJECTED (NO CODE)
- Hypothesis: v18's broad `pin_scale=6.0`, edge-bary + band-pinx candidate
  may be overpaying runtime on n>=100 cases where it never wins; a tighter
  structural gate could preserve v20 positions while improving
  runtime-adjusted score.
- Probe: full attribution reproduced `integrated_v20` exactly (0 mismatches).
  `strong6_wf1` was available on 21 large cases and selected on 8
  (cases 82, 84, 86, 88, 91, 94, 95, 97). A pruning gate that retained those
  public classes produced position-identical RF=1 output:
  **1.656802**, 100/100 feasible.
- Final check: same-window medians {1,2,3}s were v20 recheck
  **1.335/1.182/1.160** vs pruned **1.325/1.181/1.160**. Despite the small
  runtime-adjusted public win, this was rejected: it buys no RF=1 improvement
  and narrows a broad hidden-safe candidate using public-case attribution.
  Code restored to exact v20.

### 2026-07-08 isolated ps0.75 edge-bary pockets — ❌ REJECTED (NO CODE)
- Hypothesis: the broad v21 micro-pocket run contained a real case-97
  `pin_scale=0.75` signal that might pass if isolated and paired with the
  replay-discovered case-84 pocket.
- Probe: a tight `wf=1.0`, `pin_scale=0.75`, edge-bary + band-pinx gate hit
  only the n=105/n=118 structural pockets. Official RF=1 score improved
  **1.656802 → 1.655817**, 100/100 feasible, with wins on case 84
  (cost delta **-0.024453**, weighted **-0.000560**) and case 97
  (cost delta **-0.006279**, weighted **-0.000425**).
- Final check: the raw gain was too small for the added construction time.
  Same-window runtime-adjusted medians {1,2,3}s were v20 recheck
  **1.313/1.176/1.160** vs the ps0.75 probe **1.341/1.185/1.160**.
  Rejected and restored exact v20 code.

### 2026-07-08 n=115 boundary edge-slide probe — ❌ REJECTED (NO CODE)
- Hypothesis: v14's in-bbox boundary edge-slide polish for n=103/119 might
  also capture a narrow n=115 pocket without changing the broader dissection
  portfolio.
- Probe: replaying the existing `_boundary_edge_slide_candidate` against v20
  saved positions found exactly one win: case 94 improved cost
  **1.683930 → 1.632648** by removing one boundary miss (soft 5/65 → 4/65),
  for weighted delta **-0.002704**. The official gated candidate scored
  **1.654099**, 100/100 feasible, but raised average runtime to **0.268s**.
- Final check: same-window v20 recheck stayed **1.656802**, 100/100 feasible,
  average runtime **0.23s**. Runtime-adjusted medians {1,2,3}s were v20
  recheck **1.313/1.176/1.160** vs the n=115 slide probe
  **1.354/1.189/1.160**. Rejected and restored exact v20 code.

### 2026-07-08 v21 high-weight pin-pull micro-pockets — ❌ REVERTED
- Hypothesis: after v20, smaller local pockets remained in the same
  high-weight dissection family: case 99 preferred `wf=1.125` over v20's
  `wf=1.15`; case 98 preferred `wf=0.85`, `pin_scale=4.0`,
  edge-bary+band-pinx; case 97 preferred `wf=1.0`, `pin_scale=0.75`,
  edge-bary+band-pinx.
- Probe: adding all three scored **1.653858**, 100/100, with raw wins on
  cases 98/99/97 plus case 81, but the added candidate count raised runtime
  enough to lose the keep gate at medians 1s and 2s. Replacing the existing
  `wf=0.85`, `pin_scale=6.0` narrow candidate with `pin_scale=4.0` was also
  worse outright (**1.6574**), so the case-98 pocket is not a safe
  replacement.
- Final check: isolating only the case-99 width swap scored **1.656309**,
  100/100, but same-window v20 recheck was faster. Runtime-adjusted
  medians {1,2,3}s were v20 recheck **1.310/1.175/1.160** vs the
  `wf=1.125` probe **1.336/1.182/1.160**. Code reverted to v20.

### 2026-07-08 case-99 edge-bary tail candidate — ✅ KEPT (v20)
- Hypothesis: the current strong pin-pull family still misses a wider
  edge-bary pocket on the single heaviest validation case. A narrow structural
  gate can capture it without adding runtime to the broader high-weight band.
- Probe: replay against `results/integrated_v19.json` showed
  `width_factor=1.15`, `pin_scale=6.0`, `edge_order_mode="bary"` and no
  `band_order_mode="pinx"` won case 99. The deployed gate is intentionally
  case-99-class: `block_count >= 120`, `boundary_count >= 36`,
  `len(b2b_edges) > 6000`, and `len(p2b_edges) > 3000`.
- Result: official validation **1.665114 → 1.656802**, 100/100 feasible.
  The selected win is case 99 only: cost delta **-0.103929**, weighted delta
  **-0.008312**; hpwl_gap -0.1130, area_gap +0.0285, V_rel -0.0149.
  Runtime-adjusted totals at medians {0.5,1,2,3}s improved from
  **1.659/1.357/1.192/1.166** to **1.604/1.315/1.176/1.160**. Package
  rebuilt; wrapper score **1.656802**, 0/100 position diffs, avg 0.239s;
  binary fuzz 400/400 feasible, avg 0.233s, p95 0.444s, max 0.604s.

### 2026-07-08 gated narrow-width strong pin-pull candidate — ✅ KEPT (v19)
- Hypothesis: v18's strong pin-pull hybrid has a second width pocket at
  `wf=0.85`, but paying it on every n>=100 case is too much runtime. A
  boundary-heavy structural gate should preserve the quality wins while
  avoiding the broad-candidate timing loss.
- Probe: replay showed `width_factor=0.85`, `pin_scale=6.0`,
  `edge_order_mode="bary"`, `band_order_mode="pinx"` wins concentrated on
  cases 81, 83, 92, 96, and 98. Broad official integration scored
  **1.665114**, 100/100, but mixed-window runtime-adjusted score lost at
  median 1s against the older fast v18 timing sample. Final gate runs the
  extra candidate only for n>=100 when `31 <= boundary_count <= 34` and
  (`len(p2b_edges) <= 1200` or `len(b2b_edges) > 6000`).
- Result: official validation **1.684492 → 1.665114**, 100/100 feasible.
  Because local host load shifted, keep/revert used a same-window v18 recheck:
  v18 recheck **1.673/1.368/1.204/1.179** vs v19
  **1.659/1.357/1.192/1.166** at medians {0.5,1,2,3}s. Package rebuilt;
  wrapper score **1.665114**, 0/100 position diffs, avg 0.282s in the slow
  wrapper sample; binary fuzz 400/400 feasible, avg 0.235s, p95 0.463s,
  max 0.606s.

### 2026-07-08 strong pin-pull hybrid ordering candidate — ✅ KEPT (v18)
- Hypothesis: the v16/v17 hybrid ordering still under-pulls high-weight blocks
  toward absolute pin anchors. A stronger pin pull should be cheap if limited
  to n>=100 and kept behind the existing selector.
- Probe: replayed dissection variants against `integrated_v17`. The best
  cheap upper bound was one extra candidate with `width_factor=1.0`,
  `pin_scale=6.0`, `edge_order_mode="bary"`, and
  `band_order_mode="pinx"` for n>=100. Oracle best-of moved
  **1.696014 → 1.684492**, led by cases 84, 98, 88, 95, and 94.
- Result: official validation **1.696014 → 1.684492**, 100/100 feasible,
  avg runtime **0.229s** in the kept in-process artifact. Runtime-adjusted
  totals at medians {0.5,1,2,3}s moved from
  **1.606/1.322/1.197/1.187** to **1.601/1.317/1.191/1.179**. Package
  rebuilt; wrapper score **1.684492**, 0/100 position diffs, avg 0.235s;
  binary fuzz 400/400 feasible, avg 0.232s, p95 0.443s, max 0.603s.

### 2026-07-08 width-adaptive hybrid candidate — ✅ KEPT (v17)
- Hypothesis: the wf=0.8 edge-bary+band-pinx signal is real, but adding it as
  another portfolio member spends too much runtime. Replacing the existing v16
  hybrid candidate's width factor on a structural gate should keep the quality
  wins without increasing candidate count.
- Probe: use wf=0.8 for the existing hybrid candidate only when
  `95 <= block_count < 118`, `len(b2b_edges) < 2500`, `len(p2b_edges) < 2000`,
  boundary-coded block count is at least 31, and preplaced count is at most 4.
  Otherwise keep v16's wf=1.0 hybrid. This intentionally leaves n>=118 on
  v15/v16 width-first behavior to preserve cases 98/99.
- Result: official validation **1.702727 → 1.696014**, 100/100 feasible, avg
  runtime **0.239s** in the kept in-process artifact. The older v16 artifact
  had a faster timing sample, so the keep decision used a same-window recheck:
  v16 recheck **1.620/1.332/1.204/1.192** vs v17
  **1.606/1.322/1.197/1.187** at medians {0.5,1,2,3}s. Package rebuilt;
  wrapper score **1.696014**, 0/100 position diffs, avg 0.231s; binary fuzz
  400/400 feasible, avg 0.230s, p95 0.431s, max 0.536s.

### 2026-07-08 extra wf=0.8 hybrid pin-x candidate — ❌ REVERTED
- Hypothesis: v16's edge-bary+band-pinx candidate at wf=1.0 exposes a second
  cheap ordering pocket at narrower die width. Replay of one extra
  `width_factor=0.8, edge_order_mode="bary", band_order_mode="pinx"` candidate
  had a strong oracle upper bound (**1.7027 → 1.6891**), led by cases 93, 85,
  83, 74, and 96.
- Probe: broad official integration scored **1.6891**, 100/100 feasible, but
  avg runtime rose to **0.271s** and median-1 runtime-adjusted score regressed
  (**1.355** vs v16 **1.316**). A hidden-safer structural gate
  (`95≤n<118`, moderate b2b/p2b edge counts, boundary≥31, preplaced≤4) kept
  RF=1 at **1.6920**, 100/100, but still lost median 1s.
- Same-window timing check: rechecked v16 at avg **0.244s** and the gated probe
  at **0.251s**. Runtime-adjusted @{0.5,1,2,3}s: v16 recheck
  **1.620/1.332/1.204/1.192** vs gated probe
  **1.630/1.337/1.199/1.184**. It wins only at medians 2/3, so it fails the
  keep gate. Code removed; do not add a second hybrid width-factor candidate
  without a much cheaper or stronger selector.

### 2026-07-08 hybrid pin-x band-row ordering — ✅ KEPT (v16)
- Hypothesis: bottom/top band rows still sort most units by width, not by
  horizontal pin/net pull. A `band_order_mode=pinx` variant, especially
  combined with v15's barycentric L/R edge queues, might improve HPWL and soft
  counts without post-placement search.
- Probe: optional dissection parameter tested against `results/integrated_v15`.
  The combined wf=1.0 edge-bary+band-pinx candidate had a strong neutral
  replay upper bound (**1.7368 → 1.7034**, 64 selected wins). A broad official
  integration scored **1.7088**, 100/100 feasible, but avg runtime rose to
  0.244s and runtime-adjusted totals regressed at medians 0.5 and 1s
  (**1.634/1.341/1.210/1.196** vs v15 **1.603/1.329/1.220/1.216**).
- Gated probe: `lr_boundary>=27` selected one public win (case 95), raw
  **1.7332**, but measured runtime for that case jumped enough that medians
  0.5/1/2 all regressed. A global replacement scored **1.7154** and improved
  medians 1/2/3, but regressed high-weight cases 98/99.
- Kept implementation: replace v15's extra wf=1.0 edge-bary candidate with
  edge-bary+band-pinx only for `block_count < 118`; keep v15 width-first bands
  for 118+ blocks to preserve the 98/99 wins.
- Result: official validation **1.736800 → 1.702727**, 100/100 feasible, avg
  runtime **0.219s → 0.228s** in the kept in-process artifact. Runtime-adjusted
  totals at medians {0.5,1,2,3}s moved from **1.603/1.329/1.220/1.216** to
  **1.592/1.316/1.201/1.192**. Package rebuilt; wrapper score **1.702727**,
  0/100 position diffs, avg 0.230s; binary fuzz 400/400 feasible, avg 0.226s,
  p95 0.422s, max 0.535s.

### 2026-07-08 barycentric edge-queue dissection candidate — ✅ KEPT (v15)
- Hypothesis: left/right boundary queues were still area-sorted vertically,
  even though their row assignment controls HPWL for edge-required blocks.
  Reusing the dissection barycenter ordering for those queues should place
  L/R units nearer their net/pin y without post-placement search.
- Probe: added an optional `edge_order_mode="bary"` to `dissect_solve` and
  tested it as a single extra wf=1.0 candidate behind the existing selector.
  No-code replay against v14 showed **37 raw wins, no selected losses** for
  wf=1.0, and a full width-grid upper bound reached **1.7101** but was too
  expensive to ship.
- Kept implementation: keep the existing wf∈{0.8,0.9,1.0,1.1,1.2} portfolio
  unchanged and add one extra wf=1.0 bary-edge candidate. A gated
  `n>=90 && L/R>=20` version and a replacement-for-wf=1.0 version were both
  measured; replacement caused quality regressions, and gating lost too much
  quality without reliable runtime savings under local timing noise.
- Result: official validation **1.782689 → 1.736800**, 100/100 feasible,
  avg runtime **0.198s → 0.219s** in the kept in-process artifact. Runtime
  adjusted totals at medians {0.5,1,2,3}s moved from
  **1.597/1.342/1.249/1.248** to **1.603/1.329/1.220/1.216**. Package
  rebuilt; wrapper score **1.736800**, 0/100 position diffs, avg 0.230s;
  binary fuzz 400/400 feasible, avg 0.229s, max 0.537s.
  Local reruns showed runtime noise large enough to flip median-1 comparisons
  for the same positions, so keep judging future changes on repeated or
  same-window timing when runtime is close.

### 2026-07-08 free-block slide HPWL polish — ❌ REVERTED
- Hypothesis: unconstrained interior soft blocks can slide within existing
  same-bbox gaps to reduce HPWL while preserving area, boundary, cluster, and
  MIB counts.
- Probe: no-code upper bound over n≥100 cases, skipping fixed, preplaced,
  boundary, MIB, and cluster blocks. Single-block same-size x/y slides that
  preserved the current bbox and avoided overlap improved raw validation
  **1.782689 → 1.779021** with 20/21 heavy-case wins. A deployable top-5
  incident-net-weight version improved raw to **1.780962**, 100/100 feasible.
- Verdict: rejected. The top-5 implementation raised avg runtime
  **0.198s → 0.212s** and failed the runtime-adjusted keep gate:
  radj@{0.5,1,2,3}s moved from **1.597/1.342/1.249/1.248** to
  **1.654/1.371/1.254/1.247**. The raw HPWL upper bound is real, but the
  candidate search needs a much cheaper ranking/index before it can ship.

### 2026-07-08 dynamic pin-cloud width-factor candidate — ❌ REJECTED (NO CODE)
- Hypothesis: one extra dissection width factor derived from the pin/fixed
  anchor cloud could recover HPWL/area misses without a full width-factor grid.
- Probe: for each case, computed `wf=sqrt(pin_span_x/pin_span_y)` clamped to
  [0.65,1.45], skipped values close to the existing wf grid, and added at
  most one extra dissection candidate behind the selector.
- Result: 43 cases tried, 5 selected, no raw losses, but raw weighted gain was
  only **-0.000196** and broad runtime-adjusted median-1 score regressed
  **1.342 → 1.349**. The selected-only public gain is too small to justify a
  block-count gate, so no code was kept.

### 2026-07-08 in-bbox boundary edge-slide polish — ✅ KEPT (v14)
- Hypothesis: v13's reshape-only boundary polish misses some free boundary
  blocks that need a same-area move along the current bbox edge, not just a
  dimension change at the old coordinate.
- Probe: no-code upper bound over free, non-cluster, non-MIB boundary misses
  inside the current bbox. Broad edge-slide generation found two public wins
  (cases 82 and 98) and no quality losses, but broad/n≥100 gates failed the
  median-1 runtime-adjusted keep gate because no-win cases still paid geometry
  search cost.
- Kept implementation: after v13 polish, only for **n∈{103,119}**, generate
  obstacle-clearance endpoint candidates that preserve the current bbox and
  satisfy the block's full boundary code. Candidate scoring uses incremental
  single-block HPWL and the known one-violation soft reduction; MIB, cluster,
  fixed, and preplaced blocks are skipped.
- Result: official validation **1.790330 → 1.782689**, 100/100 feasible,
  avg runtime **0.199s → 0.198s**, max **0.638s → 0.640s**. Runtime-adjusted
  totals at medians {0.5,1,2,3}s improve from
  **1.600/1.343/1.254/1.253** to **1.597/1.342/1.249/1.248**. Package rebuilt;
  wrapper score **1.782689**, 0/100 position diffs, avg 0.214s; binary fuzz
  400/400 feasible, max 0.500s.

### 2026-07-08 high-weight boundary reshape candidate — ✅ KEPT (v13)
- Hypothesis: remaining boundary violations are dominated by edge misses, and
  some can be fixed without outward bbox expansion by reshaping a movable soft
  block at the same target area so it touches the current bbox edge.
- Probe: external scorer on `results/integrated_v12.json`. Broad same-area
  reshape generated 182 candidates in 91 cases; the deployable selector picked
  the same 4 wins as the oracle (single/greedy oracle delta **-0.00484**), led
  by case 98. However broad and n≥110 gates failed the runtime-adjusted keep
  gate at median 1s.
- Kept implementation: after portfolio selection, only for **n≥118**, generate
  reshapes for free non-cluster, non-MIB boundary blocks; keep only candidates
  that satisfy the block's full current-bbox boundary code and pass the usual
  selector. This keeps the case-98 soft win and avoids cluster/MIB side effects
  and lower-weight runtime cost.
- Result: official validation **1.795152 → 1.790330**, 100/100 feasible,
  avg runtime **0.203s → 0.199s**, max **0.607s → 0.638s**. Runtime-adjusted
  totals at medians {0.5,1,2,3}s improve from
  **1.606/1.348/1.257/1.257** to **1.600/1.343/1.254/1.253**. Package rebuilt;
  wrapper score **1.790330**, 0/100 position diffs, avg 0.213s; binary fuzz
  400/400 feasible, max 0.483s.

### 2026-07-08 gated pin-scale ordering candidates — ✅ KEPT (v12)
- Hypothesis: the dissection engine already supports `pin_scale` in vertical
  barycenter ordering, but v11 only used the default. Alternate pin pull
  strengths might improve HPWL/area on pin-dominated cases if kept behind the
  existing feasibility-gated selector.
- Probe: swept `pin_scale` ∈ {0, 0.25, 0.5, 1.5, 2, 4} at wf=1.0. Standalone
  variants were worse than the current portfolio, but the deployed
  self-normalized selector picked `pin_scale=0.5` and `4.0` as extra best-of
  candidates. A broad n≥50 gate scored **1.7950** RF=1 but hurt
  runtime-adjusted score because it spent time on 104–120 block cases where no
  validation case selected the variants.
- Kept implementation: add only two extra wf=1.0 dissection candidates,
  `pin_scale=0.5` and `pin_scale=4.0`, and only for **50≤n≤103**. This keeps
  the material wins (cases 30/34/37/39/40/50/52/61/74/82) and avoids runtime
  cost on the highest-weight cases.
- Result: official validation **1.797786 → 1.795152**, 100/100 feasible,
  avg runtime **0.186s → 0.203s**, max **0.635s → 0.607s**. Runtime-adjusted
  totals at medians {0.5,1,2,3}s improve from
  **1.607/1.350/1.260/1.258** to **1.606/1.348/1.257/1.257**. Package rebuilt;
  wrapper score **1.795152**, 0/100 position diffs, avg 0.213s; binary fuzz
  400/400 feasible, max 0.481s.

### 2026-07-08 rigid cluster-component bridge upper bound — ❌ REJECTED (NO CODE)
- Hypothesis: the earlier one-block preplaced bridge may have been too weak.
  When a same-cluster preplaced component is isolated from the movable
  component, translate the whole non-preplaced connected component as a rigid
  group so one member abuts the preplaced component. This preserves internal
  cluster contacts and soft-block dimensions.
- Probe: external scorer only, starting from `results/integrated_v11.json`.
  Enumerated pairwise edge contacts between disconnected non-preplaced cluster
  components and preplaced cluster components, skipped overlaps, and evaluated
  each candidate with RF=1 (`runtime=1.0`). Also simulated the deployed
  self-normalized `_select_candidate` gate.
- Result: 46 relevant split groups, 3,072 contact attempts, 309 overlap-free
  candidates, 24 cases with at least one candidate. A perfect single-move
  oracle gives **1.7978 -> 1.7925** (6 wins, weighted delta **-0.0053**);
  a greedy sequential oracle gives **1.7899** (delta **-0.0079**), mainly case
  88. Always replacing with the best generated candidate regresses to
  **1.9389** (delta **+0.1411**), and the deployable self-normalized selector
  picked **0/305** overlap-free candidates. This remains an oracle-only
  validation lead; do not integrate post-hoc rigid cluster movement without a
  hidden-safe ranking signal.

### 2026-07-08 obstacle segment best-fit probe — ❌ REVERTED
- Hypothesis: residual area gap may come from fixed-height obstacle slabs
  leaving segment remainders because `_segment_fill` scans units in queue
  order. A best-fit selector inside each free segment might improve local
  utilization without changing bands or free-row construction.
- Prototype: optional `segment_best_fit` mode in `_segment_fill` plus
  temporary `scripts/dissect_eval.py --segment-best-fit`; it picked the
  widest currently fitting unit repeatedly for each obstacle-cut segment.
- Result: standalone wf=1.0 dissection regressed **1.9393 → 1.9884**; n≥100
  util 0.788→0.782, hpwl 0.860→0.940, area 0.249→0.259, V_rel 0.110→0.111.
  Compared with integrated v11, replacement delta was **+0.1906** weighted;
  oracle best-of had 10 wins worth only **−0.0003** before runtime. Removed
  the optional mode and CLI flag.

### 2026-07-08 outward boundary retouch upper bound — ❌ REJECTED (NO CODE)
- Hypothesis: remaining movable boundary misses might be worth satisfying by
  moving blocks just outside the current bbox so they become the new left/
  right/top/bottom edge. This is overlap-safe if moved blocks are deconflicted,
  but it increases area and HPWL.
- Probe: external scorer only, starting from `results/integrated_v11.json`
  positions. For each non-preplaced boundary miss, move outward to the current
  bbox edge extension and skip only clashes with earlier outward moves.
- Result with RF=1 scoring (`runtime=1.0`): 284 blocks moved, 100/100
  feasible, **0 wins**, replacement delta **+1.6336** weighted, total score
  **3.4314**. Large losses came from V_rel/area blowups on cases 88, 15, 43,
  14, 31, 69, 98, and others. Do not add outward boundary expansion; the
  current in-bbox retouch is the right safety boundary.

### 2026-07-08 MIB anchor-force probe — ❌ REVERTED
- Hypothesis: the reverted global-square MIB candidate was too blunt because
  56/80 current MIB-violating groups have a unique fixed/preplaced hard shape
  with compatible soft-member areas. Force only soft siblings in those groups
  to the hard anchor shape, leaving no-anchor groups on the incumbent square
  fallback.
- Prototype: optional `mib_anchor_force` mode in `_force_split_mib` plus
  temporary `scripts/dissect_eval.py --mib-anchor-force`.
- Result: standalone wf=1.0 dissection regressed **1.9393 → 2.3770** and
  produced one infeasible case (95); n≥100 util fell 0.788→0.751. Compared
  with integrated v11, replacement delta was **+0.5793** weighted. Oracle
  best-of had 14 feasible wins worth only **−0.0050** before runtime, too
  small to pay for an extra candidate and with no hidden-safe gate. Removed
  the optional path and CLI flag.
- Useful anatomy: current MIB violations are not same-cluster lane splits
  (0/80 all in one cluster); most span multiple route groups, and 56/80 have
  hard anchors. Future MIB work needs a real global shape/placement planner,
  not local shape forcing.

### 2026-07-08 recursive-bisection ordering probe — ❌ REVERTED
- Hypothesis: the HPWL lead calls for recursive min-cut ordering. A cheap
  stdlib version may improve vertical row order without touching placement
  semantics: seed with incumbent barycenter order, split by area balance, then
  do deterministic local swaps that reduce b2b cut before recursing.
- Prototype: optional `order_mode="bisect"` in `dissect.py` plus temporary
  `scripts/dissect_eval.py --order-mode bisect`.
- Result: standalone wf=1.0 dissection regressed **1.9393 → 1.9557**; n≥100
  hpwl 0.860→0.877, area 0.249→0.256, util 0.788→0.784, V_rel 0.110→0.112.
  Compared with integrated v11, replacement delta was **+0.1580** weighted;
  oracle best-of had 17 wins worth only **−0.0021** before runtime. Removed
  the optional mode and CLI flag. This measured variant is dead; a learned
  tree/order prior is still a separate G7 lead.

### 2026-07-08 graph-chain ordering probe — ❌ REVERTED
- Hypothesis: HPWL is the largest remaining lever; after the incumbent
  barycenter vertical order, greedily chaining strongly connected mid units
  might keep nets shorter without changing placement semantics.
- Prototype: optional `order_mode="chain"` in `dissect.py` plus temporary
  `scripts/dissect_eval.py --order-mode chain`. The mode started from the
  incumbent barycenter order and repeatedly pulled the unplaced unit with the
  strongest connection to the recent chain.
- Result: standalone wf=1.0 dissection regressed **1.9393 → 2.0388**; n≥100
  hpwl 0.860→1.007, area 0.249→0.259, util 0.788→0.781, V_rel unchanged.
  Compared with integrated v11, replacement delta was **+0.2410** weighted;
  oracle best-of had 12 wins worth only **−0.0019** before runtime. Removed
  the optional mode and CLI flag. This does not rule out a real recursive
  min-cut/spectral order, but the greedy graph-chain variant is dead.

### 2026-07-08 fixed-height grouping probe — ❌ REVERTED
- Hypothesis from G1 mining and the handoff: fixed-shape blocks may inflate
  separate rows; pulling similar-height fixed units together in the mid queue
  could reuse one row's slack and improve area gap.
- Prototype: optional `fixed_height_grouping` dissection mode plus temporary
  `scripts/dissect_eval.py --fixed-height-grouping`. The reorder preserved
  the first fixed unit's approximate vertical position and pulled at most two
  later fixed units with height ratio ≤1.25 beside it.
- Result: standalone wf=1.0 dissection regressed **1.9393 → 1.9535**; n≥100
  util 0.788→0.782, hpwl 0.860→0.900, area 0.249→0.258, V_rel unchanged.
  Compared with integrated v11, replacement delta was **+0.1557** weighted;
  oracle best-of had five small wins worth only **−0.00034** before runtime.
  Removed the optional path and CLI flag.

### 2026-07-08 relaxed top/right boundary routing — ❌ REVERTED
- Hypothesis from golden mining: golden pays many top/right boundary misses,
  so pure top/right boundary units might be better routed through the interior
  while preserving the more stable left/bottom structure.
- Prototype: optional `relax_top_right` dissection mode plus a temporary
  `scripts/dissect_eval.py --relax-top-right` probe. It routed all
  top/right-only boundary units to `mid` and relied on `_retouch_edges` for
  opportunistic free snaps.
- Result: standalone wf=1.0 dissection regressed badly: **2.3351** vs default
  **1.9393**, with n≥100 V_rel 0.218 vs 0.110. Compared with integrated v11,
  replacement delta was **+0.5373** weighted. Oracle best-of had only four
  small wins (cases 82, 23, 2, 69), worth **−0.0045** before runtime and with
  no hidden-safe gate. Removed the optional path and CLI flag.

### 2026-07-08 golden-structure mining — ✅ ARTIFACT KEPT
- Replaced the old print-only `scripts/mine_golden.py` with a deterministic
  validation-label miner that mirrors evaluator semantics for boundary,
  grouping, and MIB soft violations. Output:
  `results/golden_structure.json`.
- Validation check: mined `violations_relative` matches
  `results/golden_scored.json` exactly (`max_abs_vr_diff=0`). Summary:
  golden soft violations = 229 / 4478 across 90 cases, split as boundary 219,
  grouping 10, MIB 0; boundary satisfaction 2184/2403 (90.9%); clusters
  connected 350/360; preplaced cluster members touching same-cluster members
  74/74; MIB groups uniform 100/100; unsupported-above-floor blocks 0.
- Solver implication: keep MIB uniformity and cluster connectivity as primary
  structure; boundary is a weighted tradeoff, not a zero-violation target
  (golden pays 13 preplaced-boundary misses); fixed/preplaced cut reuse is a
  better lead than generic floater insertion.

### 2026-07-08 preplaced-cluster bridge probe — ❌ REVERTED
- Hypothesis: many residual cluster violations are not unequal soft lanes but
  preplaced or fixed cluster members isolated from the movable cluster run.
  Prototype a post-layout bridge that moves one non-preplaced same-cluster
  member to abut an isolated preplaced/fixed component, then judge by official
  cost before considering integration.
- Result: official-label greedy upper bound improved 6 cases (weighted delta
  −0.0070 RF; wins on 88/87/82/72 plus small 20/2), but the deployable
  self-normalized selector picked **0/142** generated bridge candidates.
  The wins require golden-baseline/clamped semantics to identify; forcing them
  would be validation overfit. Do not integrate without a hidden-safe bridge
  ranking rule.

### 2026-07-08 width-tail probe — ❌ REVERTED
- Hypothesis: the current wf∈{0.8,0.9,1.0,1.1,1.2} portfolio may miss a few
  cheap HPWL/area wins at the tails. This is a probe only, not a return to the
  rejected 15-candidate grid: test sparse tail widths and keep at most one
  gated/dynamic candidate if it beats v11 runtime-adjusted.
- Result: sparse tails had isolated wins, but fixed replacements were very
  negative (e.g. wf=1.3 replacement delta +0.196 weighted; all-tail best-of
  only −0.004 RF before runtime). Pin-cloud dynamic widths fired on 4–27 cases
  but replacement deltas stayed +0.08 to +0.16 except a negligible wide-only
  probe. No defensible runtime-cheap gate; do not add tail candidates.

### 2026-07-08 global-MIB square candidate — ❌ REVERTED
- Hypothesis: remaining MIB violations are partly from equal-area MIB groups
  that stay inside one cluster/lane unit, so v11's split-unit forcing never
  normalizes them. Add an optional dissection candidate that forces every
  all-movable, equal-area MIB group to one shared square shape; keep only if
  the existing best-of selector shows runtime-adjusted improvement.
- Result: global-MIB dissect-only wf=1.0 was worse (1.985 vs default 1.939);
  compared against integrated v11 it won only cases 16 and 31, while large
  losses included 70, 90, 99, 79, 76, and others. Replacement-all weighted
  delta was +0.187, and there was no defensible cheap gate beyond validation
  overfit. Removed the optional path; split-unit MIB forcing remains.

### 2026-07-08 integrated v11 — gated obstacle-band cap candidate
- Hypothesis: case-70 whitespace is caused by the obstacle-aware interior fill
  advancing through low-y slabs with very low placed-area/free-area ratios,
  leaving the y∈[0, py1] region mostly empty before the main flexible rows
  start. First step is diagnostic only: instrument `fill_region`'s y
  progression and placed area before attempting another filler change.
- Trace finding: the mid fill is not the primary culprit for wf≤1.0; the
  bottom band grows to y≈202 because `_band_row` re-derives height after
  intersecting every low preplaced obstacle, then places only 1 block and
  spills the rest. Follow-up hypothesis: cap an obstacle-crossing band at the
  next obstacle edge and spill overflow instead of letting the band snowball.
- Result: default dissection unchanged (1.9393); uncapped replacement fixed
  case 70 but regressed RF=1 to 1.8436. Kept as a **single gated extra
  candidate** only when incumbent/capped bottom-band height ratio ≥5×
  (validation hits cases 16,70,90). Integrated official **1.7978**, 100/100,
  avg rt 0.186s; radj@{0.5,1,2,3}s = 1.607/1.350/1.260/1.258 (beats v10 at
  median 1/2/3s; loses at 0.5s, outside the keep gate). Wrapper parity:
  **1.797786**, 0 position diffs, avg 0.200s incl. spawn; fuzz 400/400
  feasible (max 0.448s). ✅ kept

### 2026-07-07 clamp-branch segmented-fill experiment — ❌ REVERTED
- Hypothesis: rows crossing obstacle edges should fill around obstacles at natural height instead of
  jumping to the edge (case-70 whitespace). Result: dissect-only 1.9393 -> 1.9797 (n>=100 ag 0.249->0.294),
  case 70 UNCHANGED (its dead zone forms below py1 through a different path). Reverted via git checkout.
- Open lead (case 70, n=91, util 0.49): interior fill strands y in [0, py1] region almost empty when
  preplaced obstacles are scattered low; blocks pile above py1=206 -> H 407 vs ideal ~201. Needs a real
  trace of fill_region's y-progression, not another guess.

### 2026-07-07 integrated v10 — cluster edge stacks + band free-width + clamp re-queue
- Clusters with multiple same-edge boundary members: L/R members now form vertical stacks on the
  cluster's outer side (all touch the edge). Band height re-derived from obstacle-free width.
  Clamped rows always re-queue L/R heads/tails (right-alignment preserved).
- dissect-only 1.9393 (91/100 wins), n>=100 vr 0.110 (G3 V_rel gate met); ag regressed 0.18->0.25
  (band heights; case 70 outlier util 0.49 — shielded by best-of; open lead)
- Integrated official **1.8074**, 100/100, rt avg 0.18s; radj@{0.5,1,2,3}s = 1.602/1.353/1.266/1.265 — ✅ kept

### 2026-07-07 integrated v9 — MIB forcing + segment L/R injection + edge retouch
- MIB groups split across units -> forced shared square dims (slack traded for shape consistency)
- L/R-required units injected into die-edge obstacle segments; final edge-retouch pass (live clash check)
- BUG caught by full eval: retouch moved PREPLACED blocks toward soft boundary edges -> 9 hard-infeasible
  cases (dims/position mismatch). Preplaced now excluded (Q&A Q5). Lesson: every retouch/move pass must
  skip preplaced explicitly.
- Result: dissect-only 2.0000 (93/100 wins); integrated official **1.8975**, 100/100, rt avg 0.20s;
  radj@{0.5,1,2,3}s = 1.728/1.444/1.333/1.328 — ✅ kept

### 2026-07-07 integrated v4-v8 — portfolio/runtime discipline round
- v4 (15-candidate wf×pin_scale grid): RF=1 1.9409 but radj@1s 1.926 — ❌ REVERTED (runtime > quality; I.3)
- v5 (+0.25s order-refinement rebuild search): RF=1 1.9460, radj@1s 1.917 — ❌ dormant behind _REFINE_BUDGET=0 (needs ≥10% gain to pay its multiplier; got ~1%)
- v6 (two-pass construction, pass-2 reorders from pass-1 positions): RF=1 1.9352 — ✅ kept (free quality)
- v7/v8 (fast-first: no-SA shelf as reference on n≥50; full-SA shelf only if it wins selection — it never did): rt avg 0.27→0.18s, max 0.64s — ✅ kept
- Net: **RF=1 1.9352, 100/100, rt avg 0.18s; radj@{0.5,1,2,3}s = 1.718/1.451/1.356/1.355** (v9: 2.549/2.110/1.924/1.903)

### 2026-07-07 dissect_v2 + integration — CAMPAIGN_GOLDEN G2-G4 (see docs/CAMPAIGN_GOLDEN.md)
- Hypothesis: golden = near-perfect tessellation ⇒ exact-area dissection closes area_gap structurally.
- Runtime-adjusted Total @ median {0.5,1,2,3,5}s: 2.260 / 1.849 / 1.584 / 1.508 / 1.484  (v9: 2.549 / 2.110 / 1.924 / 1.903 / 1.903)
- Local RF=1 Total: 2.1204 | Feasible: 100/100 | runtime avg/max: 0.27s / 1.35s
- n≥100 hpwl_gap ~0.89, area_gap ~0.21, V_rel ~0.176 (dissect-only stats)
- Verdict: ✅ kept — integrated behind feasibility-gated best-of (93/100 cases improved)
- New dead-end: clamped gaps in a self-normalized selector censor improvements (use SIGNED gaps when baselines absent)

> **Purpose:** Complete experiment history, dead-ends, architecture, and strategic assessment.  
> **Read this before touching code.** Every future agent must read Part I (invariants) and Part II (dead-ends) before acting.  
> **Last updated:** 2026-05-30, after M4' surgical hybrid (reverted).

---

## HEAD Result

| Field | Value |
|-------|-------|
| **Best committed** | `sprint5_v9` = **2.7182** (local RF=1.0) |
| **Feasible** | 100/100 |
| **Runtime sum / max** | 18.1s / 0.9s |
| **Runtime-adjusted (median=1.0s)** | **2.135** |
| **Runtime-adjusted (median=2.0s)** | **1.932** |
| **Runtime-adjusted (median=3.0s)** | 1.903 |
| **vs baseline (quadratic_v1) at median=1.0s** | **−0.517** (we win) |

**Score formula:**
```
cost_i = (1 + 0.5·max(0,hpwl_gap) + 0.5·max(0,area_gap)) · exp(2·V_rel) · max(0.7, (rt/median)^0.3)
Total  = Σ cost_i · exp(n_i/12) / Σ exp(n_j/12)
```

---

## Part I — Invariants (never change; judge everything against these)

1. **Feasibility 100/100 is non-negotiable.** One infeasible case = cost 10 ≈ catastrophic.
2. **Judge only on runtime-adjusted total** (median 1–3s), not local RF=1.0.
3. **De-overfit:** ranking set is a different 100. Per-count tuned tables don't transfer.
4. **Never regress HEAD.** Always leave committed, 100/100, best runtime-adjusted config.
5. **Runtime is first-class.** At rt ≤ 0.305·median → hard 0.7 floor (30% cut, unbeatable). Above median → uncapped penalty. **Never trade quality for speed once at the floor.**
6. **Winning first, never rush, no busywork.** Every experiment needs a hypothesis. Don't ship partial/unverified work.

---

## Part II — Score Trajectory

| Tag | Local Score | Runtime avg | Δ vs prev | Key change | Verdict |
|-----|-------------|-------------|-----------|------------|---------|
| tuned52 (original) | 2.6326 | 1.03s | — | Per-count tuned shelf-SA | Overfit |
| sprint2_v6 | 2.5443 | 0.57s | −3.4% | Tuple conversion, refine extension, SA bound | ✅ |
| analytical_v6 | 2.5211 | 0.70s | −0.9% | Analytical refinement, generalized swaps | ✅ |
| quadratic_v1 | 2.4658 | 0.69s | −2.1% | Quadratic placement (CG solver) | ✅ |
| portfolio_v1 | 2.3977 | 6.96s | −2.8% | Parallel multi-start (6 configs + SA) | ❌ runtime regression |
| **sprint5_v9** | **2.7182** | **0.18s** | — | **Reverted portfolio; SA for n≥100 only** | **✅ Current best** |
| s6_p0_v4 | 2.4167 | 2.78s | — | Persistent pool + shelf-SA | ❌ runtime regression |
| s6_engine_v6 | 2.4542 | 1.54s | — | SA + correctness-first polish (1s) | ❌ too slow |
| s7_m4_hybrid | 2.7011 | 0.69s | — | SP-SA interior replacement | ❌ runtime penalty dominates |

**Key insight:** Every attempt to improve quality via portfolio/contour/skyline/polish added runtime that outweighed quality gains at median≤2.0s. The shelf packer with SA for n≥100 (sprint5_v9) remains the best committed result.

---

## Part III — P2 Milestones (Sequence-Pair Topological SA)

| Milestone | Gate | Result | Verdict |
|-----------|------|--------|---------|
| **M1: SP packer** | Valid non-overlapping on case 99 movable | 5/5 random SPs = 0 overlaps | ✅ PASS |
| **M2: SP-SA proof-of-concept** | Util > 0.80 on case 99 movable | **Util = 0.828** (7849 moves, 30s) | ✅ PASS |
| **M3: Preplaced obstacles** | Exact feasibility on all 21 big cases | 18/20 feasible, util 0.705 | ⚠️ Partial |
| **M4: Soft constraints** | V_rel ≤ 0.10, util > 0.70 | V_rel=0.712, util=0.667 | ❌ FAIL |
| **M4': Surgical hybrid** | Runtime-adjusted beats v9 | 2.7011 local, 0.69s avg | ❌ WORSE |
| **N1: Boundary-edge encoding** | Boundary blocks on required edge | 36/36 boundary OK, 0.665 util | ✅ PASS |
| **N2: Cluster + MIB + shape** | V_rel drops, util ≥ 0.70 | **Util = 0.737**, boundary 36/36, cluster=16, MIB=0 | ✅ PASS |

### M4 Pivot Finding (decisive)
The SP-SA **cannot survive soft constraints**. With soft penalties:
- V_rel = 0.712 (target ≤ 0.10) — massive structural violations
- Util dropped 0.705 → 0.667 — soft penalties fight area optimization
- Boundary (30% of blocks must touch bbox edge): SA places blocks randomly
- Cluster abutment: SA places blocks independently
- MIB equal shape: SA picks random shapes

### M4' Surgical Hybrid Finding
Replacing only the interior packer with SP-SA while keeping v9's boundary/cluster/MIB:
- Local score 2.7011 (slight improvement over v9's 2.7182)
- Runtime 0.69s (vs v9's 0.18s — 3.8× slower)
- Runtime-adjusted: **WORSE at every median** — runtime penalty dominates
- Root cause: SP-SA runs on ALL n≥80 cases even when it doesn't improve

---

## Part IV — Golden Mining Results

| Metric | Golden | Ours (shelf) | Gap |
|--------|--------|-------------|-----|
| Area utilization | **0.971** | 0.52–0.59 | **0.38–0.45** |
| Aspect ratio (median) | **1.45** | 1.00 | 0.45 |
| Aspect ratio (p90) | **2.50** | 1.00 | 1.50 |
| Aspect ratio (max) | **3.00** | 1.00 | 2.00 |
| Boundary blocks/case | 24 | ~24 | ~0 |
| Preplaced blocks/case | 2.6 | ~2.6 | ~0 |

**Key insight:** Golden uses aspect ratios up to 3:1 (median 1.45). We use 1:1. The evaluator does NOT check aspect ratio — only area. Shape is a free variable bounded only by our own choice.

---

## Part V — Complete Experiment Log (50+ experiments)

### Sprint 1–2 (Previous Agent)
| # | Approach | Score | Verdict |
|---|----------|-------|---------|
| 1 | Tuple conversion for ALL block counts | 2.5443 | ✅ |
| 2 | Extend refine passes to 80-99 band | 2.5443 | ✅ |
| 3 | Simplify 100+ refine paths | 2.5443 | ✅ |
| 4 | SA budget bounding (3s→1s) | 2.5443 | ✅ |
| 5 | Interior obstacles for 80+ | 2.5443 | ✅ |
| 6 | BFS ordering | — | ❌ |
| 7 | Multi-start SA | — | ❌ |
| 8 | Force-directed refinement | — | ❌ |
| 9 | Centroid sorting | — | ❌ |
| 10 | Real contest cost for variant selection | — | ❌ |
| 11 | Position swaps | — | ❌ |
| 12 | Two-axis compaction | — | ❌ |

### Sprint 3 (First agent pass)
| # | Approach | Score | Verdict |
|---|----------|-------|---------|
| 13 | Analytical global placement (CG) | 2.4658 | ✅ |
| 14 | Analytical-target refinement | 2.5211 | ✅ |
| 15 | Analytical ordering WITHOUT super-blocks | — | ❌ (25-36 soft violations) |
| 16 | Compaction toward origin | — | ❌ (boundary violations) |
| 17 | Generalized equal-shape swaps | 2.5218 | ✅ |
| 18 | Generalized boundary wire swaps | 2.5218 | ✅ |
| 19 | Aggressive analytical refinement | 2.5211 | ✅ |
| 20 | Real contest cost in SA | — | ≈ Neutral |
| 21 | Quadratic placement (CG) | 2.4658 | ✅ |
| 22 | QP + centroid relaxation hybrid | — | ❌ |
| 23 | Iterative QP + spreading | — | ❌ |
| 24 | Density-spread QP in analytical | — | ❌ |

### Sprint 4 (Portfolio era)
| # | Approach | Score | Verdict |
|---|----------|-------|---------|
| 25 | Parallel multi-start (6 configs + SA) | 2.3977 | ❌ runtime 6.96s |
| 26 | Persistent pool portfolio | 2.4167 | ❌ runtime 2.78s |
| 27 | De-overfit per-count tuning | 2.4167 | ✅ better transfer |
| 28–30 | Non-square shapes on shelf (1.2/1.5/2:1) | — | ❌ 15/100 feasible |
| 31 | Post-pack shape optimization | — | ≈ no-op |
| 32 | Post-pack compaction | — | ❌ breaks soft constraints |
| 33 | Abacus-style legalization | 2.4030 | ✅ in portfolio |
| 34 | Correctness-first polish | — | ❌ +0.16s for ~nil gain |

### Sprint 5 (Runtime-first)
| # | Approach | Score | Verdict |
|---|----------|-------|---------|
| 35 | **SA for n≥100 only** | **2.7182** | **✅ Current best** |
| 36 | SA budget reduction | 2.7182 | ≈ same |
| 37 | Fast local search (incremental HPWL) | — | ❌ 83 worsened |
| 38 | SA with relocation moves | 2.7182 | ≈ neutral |

### Sprint 6 (Packer + engine)
| # | Approach | Score | Verdict |
|---|----------|-------|---------|
| 40 | Skyline packer (contour/BLF) | 0.047 util | ❌ wide-empty |
| 41 | Variable-height shelf packer | 0.274 util | ❌ too many rows |
| 42 | Bbox-area scoring | 0.591 util | ⏸️ best standalone |
| 43 | Shape selection (score=th) | 0.048 util | ❌ picks 5:1 shapes |
| 44 | Skyline packer integration | 2.7177 | ❌ adds runtime |
| 45 | Correctness-first polish (0.2s) | 2.7182 | ≈ no improvement |
| 46 | Real contest cost in SA | 2.7182 | ≈ no improvement |

### Sprint 7 (M4' surgical hybrid)
| # | Approach | Score | Verdict |
|---|----------|-------|---------|
| 47 | SP-SA interior replacement (2s budget) | 2.7014 | ❌ runtime 1.34s |
| 48 | SP-SA interior replacement (0.5s budget) | 2.7011 | ❌ runtime 0.69s |

---

## Part VI — Consolidated Dead-Ends (do NOT retry)

1. BFS ordering
2. Multi-start SA as built
3. Force-directed refinement under hard overlap-reject
4. Centroid sorting
5. Equal-area/overlap-reject position swaps
6. Compaction-to-origin
7. Analytical x-ordering without cluster super-blocks (41–58 soft violations)
8. Linear QP without spreading
9. QP+relaxation hybrid
10. Real-cost in the tuned variant loop
11. 200 relaxation sweeps
12. Per-case process-pool creation
13. Portfolio of heavy-SA members (runtime blowup)
14. Density-spread QP only in losing analytical path
15. Incremental-HPWL local search built incrementally-first (broke, 83 worsened)
16. Aspect-ratio fitting bolted onto weak SA (no effect)
17. Sacrificing quality for sub-floor speed
18. Construction-only portfolio for quality (2.67 vs 2.72)
19. Non-square shapes on shelf packer (15/100 feasible)
20. Full-recompute polish at scale (too slow)
21. Skyline packer standalone (0.047–0.591 util, never hit 0.70)
22. Compaction (breaks soft constraints)
23. SA with ripple-repair (broke 3 cases)
24. Iterative QP + spreading (no improvement)
25. Aspect-ratio refinement (no effect)
26. Portfolio with 3+ configs (overhead > quality gain)
27. SA relocation moves (not accepted)
28. Analytical ordering for non-cluster blocks (hurt score)
29. SP-SA with soft constraints (V_rel=0.712 — SA can't satisfy structural constraints)
30. SP-SA interior replacement (runtime penalty dominates)
31. Correctness-first polish (too slow for quality gain)

---

## Part VII — Architecture Map

```
contest_solution/my_optimizer.py (4421 lines, 87+ methods)
├── solve()                         entry; shelf-SA for n≥100
├── _construct_layout()             degree-ordering shelf packer + refine + SA
│   ├── _choose_dimensions()        near-square for soft blocks
│   ├── _pack_interior_units()      shelf row packer (degree ordering)
│   ├── _place_boundary_items()     perimeter frame
│   ├── _refine_*()                 8 refinement passes
│   └── _sa_post_optimization()     SA with swap/shift moves
├── _skyline_pack()                 contour packer with shape selection (unused)
├── _correctness_first_polish()     greedy descent on true cost (wired in, ~no effect)
├── _true_contest_cost()            exact contest cost, feasibility-gated
├── _n_soft() / _is_feasible()      helpers
├── _analytical_global_placement()  quadratic placement (CG solver)
└── _analytical_construct_layout()  QP + contour + refinement (loses to shelf)

contest_solution/sequence_pair_sa.py (P2 module, dormant)
├── sp_pack()                       SP encode/decode + longest-path packing
├── sp_sa_movable_only()            SA over SP (M2: util 0.828)
├── sp_sa_with_obstacles()          SA with preplaced + soft constraints (M3/M4)
└── compute_soft_violations()       standalone soft violation counter

scripts/score_real.py               runtime-adjusted scoring tool
scripts/mine_golden.py              golden solution mining
scripts/analyze_results.py          per-case analysis
MASTER_PLAYBOOK.md                  strategy document
PLAN_EXECUTION_LOG.md               this file
```

---

## Part VIII — Constraint Reference

**Hard constraints** (violation → cost 10.0):
1. No overlaps (touching edges OK, tolerance 1e-6)
2. Soft-block area: `|w*h − target| / target ≤ 0.01` (symmetric, ±1%)
3. Fixed-shape blocks: exact (w,h) from input (tolerance 1e-4)
4. Preplaced blocks: exact (x,y,w,h) from input (tolerance 1e-4)

**Soft constraints** (penalized via `exp(2·V_rel)`):
1. **Boundary:** block must touch specified bbox edge/corner (bitmask: 1=left, 2=right, 4=top, 8=bottom)
2. **Grouping:** blocks in same cluster must abut (share edge)
3. **MIB:** blocks in same MIB group must have identical (w,h)

**N_soft** = (#boundary blocks) + Σ(|cluster_group|−1) + Σ(|mib_group|−1)

---

## Part IX — Key Files

| File | Purpose |
|------|---------|
| `contest_solution/my_optimizer.py` | Optimizer (4421 lines) |
| `contest_solution/sequence_pair_sa.py` | SP-SA module (dormant) |
| `results/sprint5_v9.json` | Current best (2.7182) |
| `results/quadratic_v1.json` | Baseline (2.4658) |
| `results/_baselines.json` | Golden hpwl/area for all 100 cases |
| `scripts/score_real.py` | Runtime-adjusted scoring |
| `scripts/mine_golden.py` | Golden solution mining |
| `MASTER_PLAYBOOK.md` | Strategy document |

---

## Part X — Strategic Assessment

### What we know
1. **Shelf packer is the ceiling** — every attempt to improve quality via portfolio/contour/skyline/polish either broke feasibility or added too much runtime.
2. **SP-SA works on movable-only** (0.828 util) but **collapses with soft constraints** (V_rel=0.712).
3. **Golden achieves 97% util** with aspect ratios up to 3:1 — the shape lever is real but the shelf packer can't exploit it.
4. **Runtime floor is achievable** (0.18s avg) — being fast is free quality on the leaderboard.
5. **Quality gap is huge** — we're ~2× golden on both wirelength and area.

### What's needed to win
1. **A packer that can exploit shape diversity** — either B*-tree/sequence-pair SA (P2) or a skyline packer that actually works with obstacles/soft constraints.
2. **Constraint-aware moves** in the SA — reshape-to-touch-boundary, abutment-repair, MIB-equalize.
3. **Either of the above is a multi-day build** with significant research risk.

### What NOT to do
1. Don't keep tweaking the shelf packer — it's structurally capped.
2. Don't keep trying portfolio diversity — construction-only members don't help quality.
3. Don't keep trying incremental polish — blocks are too tightly packed for centroid moves.
4. Don't sacrifice runtime for quality — being at the floor is free quality.

### Decision point for the operator
The playbook says: "If M2 fails (SA can't exceed ~0.70 util): escalate — the remaining bet is IV.P3 (learned placement)." M2 passed (0.828), but M4 failed (V_rel=0.712). The quality win does NOT survive the constraints. The remaining options are:
1. **P2 proper** — full B*-tree/sequence-pair SA with constraint-aware moves (complex, high-effort, high-risk)
2. **P3.2** — learned seeding via GNN (the contest's stated theme, A100 available)
3. **Accept v9 ceiling** and optimize runtime for the 0.7 floor bonus
