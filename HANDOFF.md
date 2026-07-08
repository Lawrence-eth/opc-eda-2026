# HANDOFF — ICCAD 2026 Problem C (FloorSet Challenge)

> For the next agent taking over this project. Current head was measured on
> 2026-07-08 after integrated v17. Nothing is aspirational.
> Operator's standing orders: **the only goal is to WIN. No time limit. Never
> regress the verified state.**

---

## 1. Current verified state (the numbers that matter)

| Metric | Value | Artifact |
|---|---|---|
| Official validation score (RF=1) | **1.6960** | `results/integrated_v17.json` |
| Feasible | **100 / 100** | same |
| Runtime | avg 0.239s/case, max 0.746s (in-process) | same |
| Runtime-adjusted @ median {0.5, 1, 2, 3}s | 1.606 / **1.322** / 1.197 / 1.187 | recompute per §5.3 |
| Packaged binary via official command | **1.696014, 0/100 position diffs**, avg 0.231s incl. spawn | `results/wrapper_v17.json` |
| Binary fuzz (400 training instances, wrapper protocol) | 400/400 hard-feasible, avg 0.230s, p95 0.431s, max 0.536s | rerun per §5.5 |
| Tests / audit / release gate | 51/51 PASS / PASS / PASS | `pytest`, §5.6 |

Reference points: the pre-campaign optimizer (v9) scored 2.7182 (radj@1s 2.110).
Golden-equivalent play = 1.108 RF=1 / 0.776 at the runtime floor (the golden
layouts themselves violate soft constraints on 90/100 cases —
`results/golden_scored.json`). Absolute bound 0.70. **Distance travelled:
2.718 → 1.696; distance remaining to golden-equivalent: 1.696 → 1.108.**

## 2. Read these, in order

1. `CLAUDE.md` (repo root) — hard rules + environment.
2. This file.
3. `docs/CAMPAIGN_GOLDEN.md` — the campaign: evidence, milestones G1–G7 with
   measured gates, open leads.
4. `PLAN_EXECUTION_LOG.md` — every experiment with verdicts, **including the
   reverted ones** (top of file = most recent).
5. `SUBMISSION_PLAN.md` — organizer submission format + package + rebuild gate.
6. `MASTER_PLAYBOOK.md` — historical strategy; **Part II.6 dead-end list is
   still binding** (do not retry those approaches).

## 3. What the solver is now (architecture)

`contest_solution/my_optimizer.py::MyOptimizer.solve()` does, per case:

1. **Fast shelf** (v9's constructive pipeline with `no_sa=True` for n≥50) —
   the reference/fallback candidate, ~0.03–0.1s.
2. **Dissection portfolio**: `contest_solution/dissect.py::dissect_solve()`
   for width_factor ∈ {0.8, 0.9, 1.0, 1.1, 1.2} (~0.01–0.03s each).
3. **Gated band-cap candidate** (v11): one extra `band_edge_cap=True`
   dissection at wf=1.0 only when `should_try_band_edge_cap()` sees a ≥5×
   incumbent/capped bottom-band height ratio. This targets the case-70/90
   obstacle snowball without paying a candidate on every case.
4. **Gated pin-scale candidates** (v12): two extra wf=1.0 dissection
   candidates with `pin_scale` 0.5 and 4.0 only for 50≤n≤103. They are
   HPWL/area ordering variants and are kept behind the same selector; the
   upper block-count gate avoids spending runtime on 104–120 where validation
   selected none and runtime dominates.
5. **Width-adaptive hybrid edge-ordering candidate** (v17): one extra dissection
   candidate orders left/right boundary queues by the barycenter signal
   instead of area. For n<118 it also sorts bottom/top band units by pin/net-x
   pull; for n>=118 it preserves v15 width-first band ordering to keep the
   measured case-98/99 wins. v17 uses wf=0.8 instead of wf=1.0 for
   high-boundary, moderate-net 95..117 block cases. It is kept behind the same
   selector and captures HPWL/soft wins without post-placement search.
6. **High-weight boundary reshape candidate** (v13): after portfolio
   selection, only for n≥118, try same-area reshapes of free non-cluster,
   non-MIB boundary blocks that satisfy their full current-bbox boundary code.
   This is also selector-gated; measured validation win is case 98.
7. **Boundary edge-slide polish** (v14): only for n∈{103,119}, try same-area
   in-bbox placements for free non-cluster, non-MIB boundary misses using
   obstacle-clearance endpoints. It is cost-gated and keeps the current bbox;
   measured wins are validation cases 82 and 98.
8. **Selection**: `_select_candidate` — feasibility-gated exact-cost shape.
   With golden baselines absent (deployment) it normalizes against the first
   candidate and uses **SIGNED gaps** (clamping against a non-golden reference
   censors improvements — this bug cost a day; don't reintroduce it).
9. If the fast shelf won, the **full-SA shelf** is run and the selection
   redone. The dissection family wins most cases; shelf remains the fallback.
10. Dormant, behind flags: SP-SA (`_ENABLE_SP_SA=False`), order-refinement
   local search (`_REFINE_BUDGET=0.0` — see §6 for why).

`dissect.py` — the campaign engine (exact-area dissection):
- Layout = full-width **exact-fill rows**: row height derives from membership
  (`h = soft_area / (W − fixed_w)`), soft dims `w = area/h` ⇒ areas exact to
  float ulp, overlap-free by construction.
- **Frame**: one-row bottom band (bottom-required blocks, all touch y=0),
  one-row top band (flush via `_retouch_top`), left/right-required units
  injected at row ends (`l_queue`/`r_queue`, vertical **stacks** when queues
  are long, right-aligned tails).
- **Obstacles** (preplaced): pinned; die width matched to obstacle-forced
  height (`W = max(min(√A, A/max(√A, py1)) · wf, px1)`); slabs crossed by
  obstacles get segmented fixed-height fills.
- **Clusters** = contiguous runs (flat single-lane in bands, two stacked
  lanes elsewhere; members requiring L/R edges become vertical stacks on the
  cluster's outer side). Cluster abutment is structural — this was THE
  blocker for every pre-campaign approach.
- **MIB**: same-row identical slots when the group is one unit; groups split
  across units get **forced shared square dims** (slack traded for shape
  consistency).
- **Ordering** (HPWL): vertical = barycenter iteration over b2b + absolute
  pin-y anchors; horizontal = within-row sort by pull toward placed neighbors
  + pin-x. The v17 hybrid candidate can also sort bottom/top band units by
  pin/net-x pull and choose wf=0.8 for one structural high-weight gate.
  **Two passes**: pass 2 re-orders using pass 1's actual
  positions.
- Final `_retouch_edges`: flush boundary-coded blocks to the bbox edge when
  the destination is empty — **never moves preplaced** (hard constraint,
  Q&A Q5; violating this cost 9 infeasible cases before it was caught).

## 4. The scoring model you are optimizing (memorize)

```
cost = (1 + 0.5·(max(0,hpwl_gap) + max(0,area_gap))) · exp(2·V_rel) · max(0.7, (rt/median)^0.3)
     = 10.0 if ANY hard violation (overlap>1e-6, soft area off by >1%,
             fixed dims off by >1e-4, preplaced x,y,w,h off by >1e-4)
Total = Σ cost_i · exp(n_i/12) / Σ exp(n_j/12)      (n≥100 ≈ 79% of weight)
```
- Gaps are vs GOLDEN baselines (hidden set: computed server-side), clamped ≥0.
- Runtime is measured around `solve()` per case — **for the packaged binary
  that includes process spawn + imports + JSON I/O** (organizers call the
  executable through their `op_wrapper.py`, one spawn per case, 60s timeout).
- **Runtime discipline (verified twice this session):** near the floor, a
  +0.25s budget needs ≥10% quality gain on the affected cases to break even.
  Two experiments (15-candidate grid; 0.25s refinement) improved RF=1 and
  LOST runtime-adjusted. Judge every change at median ∈ {1,2,3}s (§5.3).
- Grouping abutment check has **no float tolerance** (shapely union) — shared
  edges must be exact coordinates. The exact-fill construction guarantees
  this; keep it that way.

## 5. Workflows (copy-paste)

Environment bootstrap (fresh clone; ~5 min):
```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install numpy shapely matplotlib tqdm requests pytest pyinstaller
mkdir -p external && git clone https://github.com/IntelLabs/FloorSet.git external/FloorSet
# validation data (~15MB) auto-downloads on first evaluator run
# training data (1M samples, needed only for fuzz/mining):
#   curl -L -o external/FloorSet/LiteTensorData_v2.tar.gz \
#     "https://huggingface.co/datasets/IntelLabs/FloorSet/resolve/main/LiteTensorData_v2.tar.gz"
#   tar xzf external/FloorSet/LiteTensorData_v2.tar.gz -C external/FloorSet
```

5.1 **Official evaluation** (the number that counts):
```bash
cp contest_solution/my_optimizer.py contest_solution/dissect.py \
   contest_solution/sequence_pair_sa.py external/FloorSet/iccad2026contest/
cd external/FloorSet/iccad2026contest
PYTHONPATH=.. ../../../.venv/bin/python iccad2026_evaluate.py \
    --evaluate my_optimizer.py --output ../../../results/<tag>.json
```

5.2 **Fast engine-only iteration** (~90s for all 100, prints per-case worst):
```bash
.venv/bin/python scripts/dissect_eval.py --cases all        # or --cases 95,97,99
```

5.3 **Runtime-adjusted comparison** (the KEEP/REVERT criterion):
```bash
.venv/bin/python - <<'EOF'
import json, math
def radj(path, med):
    d=json.load(open(path))['test_results']; Z=tot=0.0
    for r in d:
        w=math.exp(r['block_count']/12); Z+=w
        base=(1+0.5*(max(0,r['hpwl_gap'])+max(0,r['area_gap'])))*math.exp(2*r['violations_relative'])
        c=10.0 if not r['is_feasible'] else min(base*max(0.7,(r['runtime_seconds']/med)**0.3),10-1e-6)
        tot+=c*w
    return tot/Z
for m in (0.5,1,2,3):
    print(m, radj('results/<candidate>.json', m), radj('results/integrated_v17.json', m))
EOF
```
KEEP iff it beats the incumbent at median ∈ {1,2,3}s AND stays 100/100.

5.4 **Package rebuild + parity gate** (MANDATORY before any resubmission):
```bash
bash packaging/build_submission.sh
cd external/FloorSet/iccad2026contest && rm -rf dist
cp -r ../../../submission/dist . && cp ../../../packaging/op_wrapper.py .
PYTHONPATH=.. ../../../.venv/bin/python iccad2026_evaluate.py \
    --evaluate op_wrapper.py --output ../../../results/wrapper_check.json
# REQUIRED: same total as the in-process run and 0 position diffs
```

5.5 **Binary fuzz** (hidden-set insurance): `.venv/bin/python scripts/fuzz_binary.py --num 400` → must be 0 failures.

5.6 **Repo gates**: `.venv/bin/python -m pytest -q` (51 tests) and
`.venv/bin/python scripts/check_public_release.py` (defaults now point at
`results/integrated_v17.json`, max-score 1.70 — bump the threshold when you
beat it).

## 6. Open leads, ranked (with the evidence)

Current integrated v17 decomposition (n≥100): **hg 0.645, ag 0.162,
vr 0.093**. Score-weighted heavy-case averages: **hg 0.645, ag 0.162,
vr 0.093**. Violation ledger across 100 cases: boundary 332, grouping 56,
MIB 123. A chunk of boundary violations are preplaced-with-boundary-codes,
which are UNFIXABLE by rule and golden pays them too.

Golden-structure mining is now measured in `results/golden_structure.json`
(`scripts/mine_golden.py`; exact `violations_relative` match to
`results/golden_scored.json`). Golden itself has boundary 219, grouping 10,
MIB 0; clusters are connected on 350/360 groups; MIB groups are 100/100
uniform; every preplaced cluster member touches a same-cluster member; and no
validation golden block is unsupported above the floor. Implication: keep MIB
and cluster structure strict, but don't force boundary to zero at the expense
of area/HPWL, and don't chase generic "floater" insertion as a primary lead.

1. **Residual area-gap / obstacle-band generalization.** Case 70 was traced
   and mitigated in v11: the bottom band grew to y≈202 before mid fill,
   placing one block and spilling the rest; the kept fix is a single gated
   `band_edge_cap` candidate when incumbent/capped bottom-band height ratio
   is ≥5× (validation hits 16/70/90). Do not make capped bands the default:
   that regressed integrated score to 1.8436. Next useful work is a broader
   but still runtime-cheap predictor or a segmented band layout that improves
   the remaining ag residuals without firing on the large-case regressions
   (88/98/99 were red flags in the broad attempt).
2. **hg 0.65 (still the biggest single lever).** Recursive bisection and
   graph-chain ordering were already rejected; do not retry them. Remaining
   ordering ideas need to be construction-cheap: better within-row second pass,
   safer edge/interior co-ordering, or a learned/golden-derived row assignment.
   REMEMBER: must be runtime-free (construction is ~10–30ms; keep it there).
3. **ag 0.16 residuals**: band heights from the free-width iteration can
   overshoot; fixed blocks are not grouped by height into shared rows
   (each tall fixed block inflates its row); obstacle-segment remainders.
4. **MIB 123**: golden has 0 MIB violations, so equality is not optional.
   The reverted global-square candidate proves that a blunt extra candidate is
   worse; the useful version is a hidden-safe shape planner that selects one
   (w,h) per group without exploding area or width on large cases.
5. **Grouping 56**: golden connects 350/360 groups and every preplaced cluster
   member touches its cluster. The reverted preplaced-bridge probe found real
   upper-bound wins but no deployable selector; a stronger rigid-component
   bridge oracle also improved only -0.0079 weighted and the deployable
   selector again picked 0 candidates. Next cluster work should be a ranking
   signal/equal-width lane rule, not forced post-hoc movement.
6. **G7 — learn the ordering from the 1M golden trees** (`tree_sol` is a
   B*-tree: side 0 = right-adjacent, side 1 = stacked-above; semantics
   verified). A model that predicts the vertical order / row assignment
   would replace the barycenter heuristic; decode stays exact so
   feasibility is untouched. Needs GPU (this VM is 2-core CPU).
7. **If the field median proves slow** (leaderboard signal): re-enable
   `_REFINE_BUDGET` and/or the full-SA shelf unconditionally — there is
   headroom to spend runtime if everyone else is slow.

**Do NOT retry** (measured dead ends this session): candidate-grid blowups
(pin_scale × wf), flat +0.25s refinement budgets at median≤3s, clamped gaps in
the selector, any retouch/move pass that can touch preplaced blocks, and
the blunt relaxed top/right boundary-routing candidate, naive fixed-height
grouping in the mid queue, greedy graph-chain mid ordering, the area-balanced
recursive bisection mid-ordering probe, local MIB anchor-shape forcing, and
outward boundary expansion/retouch, obstacle-segment best-fit unit selection,
post-hoc rigid cluster-component bridge movement without a hidden-safe ranker,
and broad/global pin-x band ordering without the v16 high-weight preservation,
and everything in `MASTER_PLAYBOOK.md` Part II.6.

## 7. Submission logistics (operator-owned)

- The uploadable artifact is `submission/iccad2026_submission.tar.gz`
  (rebuilt + parity-verified for the current engine). Contents: PyInstaller
  `--onedir` executable (21MB, torch-free, crash-proof fallback), organizers'
  `op_wrapper.py` verbatim, README, stdlib-only source fallback.
- Eval host: Debian 13, Python 3.13.14, torch 2.12.0+cu130, 48-core Icelake,
  A100 80GB, 128GB RAM, no internet, cases sequential
  (`docs/extracted/C_Submission_Guidelines_20260616.txt`, `C_QA_20260618.txt`).
- Optional belt-and-suspenders: rebuild the binary in a Debian 13 container
  (glibc 2.39→2.41 is forward-compatible, so this is low-risk polish).
- Upload channel + credentials: only the operator has these
  (iccad-contest.org). Monitor the FloorSet GitHub issues for rule updates.

## 8. Repo map

| Path | What |
|---|---|
| `contest_solution/my_optimizer.py` | THE solver (shelf + dissection portfolio + selector) |
| `contest_solution/dissect.py` | the campaign engine (exact-area dissection) |
| `contest_solution/sequence_pair_sa.py` | dormant SP-SA (historical) |
| `packaging/` | executable package sources + build script + organizers' wrapper |
| `scripts/dissect_eval.py` | fast engine-only eval vs v9 per case |
| `scripts/fuzz_binary.py` | wrapper-protocol feasibility fuzz on training data |
| `scripts/match_validation_in_training.py` | retrieval scan (result: 0/1,008,000 — don't redo) |
| `scripts/check_public_release.py` | release gate (audit + docs scan) |
| `results/integrated_v17.json` | CURRENT official result (1.6960) |
| `results/wrapper_v17.json` | packaged-binary parity run (1.696014) |
| `results/integrated_v16.json` | previous official result (1.7027) |
| `results/wrapper_v16.json` | previous packaged-binary parity run (1.702727) |
| `results/integrated_v15.json` | previous official result (1.7368) |
| `results/wrapper_v15.json` | previous packaged-binary parity run (1.736800) |
| `results/integrated_v14.json` | previous official result (1.7827) |
| `results/wrapper_v14.json` | previous packaged-binary parity run (1.782689) |
| `results/integrated_v13.json` | previous official result (1.7903) |
| `results/wrapper_v13.json` | previous packaged-binary parity run (1.790330) |
| `results/integrated_v12.json` | previous official result (1.7952) |
| `results/integrated_v11.json` | previous official result (1.7978) |
| `results/integrated_v10.json` | previous official result (1.8074) |
| `results/wrapper_v10.json` | previous packaged-binary parity run (1.807413) |
| `results/v9_locked.json` | pre-campaign locked result (2.7182) |
| `results/golden_scored.json` | golden layouts scored officially (the target) |
| `results/golden_structure.json` | mined golden structure priors + scorer cross-check |
| `results/retrieval_scan.json` | proof hidden set can't be looked up |
| `docs/CAMPAIGN_GOLDEN.md` | campaign plan + measured milestones |
| `docs/extracted/` | problem v10 + Q&A 06-18 + submission guidelines (text) |
| known cruft | `dissect.py` has unused `_fill_rows`/`_fill_column` (pre-rewrite leftovers); `n6_poc.py`, `_skyline_*` are historical |

## 9. The invariant chain that keeps this safe

Every change flows: edit → `dissect_eval` (fast signal) → official eval →
radj comparison at {1,2,3}s → pytest → commit with verdict in
`PLAN_EXECUTION_LOG.md` → (if the engine changed) package rebuild + parity
gate + fuzz → push. The selector's feasibility gate plus the per-case best-of
means a broken candidate can never ship — but only if you keep running the
full 100 (single-case checks missed both real bugs this session; the full
eval caught them).
