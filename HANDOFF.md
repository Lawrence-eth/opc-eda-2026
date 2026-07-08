# HANDOFF — ICCAD 2026 Problem C (FloorSet Challenge)

> For the next agent taking over this project. Everything here was measured on
> 2026-07-07 at commit `91ceb64` (+ this handoff commit). Nothing is aspirational.
> Operator's standing orders: **the only goal is to WIN. No time limit. Never
> regress the verified state.**

---

## 1. Current verified state (the numbers that matter)

| Metric | Value | Artifact |
|---|---|---|
| Official validation score (RF=1) | **1.8074** | `results/integrated_v10.json` |
| Feasible | **100 / 100** | same |
| Runtime | avg 0.18s/case, max 0.68s (in-process) | same |
| Runtime-adjusted @ median {0.5, 1, 2, 3}s | 1.602 / **1.353** / 1.266 / 1.265 | recompute per §5.3 |
| Packaged binary via official command | **1.807413, 0/100 position diffs**, avg 0.198s incl. spawn | `results/wrapper_v10.json` |
| Binary fuzz (400 training instances, wrapper protocol) | 400/400 hard-feasible | rerun per §5.5 |
| Tests / audit / release gate | 51/51 PASS / PASS / PASS | `pytest`, §5.6 |

Reference points: the pre-campaign optimizer (v9) scored 2.7182 (radj@1s 2.110).
Golden-equivalent play = 1.108 RF=1 / 0.776 at the runtime floor (the golden
layouts themselves violate soft constraints on 90/100 cases —
`results/golden_scored.json`). Absolute bound 0.70. **Distance travelled:
2.718 → 1.807; distance remaining to golden-equivalent: 1.807 → 1.108.**

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
3. **Selection**: `_select_candidate` — feasibility-gated exact-cost shape.
   With golden baselines absent (deployment) it normalizes against the first
   candidate and uses **SIGNED gaps** (clamping against a non-golden reference
   censors improvements — this bug cost a day; don't reintroduce it).
4. If the fast shelf won (rare — 9/100 cases), the **full-SA shelf** is run
   and the selection redone. Dissection currently wins 91/100.
5. Dormant, behind flags: SP-SA (`_ENABLE_SP_SA=False`), order-refinement
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
  + pin-x. **Two passes**: pass 2 re-orders using pass 1's actual positions.
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
    print(m, radj('results/<candidate>.json', m), radj('results/integrated_v10.json', m))
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
`results/integrated_v10.json`, max-score 1.81 — bump the threshold when you
beat it).

## 6. Open leads, ranked (with the evidence)

Current dissect-only decomposition (n≥100): **hg 0.860, ag 0.249, vr 0.110,
util 0.788**. Violation ledger across 100 cases: boundary ~458 (L 175 /
R 181 / B 57 / T 32 — a chunk are preplaced-with-boundary-codes, which are
UNFIXABLE by rule and golden pays them too), cluster 58, MIB 117.

1. **Case-70-class whitespace (ag lever, ~0.05–0.1 total).** n=91: util 0.49,
   H=407 vs ideal ~201 — the y∈[0, py1] region ends up almost empty when
   preplaced obstacles are scattered low; blocks pile above py1. A
   segmented-natural-height-row fix was tried and REVERTED (made ag worse
   overall, case 70 unchanged — see log). **Trace `fill_region`'s y-progression
   on case 70 before coding anything.** Bands y[0,50)=27%, [50,102)=19%,
   [102,153)=0.2%, [153,204)=9%.
2. **hg 0.86 (the biggest single lever, worth ~0.3 at the floor if halved).**
   Ideas not yet tried: recursive min-cut bisection ordering (FM-style, maps
   naturally to the row structure; cheap at n≤120); better within-row second
   pass; die-aspect selection by pin-cloud shape rather than fixed wf grid.
   REMEMBER: must be runtime-free (construction is ~10–30ms; keep it there).
3. **ag 0.249 residuals**: band heights from the free-width iteration can
   overshoot; fixed blocks are not grouped by height into shared rows
   (each tall fixed block inflates its row); obstacle-segment remainders.
4. **MIB 117**: forced-square unification only triggers for groups split
   across units with equal areas and no fixed members; the in-cluster lane
   grouping covers same-lane members only. A global "MIB shape planner"
   (pick one (w,h) per group, place all members as rigid leaves) would cut
   most of the rest — at a small slack cost (same trade as the current
   forcing, measured net-positive).
5. **Cluster 58**: mostly two-lane clusters whose lanes disconnect when a
   lane is a single wide block stacked against a multi-block lane of unequal
   width — verify with the ledger script in §5 of the log, then consider
   equal-width lane enforcement.
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
everything in `MASTER_PLAYBOOK.md` Part II.6.

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
| `results/integrated_v10.json` | CURRENT official result (1.8074) |
| `results/wrapper_v10.json` | packaged-binary parity run (1.807413) |
| `results/v9_locked.json` | pre-campaign locked result (2.7182) |
| `results/golden_scored.json` | golden layouts scored officially (the target) |
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
