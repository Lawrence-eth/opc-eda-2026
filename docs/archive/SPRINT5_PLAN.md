# FloorSet ICCAD-2026 — Sprint 5 Plan (the plan to win)

**Author:** planning agent (Opus 4.8)
**For:** executing agent — you can run 24/7; compute/tokens are NOT a constraint. **Contest inference runtime IS a hard scoring constraint (see §1).**
**Read first:** this file end-to-end, then `PLAN_EXECUTION_LOG.md`, then skim `my_optimizer.py` around the line numbers cited here.

---

## 0. Standing & the one-paragraph thesis

- **Local best:** `portfolio_v1` = **2.3977** (RuntimeFactor forced to 1.0 locally). Committed HEAD = de-overfitted `portfolio_v7` = **2.4167**.
- **The trap:** Sprint 4's portfolio bought −2.8% local quality but pushed runtime to **6.96s avg / 14.7s max** on n≥100 (baseline was 3.2s max). The real leaderboard multiplies every case's cost by `max(0.7, (your_rt/median)^0.3)`, **uncapped above**. So Sprint 4 is **probably a leaderboard regression**, not a win (see §1 math).
- **Thesis:** The objective is **quality AND speed, jointly.** The blocks are few (21–120) but every case carries all five constraint types and 21 big cases have preplaced obstacles. The path to winning is: (A) measure the *real* (runtime-adjusted) score and make the solver **fast**, (B) replace the slow/weak position-SA with a **fast incremental, legalization-aware local search** that optimizes the *exact* contest cost (soft terms included), (C) give it good starting points via **density-spread global placement** (not the collapsing linear QP). D/E/F are higher-effort bets layered on top. Quality gap is huge (we're ~2.1× golden wirelength, ~1.8× golden area), so there is a lot of headroom.

---

## 1. The reframed objective — runtime is first-class (do the math, internalize it)

Contest cost per case (`iccad2026_evaluate.py:306`, `ALPHA=0.5 BETA=2.0 GAMMA=0.3`):
```
cost = (1 + 0.5*(max(0,hpwl_gap) + max(0,area_gap))) * exp(2*V_rel) * max(0.7, (rt/median)^0.3)
Total = Σ cost_i * exp(n_i/12) / Σ exp(n_j/12)        # n≥100 ≈ 79% of total
```
Runtime multiplier `max(0.7, (rt/median)^0.3)`:

| rt/median | 0.10 | 0.30 | 0.50 | 1.0 | 2.0 | 4.0 | 7.0 | 15 |
|-----------|------|------|------|-----|-----|-----|-----|----|
| multiplier| 0.70 | 0.70 | 0.81 | 1.00| 1.23| 1.52| 1.79| 2.25|

**Key facts:**
- Being **≤0.30× the median** earns the hard **0.7 floor** — a flat **30% cost reduction on every case**, equivalent to an enormous quality gain, and impossible to beat. This is a primary target.
- Being slow is **uncapped**: 7× median ⇒ 1.79× cost. The current 14.7s likely sits well above median.
- Locally `median=1.0` and the evaluator hard-codes `rt/median=1.0` (`evaluate_solution(..., median_runtime=1.0)`), so **the local score hides all of this.** You must simulate it (Thrust A).

**Design consequence:** prefer **many fast attempts in parallel (best-of)** over **one deep serial anneal**. Target per-case wall-clock **≤ ~1.0–1.5s** (chasing the 0.7 floor), never exceeding ~2–3s. Use the 48 cores for breadth, and make each attempt cheap.

---

## 2. What Sprint 4 built, and what is limiting it now (with line numbers)

Current `contest_solution/my_optimizer.py` (3741 lines):
- `solve()` :68 → builds a portfolio (`_build_portfolio` :153, only **~9 configs**), runs them via `ProcessPoolExecutor` (`_worker_solve` :26) on `min(cfgs, cores, 32)` workers, picks min `_true_contest_cost` (:2056, already exact & correct).
- Paths: `_construct_layout` (shelf) :197, `_analytical_construct_layout` :2749, `_abacus_construct_layout` :2874. Global placement (linear QP/CG) `_analytical_global_placement` :3484. Legalizers incl. `_analytical_legalize` :3632.
- **The optimizer bottleneck — `_sa_post_optimization` :2521.** Three defects:
  1. **Slow:** recomputes **full** HPWL over all b2b+p2b edges (up to ~7000) **every single move** (:2618-2619) and full bbox area too. This caps moves/sec brutally and is most of the per-case runtime.
  2. **Weak moves:** `swap` only between near-equal-area blocks (:2589), and **every move rejects on overlap** (:2598-2613) instead of re-legalizing — so blocks essentially can't move. `shift`/`relocate` likewise reject on overlap.
  3. **Soft cost is frozen:** `soft_factor` is computed once at start (:2557) and reused for every candidate (:2623). The SA is blind to changes in boundary/grouping/MIB violations — it only chases hpwl+area under a constant soft multiplier.
- **Underused parallelism:** ~9 configs on 48 cores; wall-clock gated by the single slowest member (each member runs its own ~3.6s SA + heavy construction).

These four facts are the highest-leverage targets and define Thrusts A–C.

---

## 3. Constraint reality (verified across all 100 cases — shapes every method choice)

**Every one of the 100 cases has fixed + preplaced + MIB + cluster + boundary constraints.** Big-case (n≥100) profile:
- Fixed-shape blocks: 7–17 per case (fixed w,h; movable position).
- **Preplaced blocks: 1–9 per case, present in all 21 big cases** (exact x,y,w,h — hard obstacles, never move them).
- MIB groups: 1 (members must share identical (w,h)).
- Cluster (grouping) groups: 3–4 (members must abut/share an edge).
- Boundary blocks: 26–37 (~30% of all blocks must touch a bbox edge/corner) — this forces a **perimeter ring** structure.
- b2b edges: 661–7056 (some cases are densely connected → HPWL dominated by a few hubs).

**Implications:**
- Preplaced-in-every-case **rules out a naïve all-movable floorplanner** (sequence-pair/B*-tree assume compactable modules). Any topological SA (Thrust D) must treat preplaced as fixed obstacles → significant complexity. Prefer methods that handle fixed obstacles natively: **force-directed/analytical placement (fixed nodes) + obstacle-aware legalization** (Thrust C).
- ~30% boundary blocks ⇒ the optimizer must keep a perimeter frame; don't break it. V_rel is already low (~0.10) so soft is near floor — **do not chase soft; protect it while improving hpwl/area.**
- MIB shape lock and cluster abutment are cheap to maintain if moves are cluster/group-aware.

---

## 4. Priority & dependency order (work the queue top-down)

```
A. True-metric harness + runtime re-architecture     [BLOCKING — do first]
B. Fast incremental, legalization-aware local search  [CORE engine — biggest ROI]
C. Density-spread global placement (kill QP collapse)  [feeds B better starts]
D. Sequence-pair / B*-tree SA w/ fixed obstacles       [STRETCH — area breakthrough]
E. Data-driven: mine golden + learned seeding          [STRETCH — fast inference fits §1]
F. Meta-optimization + de-overfit (cross-validated)    [continuous, after B lands]
```
A and B are mandatory and likely move the real score the most. C is a strong multiplier on B. D/E are the high-ceiling bets to chase ≤1.8–1.5. F runs continuously once B is stable.

---

## THRUST A — True-metric harness + runtime re-architecture  *(do first)*

### A1. Runtime-adjusted scoring tool (so you optimize the real metric)
Write `scripts/score_real.py` that reads a result JSON (which already stores per-case `runtime`, `hpwl_gap`, `area_gap`, `violations_relative`, `block_count`) and recomputes the weighted total under a **sweep of assumed medians** and under a **self-relative** model:
- For medians `m ∈ {0.5, 1, 2, 3, 5, 8}` s: `cost_i = (1+0.5*(hg+ag))*exp(2*vr)*max(0.7,(rt_i/m)^0.3)`, then weighted total.
- Also report a **"median = our own per-n median"** scenario (a crude proxy for "everyone runs like us").
- Print, side by side, the same table for the **baseline** (`results/quadratic_v1.json`, 3.2s max) and the current best. **This reveals whether Sprint 4 is actually ahead once runtime counts.** Make this the scoreboard for every subsequent change.

Acceptance for the whole sprint becomes: **improve the runtime-adjusted total at median∈{1,2,3}s without losing feasibility**, not just the local RF=1.0 number.

### A2. Re-architecture of `solve()` for speed + breadth
Goal: per-case wall-clock ≤ ~1.5s on n=120 while keeping/raising quality.
1. **Two-stage portfolio:** Stage 1 = run **K cheap, diverse constructors (NO heavy SA), K≈min(48, cores)** in parallel (shelf/analytical/abacus × row_factors × seeds). Each returns positions + true cost fast (construction+light refine only, target ≤0.3s each). Stage 2 = take the **top-1 (or top-2)** by true cost and run **one** bounded polish (Thrust B engine) with a hard wall-clock budget. Total wall-clock ≈ slowest stage-1 member + polish budget.
2. **Hard per-case time budget** passed down from `solve()`: e.g. `budget = min(1.5, 0.4 + n*0.01)`s, split between stage-1 cap and polish. Every inner loop checks a shared deadline.
3. **Stop redundant SA in every member.** Only the stage-2 winner gets polished.
4. **Process pool reuse:** create the `ProcessPoolExecutor` **once** (module-level, lazily) and reuse across all 100 cases — avoid per-case fork/import overhead. Use `fork` (Linux default) so workers inherit imported torch. Guard with a serial fallback.
5. Re-measure with A1 after each change. Expect a large runtime drop at ~equal quality → likely an immediate real-score win via the runtime multiplier.

**Acceptance:** runtime-adjusted total (A1, median 1–3s) strictly better than baseline AND current; feasibility 100/100; max single-case wall-clock ≤ ~2s.

---

## THRUST B — Fast, incremental, legalization-aware local search  *(core engine; biggest ROI)*

Replace the defects in `_sa_post_optimization` (:2521). This is the single highest-value code change. Build it as a new method (keep the old one until the new one wins) and make it the stage-2 polish.

### B1. Incremental cost evaluation (the speed unlock)
Maintain running state so each move is O(Δ), not O(E):
- **HPWL:** keep, per net (b2b edge / p2b edge), the contribution; on moving block `i`, only re-sum the edges incident to `i` (precompute adjacency lists once — they already exist in several refine methods, e.g. around `_refine_toward_analytical` :3200). For b2b HPWL = Σ w·(|Δcx|+|Δcy|), moving one block updates only its incident edges. This turns each move from O(7000) into O(deg(i)) (typically ≪100).
- **Bbox area:** track `x_min,x_max,y_min,y_max` and the count of blocks at each extreme; a move updates extremes in O(1) amortized (full recompute only when an extreme block leaves its extreme).
- **Soft violations, incrementally and CORRECTLY (fixes the frozen-soft defect):**
  - boundary: a block's boundary-satisfaction depends only on its own edge vs current bbox extremes → O(1) per move (recheck affected blocks when an extreme changes).
  - cluster grouping: maintain per-cluster connected-component count; recompute only the touched cluster's components (small groups, O(group²) but groups are tiny).
  - MIB: maintain the set of distinct shapes per MIB group; O(1) on a reshape move.
  - Recompute `V_rel = soft/n_soft` and use the **true** `exp(2*V_rel)` in the acceptance test — never a frozen factor.
- Acceptance objective = the **exact** `_true_contest_cost` formula, evaluated incrementally.

### B2. Legalization-aware moves (the quality unlock)
Stop rejecting on overlap; instead **repair**. Moves (all preserving hard constraints: never move preplaced/fixed dims; keep soft-block area within ±1% on reshape):
1. **Relocate-and-repair:** pick block `i`, choose a target near its connectivity centroid (or a random gap), place it, and **push overlapping neighbors** out of the way along the minimal-displacement axis (a local ripple, bounded to a few blocks), or pull `i` to the nearest legal slot (contour/skyline query). Accept by Δcost.
2. **Block / cluster swap with re-legalization:** swap two blocks (or two clusters) of *different* sizes and re-legalize the local neighborhood (not global) so it stays overlap-free. Removes the equal-area-only restriction.
3. **Row/column shift:** shift an entire shelf row or a vertical slab inward to shrink the bbox (area gap), then re-touch boundary blocks to the new extremes.
4. **Aspect-ratio reshape:** for a soft block (or a whole MIB group together), re-pick (w,h) within ±1% area to (a) close a contour gap (area), (b) let it touch its boundary edge (turn a soft violation off), or (c) abut a cluster mate. Re-legalize locally.
5. **Cluster compaction:** translate a whole cluster toward its connectivity centroid as a rigid unit (keeps abutment), repairing overlaps.
- Use a **greedy + SA hybrid:** first a few greedy sweeps (accept only improving moves) for fast descent, then short SA with the incremental objective for escaping local minima. With B1, you can do 10⁴–10⁵ moves in <1s for n=120.

### B3. Parallel polish
Run several independent B-engine chains (different seeds / move-weightings) in parallel on the stage-2 winner(s); keep the best by true cost. Cheap given B1.

**Acceptance:** runtime-adjusted total improves vs Thrust A result; feasibility 100/100. Track per-band hpwl_gap/area_gap in `analyze_results.py --top 20` — expect the n≥100 hpwl and area gaps to drop.

---

## THRUST C — Density-spread global placement (kill the QP collapse)  *(multiplier on B)*

The current `_analytical_global_placement` (:3484) solves a **linear** quadratic system → with few fixed anchors it **collapses connected blocks toward each other** (minimizes squared wirelength but creates massive overlap that legalization must undo, destroying quality). This is exactly why "Quadratic placement (CG)" only bought −0.13% (see log). Fix it with **spreading**:

### C1. Force-directed / look-ahead spreading (SimPL/Kraftwerk-lite)
Iterate:
1. Solve the QP (reuse the CG solver at :3484) → overlapping global positions.
2. **Look-ahead legalize** to a coarse grid: bin the area, find over-full bins, and compute target "spread" positions that relieve density (or compute Kraftwerk move-forces from a density map).
3. Add **pseudo-nets** anchoring each block to its spread target with a small, increasing weight; re-solve QP.
4. Repeat 3–6 iterations (cheap for ≤120 blocks). Treat preplaced/fixed as **fixed nodes** (they contribute to density and anchor wirelength but never move) — this is why analytical handles obstacles cleanly.
This yields spread, wirelength-aware global positions that legalize with **small** displacement → much better hpwl and area than collapsed QP.

### C2. Obstacle-aware legalization
Legalize the spread positions with an Abacus/Tetris row legalizer (extend `_abacus`/`_analytical_legalize`) that (a) treats preplaced blocks as fixed obstacles to place around, (b) keeps clusters contiguous via super-blocks (the fix the log flagged: analytical x-order without super-blocks → 41–58 soft violations), (c) snaps boundary blocks to the perimeter. Feed the result into the Thrust B polish.

Add C as new portfolio members (selected by true cost → can't regress). **Acceptance:** improves runtime-adjusted total; feasibility 100/100; n≥100 hpwl_gap drops vs Thrust B alone.

---

## THRUST D — Sequence-pair / B*-tree SA with fixed obstacles  *(stretch; targets area gap)*

The gold-standard floorplanner for tens–hundreds of blocks. Directly minimizes area (and wirelength) by exploring all packings. **Complication: preplaced obstacles** (every case has them). Plan:

### D1. Representation & packer
- Implement **sequence-pair** (Γ⁺,Γ⁻) over movable blocks; derive x/y by longest-path on the horizontal/vertical constraint graphs (O(n log n) with the LCS/`y`-array method). Soft blocks: discretize a few aspect ratios per block in the move set (each within ±1% area), or do shape-curve optimization on the slicing.
- **Fixed obstacles:** add preplaced blocks as fixed-coordinate nodes; augment both constraint graphs with edges that force movable blocks to lie left/right/above/below each obstacle consistent with the sequence-pair, and post-shift so movable packing never overlaps the obstacle rectangles. (Reference: sequence-pair / B*-tree floorplanning with pre-placed modules.) Validate feasibility hard.
- **Speed:** pure-Python packing per move is too slow for 10⁴+ moves. Implement the packer inner loop in **numba** (`@njit`) or a small C extension. With unlimited dev time this is worth it. Parallel-temper across the 48 cores (independent chains at different temperatures, periodic best-exchange).

### D2. Objective & schedule
- Objective = exact contest cost (area + wirelength + soft via the incremental counters from B1). Adaptive cooling; restart from the Thrust C global placement (decode positions → an initial sequence-pair) rather than random.
- Hard wall-clock budget per §1 (e.g. ≤1.5s/case via parallel short chains; quality from breadth + good seed, not long single runs).

**Gate:** only pursue if A–C plateau. Prototype on the ~79 cases **without**… no — all cases have preplaced, so prototype the obstacle handling early on one big case (e.g. test 99) and verify exact feasibility before scaling. **Acceptance:** beats the C+B pipeline on the runtime-adjusted total for n≥100.

---

## THRUST E — Data-driven (the contest's theme; aligns with the speed incentive)  *(stretch)*

The FloorSet **Lite dataset is present locally** (`external/FloorSet/LiteTensorDataTest`, 256MB; loaders `lite_dataset.py`, `litetestLoader.py`; `training_example.py`). The contest is literally "Data-Driven SoC Floorplanning," and a learned model is **fast at inference** (one forward pass ≈ ms) — which directly wins the §1 runtime multiplier.

### E1. Mine the golden solutions (low-effort, high-information — do this early, even before D)
Write `scripts/mine_golden.py` to load the golden layouts (the baselines are derived from them, `_extract_baseline` :806) and measure: aspect-ratio distributions of soft blocks, area utilization (golden bbox vs Σarea), perimeter-ring structure, how clusters are shaped/placed, typical wirelength per net. Feed findings into B/C heuristics (e.g. preferred aspect ratios, target packing density). This is pure upside and informs every other thrust.

### E2. Learned seeding (higher effort)
Train (offline, unlimited compute) a **GNN** on (constraints + connectivity) → per-block (position, aspect-ratio) or → an ordering/sequence-pair, using golden solutions as supervision (and/or the contest cost as RL reward — the evaluator exposes a reward signal, see `iccad2026_evaluate.py:1155`). At inference: one forward pass → seed → fast Thrust B/C legalize+polish. Keep model small for fast CPU/GPU inference (A100 available). **Risk:** generalization to the hidden set; train with heavy augmentation and validate on a held-out split of the 100. Treat as a parallel research track that can run continuously; only ship if it beats the analytic pipeline on held-out cases.

---

## THRUST F — Meta-optimization & de-overfitting (continuous, after B is stable)

- **Overfitting is a real risk:** the hidden ranking set is a *different* 100 in the same 21–120 range. The `_layout_variants` tuned table and any per-case tuning won't transfer. Sprint 4 already de-overfit (2.4029→2.4167) — keep that discipline.
- **Cross-validated tuning:** split the 100 cases into folds; tune **count-agnostic** hyperparameters (move weights, SA schedule, spreading iterations, portfolio composition, time budgets) on training folds, measure on held-out folds, keep only settings that generalize. Use the unlimited compute for large sweeps, but **select on held-out folds**, never on the full 100.
- **Robustness to unknown median runtime:** since we can't see competitors' runtimes, optimize for the regime where being fast is safe (target the 0.7 floor); verify the chosen config is Pareto-good across median∈{1,2,3,5}s in A1.

---

## Validation protocol (every change)

```bash
cd /home/ubuntu/EDA
python3 -c "import ast; ast.parse(open('contest_solution/my_optimizer.py').read())"   # syntax gate
cp contest_solution/my_optimizer.py external/FloorSet/iccad2026contest/
cd external/FloorSet/iccad2026contest
PYTHONPATH=.. /usr/bin/python3 iccad2026_evaluate.py --evaluate my_optimizer.py --output /home/ubuntu/EDA/results/<tag>.json
cd /home/ubuntu/EDA
python3 scripts/analyze_results.py results/<tag>.json --top 20      # local RF=1 score + per-band gaps
python3 scripts/score_real.py results/<tag>.json results/quadratic_v1.json  # NEW (A1): runtime-adjusted, vs baseline
```
- **Keep a change only if it improves the runtime-adjusted total (A1, median 1–3s) AND stays 100/100 feasible.** The local RF=1.0 number is necessary but NOT sufficient.
- Single-case debug: `--test-id <id> --verbose`. Focus debugging on the top-weighted cases: 99,98,97,96,95,89 (see log Top-15).
- Commit each accepted win; tag its result JSON. Update `PLAN_EXECUTION_LOG.md` (score, Δ, feasible, **runtime sum/max**, runtime-adjusted score, what changed).

---

## Dead-ends — DO NOT retry (from logs + this analysis)

Carry forward all Sprint 1–3 dead-ends (BFS ordering; multi-start SA *as built*; force-directed *refinement under hard overlap rejection*; centroid sorting; equal-area-only/overlap-reject position swaps; compaction-to-origin; analytical x-ordering **without cluster super-blocks**; real-cost in the *tuned* variant loop; 200 relaxation sweeps; linear QP **without spreading**; QP+relaxation hybrid). Plus Sprint 4:
- Abacus legalization as a *drop-in portfolio member* did **not** beat plain portfolio (2.4030 vs 2.3977) and aspect-ratio fitting was ~0% — because they were bolted onto the same weak/slow polish and selected by true cost only at the end. Re-attempt aspect-ratio and order-preserving legalization **only** inside the new incremental engine (B) and spread global placement (C), where they have a fast feedback loop.
- **Do not** add more portfolio members that each run a heavy independent SA — that is what blew up runtime.

---

## 24/7 work queue (how to spend continuous compute without thrashing)

1. **A1 harness** → publish the real-score scoreboard (baseline vs current). *(hours)*
2. **A2 re-architecture** (two-stage portfolio, time budgets, pool reuse) → re-measure. *(expect first real-score win here)*
3. **B1 incremental cost** + **B2 legalization-aware moves** → the core engine; iterate move-set & schedule with A1 as judge. *(biggest ROI; days)*
4. **C1/C2 spread global placement + obstacle-aware legalization** → new seeds for B. *(days)*
5. **E1 mine golden** in parallel from day 1 (cheap, informs B/C). 
6. **F cross-validated tuning** running continuously once B is stable (large offline sweeps, select on held-out folds).
7. **D sequence-pair/B*-tree SA** and **E2 learned seeding** as parallel stretch tracks once A–C have banked real-score gains; ship only if they beat the analytic pipeline on held-out, runtime-adjusted score.

Between experiments, always leave the repo: committed, 100/100 feasible, with the best runtime-adjusted config as HEAD.

---

## Housekeeping
- Delete stale `results/summary.json` (bogus 1.50).
- Add `scripts/score_real.py` (A1) and `scripts/mine_golden.py` (E1) to the repo.
- Keep result JSONs tagged per experiment; never overwrite baselines.

---

## What to send back to the planning agent after this sprint

In `PLAN_EXECUTION_LOG.md`, per thrust attempted:
1. **Runtime-adjusted scoreboard** (A1) at median∈{1,2,3,5}s, current vs baseline — this is now the headline metric, not RF=1.0.
2. Local RF=1 score, feasibility, **runtime sum + max single-case wall-clock**, pool size used.
3. Per-band hpwl_gap/area_gap (n≥100) before/after; did B/C close the hpwl gap? did area drop?
4. For B: moves/sec achieved, whether incremental soft-cost tracking changed V_rel.
5. For C: did spreading reduce legalization displacement / hpwl vs the old QP?
6. Any new dead-ends (extend the dead-end list).
7. `analyze_results.py --top 20` dump of the best result, and the best result JSON path.

That lets the next plan target the new bottleneck precisely.
```
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```
