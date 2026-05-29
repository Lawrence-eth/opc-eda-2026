# FloorSet ICCAD-2026 — MASTER PLAYBOOK (self-navigating, multi-week)

**Author:** planning agent (Opus 4.8). **Supersedes** `NEXT_PLAN.md`, `SPRINT5_PLAN.md`, `SPRINT6_PLAN.md` (kept for history).
**Audience:** executing agent running 24/7 with unlimited compute/tokens. **This document is designed so you do NOT need a new plan every hour** — it encodes the decision logic. Navigate it yourself via Part III. Only stop for a re-plan when an **Escalation Trigger (Part VI)** fires.

> How to use this file: Read Part I (invariants) and Part II (facts) once. Then operate the **decision tree in Part III** — it always tells you the single next action based on your latest result. Part IV gives the deep implementation spec for whatever track Part III sends you to. Part V is the backlog for idle compute. Part VI says when to come back to the planner. Part VII is the standing protocol.

---
---

# PART 0 — CURRENT STATE & CRITICAL PATH (updated 2026-05-29, after Sprint 6 round 1)

**HEAD = sprint5_v9-equivalent: local 2.7182, 0.18s avg, 100/100. This is the best committed solution and is hard to beat because it sits at the runtime floor.** Sprint 6 round 1 tested P0, P1.A, P1.B — all reverted. The reverts were CORRECT and the evidence has converged on one root cause. Read this before navigating Part III.

### What Sprint 6 round 1 proved (3 experiments, 1 conclusion)
1. **P0 fast portfolio (construction-only members, persistent pool): reverted.** Persistent pool worked, but quality only reached **2.670** vs v9's 2.718 (a mere 1.8% gain) at 2.3× the runtime (0.41s). Runtime-adjusted, **v9 dominates at every median up to ~5s** (verified via `score_real.py`). **Correction to an earlier planning claim:** portfolio_v1's 2.40 quality came from its *per-member SA*, NOT from member diversity. **Diversity alone is not a quality lever.** Do not pursue construction-only portfolios for quality.
2. **P1.A shape-aware packing: reverted — 15/100 feasible.** Giving soft blocks non-square aspect ratios makes the **shelf packer produce overlaps** (its fixed-row-height model can't absorb wider/shorter blocks). The shape lever (the biggest area opportunity, II.2) is real but **the shelf packer physically cannot exploit it.**
3. **P1.B correctness-first polish (full-recompute greedy): reverted.** Regression-proof and feasible (good — the discipline works), but **full true-cost recompute per move is too slow** (+0.16s for ~nil gain) and the moves are still weak.

### The converged conclusion → THE CRITICAL PATH
**The shelf packer is the quality ceiling, proven three independent ways** (can't do shapes; diversity around it doesn't help; polishing its output is too slow for the gain). Patching it is exhausted. **The next move is not optional and not cheap: replace the packer.**

➡️ **CRITICAL PATH = build a skyline/tetris packer with integrated shape (aspect-ratio) selection** — see the new **Part IV.PACKER** spec below. This is the substrate that (a) unlocks the 52%→~96% utilization win (the single biggest quality lever), (b) is feasible *by construction* (skyline never overlaps), and (c) is fast (O(n log n)). It is a lighter, lower-risk first step than full topological SA (P2) and likely captures most of the area win. Everything else (incremental polish P1.B, topological SA P2) layers on top of it.

### Revised navigation (overrides Part III until the packer lands)
You are at **STATE: PACKER**. Do Part IV.PACKER. Gate: prototype must hit **>0.70 utilization on cases 99/97/95 standalone** before integration; integrated result must beat v9 runtime-adjusted at median∈{1,2,3}s, 100/100. On pass → resume Part III at STATE 2 (P2) with the new packer as substrate. Do NOT spend more runtime to win (v9 is at the floor); win on QUALITY at v9-level runtime (~0.2–0.4s).

### Standing correction to strategy
Quality comes from **a better packer + a fast optimizer**, not from portfolio breadth and not from spending more wall-clock. Target: match v9's runtime (~0.2–0.4s, at floor) while cutting hpwl_gap/area_gap. Stop re-testing portfolio-diversity and shape-on-shelf (both are now dead-ends, II.6).

---
---

# PART I — INVARIANTS (these never change; everything is judged against them)

### I.1 Mission
Win the ICCAD-2026 FloorSet contest (Problem C). The contest ranks submissions by total cost on a **hidden** 100-case set (same 21–120 block range, different instances). Minimize the **runtime-adjusted** total cost while remaining **100% feasible**.

### I.2 The objective you are actually optimizing
Per case (`iccad2026_evaluate.py:306`, `ALPHA=0.5 BETA=2.0 GAMMA=0.3 M=10`):
```
cost_i = (1 + 0.5·(max(0,hpwl_gap) + max(0,area_gap))) · exp(2·V_rel) · max(0.7, (rt_i/median)^0.3)
       = 10.0 if infeasible
Total  = Σ_i cost_i · exp(n_i/12) / Σ_i exp(n_j/12)         (n≥100 ≈ 79% of total weight)
hpwl_gap = max(0,(hpwl−hpwl_base)/hpwl_base);  area_gap = max(0,(bbox−area_base)/area_base)
V_rel    = soft_violations / n_soft   (boundary + Σ(cluster components−1) + Σ(MIB distinct shapes−1))
```
Baselines (`results/_baselines.json`) are the **golden** hpwl/bbox; gaps are clamped ≥0 (you cannot score below golden on quality).

### I.3 Runtime is first-class — the floor logic (memorize)
`max(0.7,(rt/median)^0.3)`:

| rt/median | 0.10 | 0.305 | 0.5 | 1 | 2 | 4 | 7 | 15 |
|---|---|---|---|---|---|---|---|---|
| mult | 0.70 | **0.70** | 0.81 | 1.00 | 1.23 | 1.52 | 1.79 | 2.25 |

- **At rt ≤ 0.305·median you hit the hard 0.7 floor** (a flat 30% cost cut, unbeatable). Below the floor, **more speed buys nothing**.
- Above median, penalty is **uncapped**.
- **RULE (absolute): never trade quality for speed once you are at the floor.** Sprint 5's mistake. Spend the "free budget" (rt up to ~0.3·median) entirely on quality.
- The local evaluator hard-codes `rt/median=1.0`, so the local score **hides runtime**. Judge with `scripts/score_real.py` (Part VII) across median∈{0.5,1,2,3,5}s.

### I.4 Sacred rules
1. **Feasibility 100/100 is non-negotiable.** One infeasible case = cost 10 ≈ catastrophic. Every candidate passes the hard gate: no overlap (tol 1e-6), every soft-block area within ±1% of target, fixed dims exact, preplaced (x,y,w,h) exact. Never move a preplaced block.
2. **Judge only on the runtime-adjusted total** (Part VII), not the local RF=1 number. Keep a change iff it improves the runtime-adjusted total at median∈{1,2,3}s AND stays 100/100.
3. **De-overfit:** the ranking set is a *different* 100. Tune **count-agnostic** params; select hyperparameters on **held-out folds** of the 100, never on the full 100. Per-count tuned tables do not transfer.
4. **Never regress HEAD.** Always leave the repo committed, 100/100 feasible, with the best runtime-adjusted config. Use best-of gating (true-cost) so new methods can only help.
5. **Append every experiment to `PLAN_EXECUTION_LOG.md`** (Part VII template), including failures + why.

---
---

# PART II — KNOWLEDGE BASE (verified facts; do not re-derive)

### II.1 Where the score lives
n≥100 ≈ **79%** of total; n≥116 alone ≈ 34%. n≤80 ≈ 4% combined. **Optimize the big cases; ignore small-n micro-tuning.** Top weighted cases: 99,98,97,96,95,89 (debug here).

### II.2 Quality gap vs golden (the headroom — this is the prize)
Current best per dominant band (n≥100): hpwl_gap ≈ 1.12–1.37, area_gap ≈ 0.82–1.08. Meaning we are **~2.1× golden wirelength and ~1.8–2.0× golden bbox area.**
- **GOLDEN PACKS AT 96.6% area utilization (Σblock_area/bbox). WE PACK AT ~52%.** Closing to 74% util ⇒ area_gap 0.87→0.30 ⇒ quality factor −0.28. Closing further approaches golden.
- **GOLDEN USES SOFT-BLOCK ASPECT RATIOS UP TO 3:1** (median 1.45, p90 ≈ 2.5). At **constant area** (±1% allowed). The **evaluator does NOT check aspect ratio — only area** — so shape is a *free* variable, bounded only by your own choice. We under-use it.
- **Implication:** the area gap is mostly a **shape-aware tight-packing problem**, and it is the single biggest, most certain quality lever. Tighter packing also reduces HPWL (blocks closer). Aspect change does NOT change a block's center, so it has **no HPWL downside** — it only helps area + boundary-touch.

### II.3 Constraint reality (all 100 cases)
Every case has fixed + preplaced + MIB + cluster + boundary. Big-case (n≥100) profile: fixed 7–17, **preplaced 1–9 (present in all 21 big cases — fixed obstacles)**, MIB 1 group, cluster 3–4 groups, boundary 26–37 blocks (~30% must touch a bbox edge → forces a perimeter ring), b2b edges 661–7056. V_rel already ~0.10 (near floor) → **protect soft, don't chase it.**

### II.4 Runtime status
sprint5_v9 (current best): n≥100 ≈ 0.46s/case (at floor for median≥1.5s) but **quality regressed** (hpwl 1.37/area 1.08). It wastes the free quality budget. baseline quadratic_v1: 1.47s, quality 1.17/0.87. portfolio_v1: 6.96s (too slow), quality 1.12/0.82.

### II.5 Architecture & key line numbers (`contest_solution/my_optimizer.py`, ~3741 lines)
```
_worker_solve            :26    module-level pool worker (re-instantiates MyOptimizer)
solve                    :68    builds portfolio, **creates a NEW pool PER CASE (bug, fix in P0)**, true-cost select
_build_portfolio         :88    currently only 2 configs (shrunk due to pool overhead)
_construct_layout(shelf) :197   degree-ordering shelf packer + refine + SA  (the quality workhorse)
_choose_dimensions       :1896  picks (w,h) — **shape lever lives here; under-exploited (see II.2)**
_pack_interior_units     :356   shelf row packer
_refine_free_block_shifts:1093
_selection_cost (proxy)  :2000
_true_contest_cost       :2056  EXACT cost, feasibility-gated — use for all selection/gating (correct)
_soft_violation_count    :2069  matches evaluator numerator
_sa_post_optimization    :2521  SLOW (full HPWL recompute/move over ≤7000 edges) + WEAK (equal-area swaps, overlap-reject, frozen soft factor)
_analytical_construct    :2749  QP/contour path (loses to shelf at legalization)
_abacus_construct        :2874
_analytical_global_placement(QP/CG):3484   (linear QP; collapses without spreading)
_analytical_legalize     :3632
```

### II.6 Consolidated DEAD-ENDS (do NOT retry; from all sprints)
BFS ordering; multi-start SA as built; force-directed refinement under hard overlap-reject; centroid sorting; equal-area/overlap-reject position swaps; compaction-to-origin; **analytical x-ordering without cluster super-blocks** (41–58 soft viols); **linear QP without spreading**; QP+relaxation hybrid; real-cost in the *tuned* variant loop; 200 relaxation sweeps; **per-case process-pool creation** (self-inflicted overhead); **portfolio of heavy-SA members** (runtime blowup); **density-spread QP fed only into the losing analytical path**; **incremental-HPWL local search built incrementally-first** (broke, 83 worsened — must be correctness-first); **aspect-ratio fitting bolted onto the weak SA** (no effect — but see II.2: aspect IS a top lever when done in a real packer); **sacrificing quality for sub-floor speed**. **[Sprint 6] construction-only portfolio for quality** (diversity ≠ quality; only 2.67 vs 2.72); **non-square shapes on the shelf packer** (15/100 feasible — shelf row model can't absorb them; needs the Part IV.PACKER skyline packer); **full-recompute polish at scale** (too slow per move — must use incremental cost from a verified-correct base).

---
---

# PART III — DECISION TREE (your navigator; consult after every experiment)

> Determine your current STATE from the latest committed result + log, then take the indicated ACTION. Each phase has a hard ACCEPTANCE GATE and explicit on-pass / on-fail branches. Spec for each phase is in Part IV.

### STATE 0 — start here (P0 not yet committed)
**ACTION:** Implement **P0 (fast persistent-pool portfolio)** [Part IV.P0].
**GATE:** runtime-adjusted total beats sprint5_v9 at *every* median∈{0.5,1,2,3,5}s AND beats baseline at median≤3s; 100/100; n≥100 hpwl_gap≤1.12 & area_gap≤0.82 (≥ portfolio_v1 quality).
- **PASS →** commit as new baseline `s6_p0`; go STATE 1.
- **FAIL (quality recovered but still slower than floor) →** reduce per-member work / increase parallel breadth; re-measure. If pool won't persist (fork/torch issue), use the fallback in IV.P0.5.
- **FAIL (quality not recovered even with full portfolio) →** the construction-only members are too weak; add a single short shared polish (current SA) on top-2 winners; if still short, go STATE 2 early (the polish itself is the lever).

### STATE 1 — P0 banked; attack the quality ceiling
Run **P3.1 golden-mining priors** [IV.P3.1] first (cheap, ~1 day) to set concrete shape/util targets, THEN:
**ACTION:** Implement **P1 (shape-aware tight packing + correctness-first legalization polish)** [Part IV.P1]. This is the highest-EV quality work (II.2).
**GATE:** beats `s6_p0` runtime-adjusted at median∈{1,2,3}s; 100/100; **n≥100 area_gap drops below 0.7** (utilization >0.59) as the first milestone, then chase <0.5.
- **PASS →** commit `s6_p1`; go STATE 2.
- **FAIL (polish regresses or breaks feasibility) →** you violated correctness-first. Revert to full-recompute hill-climb with feasibility asserts (IV.P1.1). Do NOT proceed to incremental cost until monotone improvement is proven on cases 99/97/95.
- **PARTIAL (area improves, hpwl stuck) →** add connectivity-aware relocation moves (IV.P1.2 move 1) and re-seed packing order from QP centroids; if hpwl still stuck, go STATE 2 (topological SA is the structural HPWL fix).

### STATE 2 — P1 banked (or P1 plateaued); go for the structural breakthrough
**ACTION:** Implement **P2 (topological-representation SA: B*-tree/sequence-pair with shape curves + fixed obstacles)** [Part IV.P2]. Prototype obstacle handling on test 99 FIRST and prove exact feasibility before scaling.
**GATE:** beats `s6_p1` runtime-adjusted at median∈{1,2,3}s; 100/100.
- **PASS →** commit `s6_p2`; go STATE 3.
- **FAIL (obstacle feasibility intractable) →** demote P2 to a *packing subroutine* inside P1 for obstacle-free sub-regions; return to STATE 1 backlog (Part V) and STATE 3.
- **FAIL (feasible but not better) →** check the numba inner loop speed and seeding (must seed from P0/P1, not random); if still not better after schedule tuning, freeze P2 and go STATE 3.

### STATE 3 — analytic pipeline matured; pursue the moonshot + harden
Run **P3.2 (learned seeding)** [IV.P3.2] as a parallel research track AND **CONTINUOUS hardening** [IV.C].
**GATE for shipping P3.2:** beats the analytic pipeline on a **held-out fold**, runtime-adjusted, 100/100.
- Always keep the best analytic pipeline as HEAD; only swap in the learned seeder if it wins held-out.
- When all of P0–P3 are committed and gates plateau (no track improves runtime-adjusted total by >0.5% in its last full pass), **fire Escalation Trigger E4 (Part VI)** for a fresh strategic plan.

### Idle / blocked at any state
If a track is mid-build or waiting, pull the next item from the **Experiment Backlog (Part V)** (ranked by EV). Never leave the 48 cores idle.

---
---

# PART IV — TRACK SPECS (deep implementation detail)

## IV.P0 — Fast persistent-pool portfolio  *(STATE 0)*

**Goal:** portfolio_v1-level quality (≥) at ~0.3–0.5s/case (at/near floor). Beats sprint5_v9 at all medians.

**P0.1 Persistent pool.** Module-level lazy `ProcessPoolExecutor(max_workers=min(48,cpu))` with **fork**, created on first `solve()`, reused for all 100 cases, closed via `atexit`. Removes per-case fork/import overhead (the real cause of "portfolio too costly"). Keep a serial fallback if pool init throws.

**P0.2 Real portfolio (construction-only members, ≤0.15s each for n=120).** K≈16–32: shelf×row_factor{0.8,0.9,1.0,1.1} × {boundary-cluster on/off}; analytical(QP+contour); abacus; a few tie-break RNG seeds. **No per-member SA.** Select best by `_true_contest_cost` (:2056).

**P0.3 One shared polish.** Run the existing `_sa_post_optimization` (or P1 engine when ready) ONLY on the top-1/top-2 winners, hard time-budgeted so total wall-clock stays ≤ ~0.3·median_guess (target n≥100 ≈ 0.3–0.5s). Best-of-gated.

**P0.4 Budget calibration.** Use `score_real.py` to confirm n≥100 sits at/near the 0.7 floor for median∈{1,2,3}s; spend all remaining "free" time on portfolio breadth + polish, none beyond the floor.

**P0.5 Fallback if fork+torch unstable:** use `multiprocessing.get_context('fork')`; if a worker imports CUDA, force CPU in workers (`CUDA_VISIBLE_DEVICES=""`); if still unstable, run K members **serially in-process but SA-free** (each ≤0.05–0.1s) — even 16 serial SA-free constructions fit in ~0.3–0.5s for n=120.

## IV.P1 — Shape-aware tight packing + correctness-first legalization polish  *(STATE 1; highest quality EV)*

This track has two coupled sub-levers. **Lever A (AREA, highest-confidence per II.2): shape-aware tight packing.** **Lever B (HPWL): connectivity-aware legalization-repair moves.** Build A first (certain win), then B.

**P1.A — Shape-aware tight packing (target util 0.96 like golden).**
- Rework dimension choice (`_choose_dimensions` :1896) and the packer so soft-block (w,h) is chosen **to tile tightly**, not near-square. Allowed shapes = any (w,h) with `|w·h−target|/target ≤ 0.01`; aspect is otherwise free (golden uses ≤3:1; you may exceed since the evaluator doesn't check aspect, but start with ≤3:1 as a sane prior and test relaxing it).
- Use a **skyline/shelf packer with shape selection**: when placing a block, pick the aspect ratio (within ±1%) that best fills the current gap (match remaining row height, or fill a column to the contour) — a 1-D strip-packing-with-resizable-items step.
- Add a **post-pack shape+compaction sweep:** for each soft block, try aspect ratios that (i) close the contour gap above/beside it, (ii) let a boundary block reach its edge, (iii) let a cluster member abut its mate; accept by exact `_true_contest_cost`, re-legalizing locally. (This is the Sprint-4 aspect idea, but now inside a packer that can exploit it.)
- **MIB group:** reshape all members together (must stay identical (w,h)).
- **Acceptance milestone:** n≥100 utilization >0.59 (area_gap<0.7), then >0.74 (area_gap<0.3).

**P1.B — Correctness-first legalization-aware polish (fixes the broken Thrust B).**
1. **Correctness first (regression-proof):** greedy hill-climb on a copy of the best layout; propose move → apply → **recompute FULL exact `_true_contest_cost`** → accept iff strictly lower, else revert. Monotone ⇒ cannot regress. After every accept, **assert hard feasibility** (overlap/area/preplaced). Validate true-cost improvement + feasibility on cases 99,97,95 before wiring in (best-of gated).
2. **Repair-not-reject moves** (preserve hard constraints): (a) relocate block toward connectivity centroid / into a gap, ripple-push the few overlappers along min-displacement axis; (b) unequal swap + local re-legalize; (c) row/slab inward shift then re-touch boundary; (d) aspect reshape (P1.A); (e) cluster rigid-move toward centroid.
3. **Only then add incremental cost** (per-net HPWL via incident edges using adjacency near :3200; bbox via tracked extremes; soft via per-cluster components / per-MIB shape sets) — **with a debug assert `incremental≈full` every N moves.** Greedy sweeps → short SA tail (exact incremental cost, true `exp(2·V_rel)`, not frozen). Run a few parallel chains on the pool; keep best.

## IV.PACKER — Skyline/tetris packer with shape selection  *(CRITICAL PATH; build this next)*

**Why:** the shelf packer is the proven ceiling (Part 0). A skyline packer is feasible-by-construction, fast (O(n log n)), and is the only substrate that can exploit the shape lever (II.2: golden = 96% util via aspect ≤3:1; we are at 52%).

**Core data structure — skyline (a.k.a. contour/Bottom-Left-Fill):**
- Maintain the upper contour of placed blocks as a list of horizontal segments `(x_start, x_end, height)`.
- To place a block of footprint (w,h): scan candidate x-positions (segment left edges and preplaced-obstacle edges); for each, the landing y = max contour height over [x, x+w]; choose the (x) that minimizes a placement score (primary: resulting max-contour-height / wasted area beneath; tie-break: proximity to the block's connectivity-centroid x for HPWL). Update the contour. This **never overlaps** (y is always above the contour).
- Preplaced blocks: initialize the contour to include them as fixed obstacles (raise the contour over their footprint to their top); movable blocks then pack around them automatically. This is how obstacles are handled cleanly — no special cases.

**Shape selection (the lever):** when placing a soft block, do NOT use a fixed (w,h). Enumerate a small set of aspect ratios r∈{1.0,1.3,1.6,2.0,2.5,3.0} (and their transposes), each giving (w,h)=(√(area·r), √(area/r)) renormalized so `|w·h−target|/target ≤ 0.01`. Pick the (r, orientation) that best fills the current contour notch — e.g. width matching a valley, or the shape that minimizes contour-height increase. This is "strip packing with resizable items." MIB group: pick ONE shared shape for all members. Boundary blocks: bias shape/placement so the block can touch its required edge.

**Ordering:** try several orders as cheap portfolio members (we have the pool): (a) decreasing area (classic BLF), (b) decreasing max-side, (c) connectivity-centroid sweep with clusters as super-blocks (super-blocks mandatory — bare analytical x-order is a dead-end). Pack clusters as contiguous macro-blocks (reuse `_cluster_local_pack`).

**Aspect ratio is free re HPWL:** changing (w,h) does not move a block's center, so it has zero HPWL cost — it is a pure area/boundary lever. You may even exceed 3:1 (evaluator doesn't check aspect; backlog item 2) — test it.

**De-risked build & validation (avoid the try→revert trap):**
1. Build the packer as a STANDALONE function. **First milestone: on cases 99, 97, 95, measure raw utilization directly** (Σarea/bbox). Gate: **>0.70** (vs current ~0.52) before any integration. If <0.70, iterate the placement score / shape set in isolation — do NOT touch `solve()` yet.
2. Assert feasibility (no overlap, areas within ±1%, preplaced untouched) inside the packer.
3. Only after the standalone util gate passes: add it as a portfolio path (`'path':'skyline'`) selected by `_true_contest_cost`; it can only help (best-of gated).
4. Then layer the P1.B incremental polish on its output, and use it as the P2 seed.

**Acceptance:** beats v9 runtime-adjusted at median∈{1,2,3}s, 100/100, at ~v9 runtime (≤~0.4s). Expect the area_gap to be the first big mover. Tag `s6_packer`.

## IV.P2 — Topological-representation SA  *(STATE 2; structural HPWL+area fix, high ceiling)*

Every analytic/QP attempt has died at legalization; **topological SA has no legalization step (every state is a valid packing)** — the reason it's the gold standard for ≤ few-hundred blocks and the route to near-golden quality.
- **Engine:** B*-tree (preferred — natural min-area + obstacle + shape-curve support) or sequence-pair; **packing/longest-path inner loop in numba `@njit` or C** (pure Python too slow for 10⁴+ moves). Soft blocks via **shape curves** (optimal aspect per slot) or discrete per-block aspect set — this is where P2 natively exploits II.2.
- **Objective:** exact contest cost (area+HPWL from packed coords; soft via P1 incremental counters).
- **Seed** from P0/P1 best (decode coords→tree), not random. **Parallel-temper** short chains across 48 cores. Hard wall-clock per I.3.
- **Fixed obstacles (do FIRST on test 99, prove exact feasibility):** B*-tree-with-preplaced (insert preplaced as fixed nodes, adjust contour so movable never overlaps them; literature: B*-tree/TCG with pre-placed modules), or pragmatic pack-then-verify/repair against obstacle rects (reject overlapping states). Ship only if 100/100 feasible.

## IV.P3 — Data-driven (contest's stated theme; fast inference ⇒ floor-friendly)

**P3.1 Mine golden (DONE once, extend as needed; do early — informs P0/P1/P2).** `scripts/mine_golden.py`. Known so far: util 0.966, aspect ≤3:1 (med 1.45). Extend to: per-cluster shape/topology, perimeter-ring layout pattern, per-net wirelength distribution, how golden places preplaced relative to clusters. Turn findings into concrete priors (preferred aspect set, target density, ring builder).

**P3.2 Learned seeding (offline training, A100 available; STATE 3).** Train a small **GNN** on (area targets + b2b/p2b connectivity + constraints) → per-block (position, aspect) and/or ordering/sequence-pair; supervise on golden (+ optional RL with the evaluator reward near :1155). Inference = one forward pass (ms ⇒ floor) → seed → P1/P2 polish. **Generalization risk:** augment heavily, **select on held-out fold**, ship only if it beats analytic pipeline held-out. Keep model small for fast inference.

## IV.C — CONTINUOUS hardening (run throughout once P0 is in)
- **Cross-validated tuning:** k-fold over the 100; tune count-agnostic params on train folds, select on held-out. Large sweeps OK (unlimited compute) but **never select on the full 100.**
- **Median-robustness:** prefer configs Pareto-good across median∈{1,2,3,5}s.
- **Feasibility fuzzing:** run on FloorSet Lite *training* instances (not just the 100) to catch hidden-set feasibility edge-cases.
- **Housekeeping:** delete stale `results/summary.json`; keep result JSONs tagged; commit each win.

---
---

# PART V — EXPERIMENT BACKLOG (ranked by EV; pull when idle/blocked)

1. **Shape-aware packing prototype** on case 99 in isolation: measure best achievable utilization with aspect∈{1:1..3:1} vs current. (Validates IV.P1.A ceiling.) *(highest EV)*
2. **Relax aspect beyond 3:1** experiment: does aspect up to 5:1 or 8:1 pack tighter / touch more boundaries without HPWL cost? (Evaluator allows it.) Keep only if runtime-adjusted improves held-out.
3. **Connectivity-aware packing order** (seed shelf/skyline order from QP centroid x, clusters as super-blocks) — careful: analytical x-order w/o super-blocks is a dead-end; super-blocks are mandatory.
4. **Boundary-ring constructor:** build the ~30% boundary blocks as an explicit perimeter frame (golden structure) then pack the interior — measure soft + area.
5. **Polish move-weight tuning** via held-out folds.
6. **Per-net weighting:** check if hub nets (high-degree) dominate HPWL; prioritize their endpoints in placement.
7. **MIB-aware shaping:** pick the single MIB-group shape that best packs across all members.
8. **Parallel-tempering schedule sweep** for P2.
9. **Warm-start P2 from P1** vs random — quantify.
10. **GNN feature ablations** (P3.2).

---
---

# PART VI — ESCALATION TRIGGERS (the ONLY times to stop for a re-plan)

Fire one of these → write the situation to `PLAN_EXECUTION_LOG.md` and request a new plan. Otherwise keep navigating Part III autonomously.
- **E1 — Feasibility cannot be maintained:** a track can't stay 100/100 on the validation set despite the gate (e.g., P2 obstacle handling fundamentally infeasible). 
- **E2 — Metric/rule surprise:** the evaluator/contest reveals behavior contradicting Part I (e.g., a hard runtime cap, aspect-ratio constraint, batch eval, or the median assumption proven wrong by an official leaderboard).
- **E3 — A track wins big and opens new questions** the playbook doesn't cover (e.g., learned seeder beats everything and you need a training-data/eval strategy).
- **E4 — Global plateau:** all of P0–P3 committed and no track improves the runtime-adjusted total by >0.5% in its last full pass. (Time for a fundamentally new idea.)
- **E5 — Contradiction with a stated fact in Part II** (e.g., golden util turns out not to be reachable due to a missed constraint).
Do NOT escalate for ordinary failed experiments — those are expected; log them and pull the next backlog item.

---
---

# PART VII — STANDING PROTOCOL

### Run + judge (every change)
```bash
cd /home/ubuntu/EDA
python3 -c "import ast; ast.parse(open('contest_solution/my_optimizer.py').read())"          # syntax gate
cp contest_solution/my_optimizer.py external/FloorSet/iccad2026contest/
cd external/FloorSet/iccad2026contest
PYTHONPATH=.. /usr/bin/python3 iccad2026_evaluate.py --evaluate my_optimizer.py --output /home/ubuntu/EDA/results/<tag>.json
cd /home/ubuntu/EDA
python3 scripts/analyze_results.py results/<tag>.json --top 20
python3 scripts/score_real.py results/<tag>.json results/<current_best>.json results/quadratic_v1.json
```
- Single-case debug: `--test-id <id> --verbose` (use 99,98,97,95,89).
- **KEEP a change iff** runtime-adjusted total (median 1–3s) strictly improves AND feasibility 100/100. Else revert.
- Commit each win: `git add -A && git commit -m "..."` ending with the Co-Authored-By line below.

### `scripts/score_real.py` (must exist from Sprint 5; if missing, build it)
Reads result JSON(s) with per-case {runtime, hpwl_gap, area_gap, violations_relative, block_count}; recomputes weighted Total under median∈{0.5,1,2,3,5}s using the I.2 formula; prints a side-by-side table for all JSONs passed. This is the scoreboard.

### Log template (append to `PLAN_EXECUTION_LOG.md` per experiment)
```
### <date> <tag> — <one-line change>
- Runtime-adjusted Total @ median {0.5,1,2,3,5}s: a / b / c / d / e   (vs prev-best: …)
- Local RF=1 Total: X | Feasible: N/100 | runtime sum/max: Ss / Ss
- n≥100 hpwl_gap: … area_gap: … (utilization …) V_rel: …
- Verdict: ✅ kept / ❌ reverted — why
- New dead-ends (if any): …
```

### Commit / PR footers
```
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

---

## One-paragraph north star (re-read when lost)
We are ~2× golden on both wirelength and area, and golden proves **96% packing via aggressive aspect-ratio use is achievable** — that's the prize. Bank a robust win first (P0: fast portfolio at the runtime floor), then close the area gap with **shape-aware tight packing** (P1.A, highest-confidence lever) and the HPWL gap with **legalization-repair local search** (P1.B, correctness-first), then break the structural ceiling with **topological SA** (P2) and finally the **learned seeder** (P3). Judge everything on the runtime-adjusted scoreboard, keep 100% feasible, de-overfit on held-out folds, and only stop for the Part VI triggers.
