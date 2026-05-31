# FloorSet ICCAD-2026 — Sprint 6 Plan (win the contest)

**Author:** planning agent (Opus 4.8)
**For:** executing agent — 24/7, compute/tokens unlimited. **Contest inference runtime is still a hard scoring input (RuntimeFactor); see §1.**
**Read first:** this file, then the Sprint 5 section of `PLAN_EXECUTION_LOG.md`, then `SPRINT5_PLAN.md` §1 (the runtime math) and the dead-end lists.

---

## 0. Where we are (verified) and the one decision that matters

| Result | local (RF=1) | n≥100 hpwl_gap | n≥100 area_gap | n≥100 runtime | wins when… |
|--------|--------------|----------------|----------------|---------------|------------|
| quadratic_v1 (baseline) | 2.466 | 1.17 | 0.87 | 1.47s | — |
| portfolio_v1 (S4, best quality) | 2.398 | 1.12 | 0.82 | 6.96s | never (too slow) |
| **sprint5_v9 (S5, current best)** | **2.718** | **1.37** | **1.08** | **0.46s** | **median ≤ ~3s** |

Sprint 5 correctly discovered runtime dominates and built the runtime-adjusted scoreboard — **but then made a fragile trade**: it deleted the portfolio and ran a single fast path, **losing ~15% quality on the band that is 80% of the score**, betting the contest median is low. At median ≥5s it loses to the baseline.

**The decisive realization (this sprint's thesis):**
- Once a case is at the **0.7 runtime floor** (reached at `rt/median ≤ 0.305`), *more speed is worthless and quality is everything.* sprint5_v9's 0.46s big-cases are already at the floor for median ≥1.5s. **It is wasting a ~0.3–0.6s/case "free quality budget."**
- The "portfolio overhead" that killed Sprint 5's portfolio was **self-inflicted**: `solve()` creates a **new process pool per case** (line ~127), paying fork+torch-import+pickle every case. Fix that and a *real* parallel portfolio costs ~0.2–0.5s wall-clock.

⇒ **Priority 0 is non-negotiable and high-confidence: a fast real portfolio on a *persistent* pool that recovers portfolio_v1 quality at floor-runtime.** That strictly dominates sprint5_v9 at *every* median (better quality, still at/near floor) and becomes the new baseline. Everything after is about breaking the quality ceiling that even portfolio_v1 (still ~2× golden) hits.

---

## 1. The metric you optimize (unchanged, but now the *only* judge)

Per case: `cost = (1+0.5·(max(0,hpwl_gap)+max(0,area_gap)))·exp(2·V_rel)·max(0.7,(rt/median)^0.3)`; weighted by `exp(n/12)` (n≥100 ≈ 79%).
- **Judge every change with `scripts/score_real.py`** (built in Sprint 5) across **median ∈ {0.5,1,2,3,5}s**, side-by-side vs the current best AND baseline. The local RF=1 number is necessary but NOT sufficient.
- **Never trade quality for speed below the floor.** Target wall-clock per case ≈ `min(median_guess·0.3, …)`; in practice aim **~0.3–0.6s for n≥100, ~0.1s for small n**. Being faster than that buys nothing (floor) and only costs quality.
- **Robustness rule:** one infeasible case = cost 10 = catastrophic on the hidden set. Feasibility 100/100 is sacred; the true-cost gate must reject any infeasible candidate (overlap / area>±1% / moved preplaced).

---

## PRIORITY 0 — Fast real portfolio on a persistent pool  *(do first; high confidence; new baseline)*

### P0.1 Persistent process pool (the unlock)
- Create ONE module-level `ProcessPoolExecutor` (fork start method) **lazily on first `solve()` and reuse for all 100 cases**; shut down via `atexit`. Workers are stateless (`_worker_solve` re-instantiates `MyOptimizer`). This removes per-case fork/import overhead — the thing that made Sprint 5 think portfolios were too costly.
- Pickling per case is just the small per-case tensors (cheap). Keep passing numpy/lists as now.
- Threads won't help (pure-Python placement is GIL-bound) — must be processes.
- Guard with a serial fallback; verify with `score_real.py` that per-case wall-clock dropped.

### P0.2 Restore a real, *fast* portfolio (no per-member heavy SA)
- Members = diverse **construction-only** configs (target ≤0.15s each for n=120): shelf × {row_factor 0.8,0.9,1.0,1.1}, analytical (QP+contour), abacus, ± boundary-cluster variants, a few RNG seeds where ordering has ties. Aim **K≈16–32** members (we have 48 cores; persistent pool makes this ~free in wall-clock).
- Select best by `_true_contest_cost` (exact, already correct at line 2056).
- **Then ONE short shared polish** (the current `_sa_post_optimization`, or the P1 engine once ready) on the **top-1 or top-2** winners only, with a hard time budget so total wall-clock stays in the §1 target. Polish is best-of-gated so it can't regress.

### P0.3 Calibrate the budget to the floor
- Use `score_real.py` to find the wall-clock that keeps n≥100 at/near the 0.7 floor for median∈{1,2,3}s, then spend exactly that much on portfolio breadth + polish. Don't exceed it.

**Acceptance:** runtime-adjusted total beats sprint5_v9 at **every** median in {0.5,1,2,3,5}s, and beats baseline at median ≤3s; feasibility 100/100; n≥100 quality ≥ portfolio_v1 (hpwl_gap ≤1.12, area_gap ≤0.82). Tag `s6_p0`. **This is the new must-beat baseline.**

---

## PRIORITY 1 — Correct legalization-aware polish (fix the broken Thrust B)  *(quality ceiling, lower risk)*

Sprint 5's "fast incremental local search" **broke (83 worsened)**. It was not disproven — it was buggy. The failure mode is classic: it jumped straight to *incremental* HPWL (fast but easy to get wrong) and optimized a corrupted signal, and/or its repair moves created worse layouts the buggy cost didn't see. Rebuild it with **correctness-first discipline**:

### P1.1 Correctness first (cannot regress)
- Implement as a **greedy hill-climb polish** on a *copy* of the best portfolio layout: propose a move, apply, **recompute the FULL exact `_true_contest_cost`**, accept only if it strictly decreases, else revert. No incremental math yet. This is monotone — it can only improve or no-op, so it is **regression-proof by construction**.
- After every accepted move, **assert hard feasibility** (no overlap, areas within ±1%, preplaced untouched). If an assert ever fails, the move generator is buggy — fix before proceeding.
- Validate on the top weighted cases (99,98,97,95,89) that it improves true cost and stays feasible. Only then wire it in (best-of-gated).

### P1.2 Moves that *repair* instead of *reject* (the quality unlock)
All moves preserve hard constraints (never move preplaced; keep area within ±1% on reshape):
1. **Relocate-and-ripple:** move block `i` toward its connectivity centroid / into a gap; push the few overlapping neighbors along the min-displacement axis (bounded ripple), or snap `i` to the nearest legal skyline slot.
2. **Unequal swap + local re-legalize:** swap two blocks/clusters of different sizes; re-pack only the affected neighborhood.
3. **Row/slab inward shift:** slide a shelf row or vertical slab inward to shrink the bbox (area gap), then re-touch boundary blocks to the new extremes.
4. **Aspect reshape (±1% area):** retune a soft block's (w,h) — or a whole MIB group together — to fill a contour gap (area), reach its boundary edge (kills a soft violation), or abut a cluster mate.
5. **Cluster rigid-move:** translate a whole cluster toward its centroid (keeps abutment), repair overlaps.

### P1.3 Then make it fast (only after it's correct)
- Add incremental cost (per-net HPWL via incident-edge re-sum using adjacency already built near line 3200; bbox via tracked extremes; soft via per-cluster component counts / per-MIB shape sets). **Keep a debug mode that asserts `incremental == full` every N moves.** This is the safety the broken version lacked.
- Greedy sweeps → short SA tail with the incremental exact cost. With this you get 10⁴–10⁵ moves in <1s for n=120. Run a few parallel chains on the pool; keep best.

**Acceptance:** beats `s6_p0` on runtime-adjusted total; 100/100 feasible; n≥100 hpwl_gap and/or area_gap drop measurably. Tag `s6_p1`.

---

## PRIORITY 2 — Topological-representation SA (B*-tree / sequence-pair)  *(high ceiling; the structural fix)*

Every analytical/QP attempt across 4 sprints has died at **legalization** (snap-to-non-overlap destroys the global solution; shelf ordering keeps winning). Topological-representation SA **has no legalization step — every state is already a valid compact packing** — which is exactly why it is the academic gold standard for ≤ a few hundred blocks and the most likely route to *near-golden* area+wirelength.

### P2.1 Engine
- Implement **B*-tree** (preferred for fixed-outline-free min-area + obstacle support) or **sequence-pair**; packing/longest-path inner loop in **numba `@njit`** or a small C extension (pure Python is too slow for 10⁴+ moves). With unlimited dev time this is worth building once and reusing.
- Soft blocks: per-block discrete aspect-ratio set (each within ±1% area) chosen in the move set, or shape-curve optimization.
- Objective = exact contest cost (area+wirelength via packed coords; soft via the P1 incremental counters).
- Seed from the P0 best layout (decode coords → initial tree/sequence) rather than random; **parallel-temper** short chains across the 48 cores; hard wall-clock budget per §1.

### P2.2 Fixed obstacles (the hard part — prototype FIRST)
Every case has 1–9 **preplaced** blocks at fixed (x,y,w,h). Before scaling, prototype obstacle handling on test 99 and **prove exact feasibility**:
- B*-tree-with-preplaced: insert preplaced as fixed nodes and adjust the contour so movable packing never overlaps them (literature: B*-tree / TCG floorplanning with pre-placed modules), OR
- Pragmatic: pack movable blocks with the SA, then verify/repair against the preplaced rectangles; reject states that overlap obstacles. Accept the engine only if it stays 100/100 feasible.
- If exact-feasible obstacle handling proves too costly, fall back: use P2 only as a **packing subroutine inside the P1 polish** for obstacle-free sub-regions.

**Gate:** pursue in parallel with P1 but **ship only if it beats `s6_p1`** on runtime-adjusted total at 100/100 feasibility. Tag `s6_p2`.

---

## PRIORITY 3 — Data-driven placer (the contest's stated theme; fast inference)  *(moonshot, highest ceiling)*

The contest is literally **"Data-Driven SoC Floorplanning"** and ships the **FloorSet Lite dataset locally** (`external/FloorSet/LiteTensorDataTest`, loaders `lite_dataset.py`/`litetestLoader.py`, `training_example.py`). A learned model is **~ms at inference → trivially at the runtime floor**, and can capture golden-quality structure heuristics miss.

### P3.1 Mine the golden solutions (cheap; do this in week 1 regardless — informs P0/P1/P2)
`scripts/mine_golden.py`: load golden layouts (via `_extract_baseline`, line ~806) and measure soft-block aspect-ratio distributions, area utilization (golden bbox / Σarea — tells us the achievable area_gap≈0 target), perimeter-ring structure, cluster shapes, per-net wirelength. Feed concrete priors into P0/P1 (preferred aspect ratios, target packing density, ring construction).

### P3.2 Learned seeding (offline training, unlimited compute, A100 available)
- Train a **GNN** on (area targets + connectivity + constraints) → per-block (position, aspect ratio) and/or an ordering/sequence-pair, supervised by golden solutions (+ optional RL with the contest reward; evaluator exposes one near line ~1155).
- Inference: one forward pass → seed → fast P1/P2 legalize+polish. Keep the model small for fast CPU/A100 inference.
- **Generalization is the risk** (hidden set differs): train with heavy augmentation, **select on a held-out split of the 100**, and ship only if it beats the analytic pipeline on held-out, runtime-adjusted score.

---

## CONTINUOUS — de-overfit, robustness, scoreboard discipline

- **Cross-validated tuning:** all hyperparameters (portfolio composition, move weights, SA schedule, budgets) tuned on training folds, **selected on held-out folds** — never on the full 100. The hidden ranking set is a different 100 in 21–120.
- **Median-uncertainty robustness:** since the competitor median is unknown, prefer configs that are Pareto-good across median∈{1,2,3,5}s (the scoreboard). Do not over-fit to one median.
- **Feasibility fuzzing:** periodically run the solver on the FloorSet Lite *training* instances (not just the 100 validation) to catch feasibility edge-cases the hidden set might trigger.
- Delete stale `results/summary.json`. Add `score_real.py` (exists), `mine_golden.py` to repo.

---

## DEAD-ENDS — do not retry (carry forward + Sprint 5)
All prior dead-ends (BFS ordering; multi-start SA as built; force-directed refinement under hard overlap-reject; centroid sorting; equal-area/overlap-reject swaps; compaction-to-origin; analytical x-ordering without cluster super-blocks; linear QP without spreading; QP+relaxation hybrid; real-cost in the tuned variant loop; 200 sweeps) **plus Sprint 5:**
- **Per-case process-pool creation** (fixed in P0.1 — do NOT revert to it).
- **Portfolio of heavy-SA members** (runtime blowup — P0 uses construction-only members + single shared polish instead).
- **Density-spread QP into the analytical path** (no effect — the analytical/contour *legalization* is the weak link, not the QP targets; don't re-tune QP without fixing legalization, which P1/P2 do structurally).
- **Incremental-HPWL local search built incrementally-first** (broke, 83 worsened — P1 rebuilds correctness-first with full-recompute gating and feasibility asserts).
- **Sacrificing quality for sub-floor speed** (sprint5_v9's fragile bet — §1 forbids it).

---

## VALIDATION PROTOCOL (every change)
```bash
cd /home/ubuntu/EDA
python3 -c "import ast; ast.parse(open('contest_solution/my_optimizer.py').read())"
cp contest_solution/my_optimizer.py external/FloorSet/iccad2026contest/
cd external/FloorSet/iccad2026contest
PYTHONPATH=.. /usr/bin/python3 iccad2026_evaluate.py --evaluate my_optimizer.py --output /home/ubuntu/EDA/results/<tag>.json
cd /home/ubuntu/EDA
python3 scripts/analyze_results.py results/<tag>.json --top 20
python3 scripts/score_real.py results/<tag>.json results/sprint5_v9.json results/quadratic_v1.json   # runtime-adjusted, vs current best + baseline
```
Keep a change only if it improves the **runtime-adjusted** total (median 1–3s) at 100/100 feasible. Commit each win; update `PLAN_EXECUTION_LOG.md`.

---

## WHAT TO LOG BACK (so the next plan is precise)
1. **Runtime-adjusted scoreboard** at median∈{0.5,1,2,3,5}s for: new best, sprint5_v9, baseline. (Headline metric.)
2. Per-band hpwl_gap/area_gap (n≥100) before/after each priority; did P1/P2 finally beat the shelf packer's quality?
3. P0: per-case wall-clock with persistent pool + K used; confirm we sit at/near the 0.7 floor.
4. P1: did the correctness-first polish stay feasible and monotone? moves/sec after incrementalization? did `incremental==full` hold?
5. P2: obstacle-handling feasibility result on test 99; did topological SA beat P1 anywhere?
6. P3.1: golden mining findings (achievable area_gap, aspect-ratio priors, ring structure).
7. New dead-ends.
8. Best result JSON path + `analyze_results.py --top 20` dump.

---

## Suggested 24/7 ordering
P0 (bank the robust win) → P3.1 mine golden (cheap, informs everything) → P1 (correctness-first polish; biggest near-term quality ROI) → P2 + P3.2 as parallel high-ceiling bets → CONTINUOUS tuning throughout. Always leave HEAD committed, 100/100 feasible, best-runtime-adjusted config.
```
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```
