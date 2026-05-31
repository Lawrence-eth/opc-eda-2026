# FloorSet ICCAD-2026 — Execution Plan (Sprint 4)

**Author:** planning agent (Opus 4.8)
**For:** executing agent (cold start — read this whole file + `PLAN_EXECUTION_LOG.md` first)
**Baseline at handoff:** **2.4658**, 100/100 feasible, ~67s total / ~3.2s max single case (verified, reproducible).
**Target:** ≤ 1.5 (winning tier). Realistic intermediate milestone this sprint: **≤ 2.2**.

---

## 0. Orientation (read before touching code)

- Repo: `/home/ubuntu/EDA`. Optimizer: `contest_solution/my_optimizer.py` (~3100 lines).
- Evaluator (DO NOT EDIT): `external/FloorSet/iccad2026contest/iccad2026_evaluate.py`.
- The optimizer is copied into the evaluator dir before each run (see Commands).
- The working tree is clean except `PLAN_EXECUTION_LOG.md` (pre-existing uncommitted edit — leave it).

### Run / measure loop (use this exact loop after every change)
```bash
cd /home/ubuntu/EDA
python3 -c "import ast; ast.parse(open('contest_solution/my_optimizer.py').read())"   # syntax gate
cp contest_solution/my_optimizer.py external/FloorSet/iccad2026contest/
cd external/FloorSet/iccad2026contest
PYTHONPATH=.. /usr/bin/python3 iccad2026_evaluate.py --evaluate my_optimizer.py \
    --output /home/ubuntu/EDA/results/<tag>.json
cd /home/ubuntu/EDA
python3 scripts/analyze_results.py results/<tag>.json --top 20    # Total score + Feasible N/100
```
Single case debug: add `--test-id <id> --verbose`. Full run is ~60–70s.

**Acceptance rule for every change:** keep it ONLY if `Total score` strictly improves AND `Feasible` stays 100/100. Otherwise revert that change before moving on. Commit each accepted win with a descriptive message.

---

## 1. Where the score actually lives (don't waste effort elsewhere)

From `analyze_results.py` on the 2.4658 result:

| Band | Score share | avg cost | avg HPWL gap | avg area gap | avg soft |
|------|-------------|----------|--------------|--------------|----------|
| 21–80  | ~4%   | ~2.7 | ~1.2 | ~1.1 | 0.13 |
| 81–100 | ~17%  | 2.73 | 1.29 | 1.06 | 0.11 |
| **101–120** | **~79%** | **2.49** | **1.17** | **0.87** | **0.10** |

Weighting is `exp(n/12)` normalized; **n≥100 is 79% of the score, n≥116 alone is ~34%.**

Sensitivity (per the analyzer): uniform **HPWL −0.1 ⇒ −0.0615 score**, **area −0.1 ⇒ −0.0615**, **soft −0.01 ⇒ −0.049**. Soft is already near its floor (~0.10) → **HPWL and dead-space (area) on n≥100 are the only levers that move the score.** Do not spend time on small-n cases or on soft-violation reduction.

Exact contest cost (replicate this; it's the objective): from `iccad2026_evaluate.py:306` with `ALPHA=0.5, BETA=2.0, GAMMA=0.3`:
```
cost = (1 + 0.5*(max(0,hpwl_gap) + max(0,area_gap))) * exp(2*V_rel) * max(0.7, R^0.3)
hpwl_gap = (hpwl - hpwl_baseline)/hpwl_baseline      # clamped >=0
area_gap = (bbox_area - area_baseline)/area_baseline  # clamped >=0
V_rel    = soft_violations / n_soft                   # see evaluator:432-549
```
- `n_soft` = (#boundary blocks) + Σ(|mib_group|−1) + Σ(|cluster_group|−1)  (evaluator:459-471)
- `soft_violations` = boundary-misses + Σ(connected_components−1 per cluster) + Σ(distinct (w,h)−1 per MIB group)
- Locally `R` (RuntimeFactor) is forced to 1.0, so the runtime term is 1.0 locally. **On the real leaderboard `R = your_runtime / cross-submission_median`, uncapped above.** Keep per-case wall time low.
- Baselines per block-count are in `results/_baselines.json`, loaded in `solve()` into `self._hpwl_baseline`/`self._area_baseline`.

---

## 2. Hardware & rules (from contest QA — exploit these)

- **A100 80GB GPU + 48-core Icelake CPU + 128GB RAM.** Test cases evaluated **sequentially**, but **per-sample multiprocessing/multithreading is explicitly ALLOWED** (Q3). → Parallel multi-start across 48 cores is legal and wall-clock-cheap. THIS IS THE BIGGEST UNUSED LEVER and is absent from prior sprints.
- **Area tolerance ±1% symmetric, HARD** (Q6): `|w*h − target|/target ≤ 0.01`. Reshaping aspect ratio within this band is allowed; growing/shrinking area beyond 1% makes the case infeasible (cost=10).
- **Preplaced = hard, boundary = soft** (Q5): never move a preplaced block to satisfy a soft constraint. A preplaced block that doesn't touch its boundary just eats a soft violation; that's correct and feasible.

---

## 3. Current architecture map (line numbers in `my_optimizer.py`)

```
solve()                              :37   entry; runs shelf-variant loop + ONE analytical path, picks min _selection_cost
  _layout_variants()                 (per-count tuned params — OVERFIT, see §7)
  _construct_layout()                :111  shelf path (degree-based ordering, refine passes, SA)
  _analytical_construct_layout()     :2677 QP centers -> contour pack -> boundary -> refine -> SA
    _analytical_global_placement()   :3065 QUADRATIC PLACEMENT via conjugate gradient (already exists!)
    _contour_pack_with_analytics()   :2797 legalizer actually used: cluster super-blocks, sorts by DEGREE/AREA (not analytical), contour lowest-y
  _selection_cost()                  :1986 PROXY: hpwl + 0.08*bbox + soft*area_scale*180  (NOT baseline-normalized)
  _soft_violation_count()            :1997 exact soft numerator (boundary + components-1 + distinct-1)
  _group_components()                :2040 edge-sharing connected components (matches evaluator semantics)
DEAD CODE (defined, NOT called):
  _analytical_legalize()             :3213 sorts movable by analytical x, contour place, min-displacement — but NO cluster super-blocks
  _analytical_compact()              :3267 compact-to-origin both axes
```

**Key insight (this is the whole game):** `_analytical_global_placement` already produces wirelength-optimal QP positions. But the legalizer actually in use (`_contour_pack_with_analytics`) re-orders units by **degree/area**, only using the analytical x as a soft target hint — so the QP solution is largely discarded. This is exactly why "Quadratic placement (CG)" only bought −0.13% in the log. **The lever is the legalizer, not the global placer.**

---

## 4. DEAD ENDS — do NOT re-try (already failed, per PLAN_EXECUTION_LOG.md)

- BFS ordering; multi-start SA *as previously implemented* (same convergence, 2× runtime); force-directed refinement (blocks can't move under overlap constraints); centroid sorting; position swaps of different-dim blocks; two-axis compaction (created overlaps).
- Replacing shelf packer with pure analytical ordering → 25–36 soft violations (cluster grouping broke).
- **Analytical x-ordering WITHOUT treating clusters as super-blocks** → 41–58 soft violations. (The fix in Phase 2 is to keep clusters contiguous via super-blocks.)
- Compaction toward origin → 11–25 boundary violations.
- **Real contest cost for variant selection (the existing `_layout_variants` loop) → REVERTED**, because those per-count params were tuned against the proxy `_selection_cost`. ⇒ Do NOT simply swap the proxy for true cost inside the existing tuned loop. True cost is only safe (a) for selecting among NEW portfolio members, or (b) AFTER the tuned table is stripped (§7).
- 200 relaxation sweeps (already converged at 50); SA relocation moves; SA time 3s→5s (no effect); quadratic+relaxation hybrid (CG solution disrupted).

---

## 5. THE PLAN — prioritized phases

> Build order matters. Do phases in order. Each is independently measurable and revertible.

### Phase A — Parallel multi-start portfolio (highest ROI, ~zero regression risk)

**Goal:** exploit the 48 cores + allowed multiprocessing. Generate K diverse candidate full-solutions per case, evaluate each with the TRUE contest cost, keep the best. The current solution is always a candidate, so the score can only improve or stay equal.

**Why it's safe w.r.t. the §4 dead-end:** this does NOT change the existing tuned variant loop's selector. Each portfolio member runs the existing pipeline (which internally still uses the proxy + tuned params), and we only use true cost to pick the *winner among finished candidates*. The tuned params keep doing what they were tuned for.

**Steps:**
1. Add a true-cost helper (for final selection only). Insert after `_selection_cost` (line ~1995):
   ```python
   def _n_soft(self, constraints, block_count):
       if constraints is None or constraints.dim() <= 1 or constraints.shape[1] < 1:
           return 0
       n = min(block_count, len(constraints)); ncols = constraints.shape[1]; s = 0
       if ncols > 4: s += int((constraints[:n, 4] != 0).sum().item())
       if ncols > 2:
           mib = constraints[:n, 2]
           for g in range(1, (int(mib.max().item())+1) if mib.numel() else 1):
               s += max(0, int((mib == g).sum().item()) - 1)
       if ncols > 3:
           cl = constraints[:n, 3]
           for g in range(1, (int(cl.max().item())+1) if cl.numel() else 1):
               s += max(0, int((cl == g).sum().item()) - 1)
       return s

   def _is_feasible(self, positions, constraints, area_targets):
       n = len(positions)
       for i in range(n):
           x1,y1,w1,h1 = positions[i]
           for j in range(i+1, n):
               x2,y2,w2,h2 = positions[j]
               if (min(x1+w1,x2+w2)-max(x1,x2) > 1e-6) and (min(y1+h1,y2+h2)-max(y1,y2) > 1e-6):
                   return False
       ncols = constraints.shape[1] if constraints is not None and constraints.dim() > 1 else 0
       for i in range(n):
           if ncols>0 and constraints[i,0]!=0: continue
           if ncols>1 and constraints[i,1]!=0: continue
           t = float(area_targets[i]) if i < len(area_targets) else 0.0
           if t <= 0: continue
           w,h = positions[i][2], positions[i][3]
           if abs(w*h - t)/t > 0.01 + 1e-9: return False
       return True

   def _true_contest_cost(self, positions, constraints, area_targets, b2b, p2b, pins_pos):
       if self._hpwl_baseline is None or self._area_baseline is None:
           return self._selection_cost(positions, constraints, area_targets, b2b, p2b, pins_pos)
       if not self._is_feasible(positions, constraints, area_targets):
           return 10.0
       hpwl = calculate_hpwl_b2b(positions, b2b) + calculate_hpwl_p2b(positions, p2b, pins_pos)
       bbox = calculate_bbox_area(positions)
       hg = max(0.0, (hpwl - self._hpwl_baseline)/max(self._hpwl_baseline,1e-6))
       ag = max(0.0, (bbox - self._area_baseline)/max(self._area_baseline,1e-6))
       v  = self._soft_violation_count(positions, constraints) / max(self._n_soft(constraints, len(positions)), 1)
       return (1.0 + 0.5*(hg+ag)) * math.exp(2.0*v)
   ```
2. Refactor `solve()` so the body that produces one full solution is a pure function of a **config dict** (e.g. `{row_factor, small_cluster, large_cluster, path: 'shelf'|'analytical', seed, sa_time}`). Call it `_solve_one(cfg, <case tensors>) -> positions`.
3. Build a portfolio of configs: the existing tuned variant(s) for this block count PLUS diversified ones (vary `row_factor` ±0.1/±0.2, both shelf and analytical paths, 2–3 SA seeds). Aim for ~16–32 configs for large n, fewer for small n.
4. Run them in parallel:
   ```python
   from concurrent.futures import ProcessPoolExecutor
   import os
   ```
   - Use a module-level worker function (picklable) or `fork` start method (Linux default — workers inherit the already-imported torch, cheap). Pool size = `min(len(cfgs), os.cpu_count() or 8)` but **cap at ~32** to leave headroom.
   - Each worker returns `(positions, true_cost)`. Pick min true_cost. Guard: if pool errors (pickling/torch-fork issue), fall back to a serial loop so the run never crashes.
   - **Tensors must reach workers.** Easiest: pass the small per-case tensors as args (they're modest), or stash them in a module global set before submission. Avoid passing `self` if it carries unpicklable state; pass only what `_solve_one` needs, or make the optimizer re-instantiate inside the worker.
5. **Runtime guard:** measure max single-case wall time after this change. If any large case exceeds ~2–3s wall, reduce portfolio size or per-member SA time. Parallelism should keep wall time ≈ one member's time, but fork/import overhead and 48-way contention can bite — verify empirically.

**Expected:** measurable score drop with no feasibility loss. Tag result `portfolio_v1`.

---

### Phase B — Ordering-preserving, cluster-aware legalization (the structural fix)

**Goal:** stop discarding the QP solution. Legalize the QP/analytical positions by **minimizing displacement** while (a) preserving the analytical left-to-right / bottom-up order and (b) keeping each cluster contiguous by treating it as a single super-block (this is what prevented the §4 "analytical x-ordering" failure).

This is an **Abacus/Tetris-style** row legalizer:
1. Start from `centers = _analytical_global_placement(...)` (already exists, line 3065).
2. Build **units**: each cluster (constraints col 3 > 0, group size ≥2) becomes one super-block via `_cluster_local_pack` (line 482) with combined (w,h) and a unit center = mean of member centers. Singletons are their own unit. (Mirror the unit-building already in `_contour_pack_with_analytics:2818-2836`, but DO NOT then sort by degree/area.)
3. **Sort units by analytical position** — primary `center_x`, then `center_y`. (Optionally assign units to rows by `center_y` first, Abacus-style, then sort within row by `center_x`.)
4. **Place by minimum displacement from the QP target**, not lowest-y. For each unit in order, find the (x,y) closest to its analytical target that doesn't overlap already-placed units/obstacles (preplaced blocks). A clean way: row-based Abacus — assign to nearest row band, then within the row place at `max(target_x, current_row_right_edge)`; periodically allow shifting the whole row's prefix to reduce total displacement.
5. Expand super-blocks back to member rects (apply `local` offsets, as in `_contour_pack_with_analytics:2888`).
6. Add this as a NEW analytical sub-path (e.g. `_abacus_construct_layout`) and register it as portfolio members in Phase A — selected by true cost, so it can never regress.

**Reference code to study/reuse:** the dead `_analytical_legalize` (line 3213) already does analytical-x sort + min-displacement contour placement — but it has **no cluster super-blocks** (that's its flaw, and exactly the §4 dead-end). Fix it by inserting the cluster-unit step, OR write fresh. Reuse `_cluster_local_pack` (482), `_overlaps_any`, `_make_boundary_cluster_units` (for perimeter clusters).

**Watch:** boundary blocks must still touch the final bbox edge. Either keep `_place_boundary_items` (2734) after legalization, or fold boundary blocks in as anchored units. Verify soft-violation count doesn't spike (compare per-case soft in the analyzer).

**Expected:** this is where HPWL gap on n≥100 should drop. Tag `abacus_v1`. If it beats baseline on the big band, this is the sprint's main win.

---

### Phase C — Aspect-ratio dead-space fitting (independent, attacks area_gap)

**Goal:** reduce `area_gap` (~0.87 on n≥100 = whitespace in the bbox) by reshaping soft blocks within the ±1% area budget to fill gaps, without growing area beyond tolerance.

1. New post-pass `_refine_aspect_to_fill(positions, dims, constraints, area_targets, ...)`.
2. For each soft block (not fixed/preplaced, not MIB-constrained unless the whole group reshapes together): the allowed shape set is any (w,h) with `|w*h − target|/target ≤ 0.01`. Try a few aspect ratios (e.g. scale w by {0.9, 0.95, 1.05, 1.1} and set h = target/w, clamped so area stays in-tolerance).
3. Accept a reshape only if: no new overlap, bbox area does not grow, and soft-violation count does not increase. Prefer reshapes that let the block touch its boundary edge or abut a cluster mate (turns a soft violation into a non-violation) OR that shrink the bbox extent.
4. **MIB groups:** all members must keep identical (w,h) — reshape the whole group together or skip.
5. Run it inside each portfolio member after legalization, before the final true-cost evaluation.

**Watch:** the ±1% check is HARD and symmetric — clamp carefully, test with `--verbose` that `area_violations=0`. Tag `aspect_v1`.

---

### Phase D — Strengthen the global placer (optional, if A–C land)

Only if A–C succeed and time remains. Options (pick one, measure):
- **Iterative QP + spreading (SimPL-style):** after QP, add pseudo-anchor nets pulling overlapping cells toward spread positions, re-solve CG, repeat 3–5×. Improves HPWL vs. a single QP solve, then legalize with Phase B. This is the principled path the log calls for.
- **Sequence-pair SA** for the mid bands as an alternate portfolio member (larger move space than the current swap/shift SA). Higher risk; gate hard on runtime.

Do NOT start D before A–C are measured and committed.

---

## 6. Validation protocol

- After EACH phase: run the full loop (§0), record `Total score` + `Feasible`, and run `analyze_results.py --top 20` to confirm the n≥100 band improved and soft didn't spike.
- Compare against `results/quadratic_v1.json` (= 2.4658 baseline) with `scripts/compare_results.py` if helpful.
- Keep a per-phase row in `PLAN_EXECUTION_LOG.md` (score, Δ, feasible, runtime sum/max, what changed).
- **Never commit a regression or a feasibility drop.** If a phase doesn't help, revert it and note why in the log (future agents must not re-try it — see §4).

---

## 7. De-overfitting (do this once a GENERAL placer beats ~2.52)

`_layout_variants` has bespoke params for ~50 block counts, fit to THESE 100 validation cases. The final ranking is a DIFFERENT hidden 100 (same 21–120 range). Overfit params will not transfer. **Once Phase B (a general placer) clears ~2.52 on its own, strip the per-count tuning** down to a small set of count-agnostic configs and re-verify the score holds. This protects the leaderboard score. Until then, leave the table (it's currently load-bearing).

Also: once true-cost selection is the norm and the tuned loop is stripped, the §4 "real cost for variant selection reverted" constraint no longer applies — you can then use `_true_contest_cost` everywhere.

---

## 8. Housekeeping

- Delete `results/summary.json` (184 bytes, stale, claims a bogus 1.50 — a footgun).
- Keep result JSONs tagged per phase (`portfolio_v1`, `abacus_v1`, `aspect_v1`, …).

---

## 9. What to send back to the planning agent after execution

Update `PLAN_EXECUTION_LOG.md` with, per phase attempted:
1. Score before/after, Δ%, Feasible N/100, runtime sum + max single-case.
2. Whether parallel multi-start actually held wall time flat (report max single-case wall time and pool size used).
3. For Phase B: did HPWL gap on the 101–120 band drop? By how much? Any soft-violation spike?
4. Any new dead ends discovered (add to a §4-style table).
5. The current `analyze_results.py --top 20` dump of the best result.

That lets the next plan target whatever the new bottleneck is.

---

## 10. Commands quick-reference

```bash
cd /home/ubuntu/EDA
# full eval
cp contest_solution/my_optimizer.py external/FloorSet/iccad2026contest/
cd external/FloorSet/iccad2026contest
PYTHONPATH=.. /usr/bin/python3 iccad2026_evaluate.py --evaluate my_optimizer.py --output /home/ubuntu/EDA/results/<tag>.json
# single case
PYTHONPATH=.. /usr/bin/python3 iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 99 --verbose
# analyze
cd /home/ubuntu/EDA && python3 scripts/analyze_results.py results/<tag>.json --top 20
# commit
git add -A && git commit -m "Sprint 4: <change> (score X -> Y)"
```
```
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```
