# FloorSet ICCAD-2026 — Plan Execution Log

**Started:** Sprint 1 (pre-existing agent work)  
**Latest update:** Sprint 3 (Opus 4.8 session)  
**Current score:** **2.4690** (verified, internally consistent)  
**Target:** ≤ 1.5 (winning-tier)

---

## Score Trajectory

| Sprint | Score | Δ | Feasible | Runtime (sum/max) | Key change |
|--------|-------|---|----------|--------------------|------------|
| **Baseline** (tuned52) | 2.6326 | — | 100/100 | 102.8s / 14.6s | Original optimizer, per-count tuned |
| **Sprint 1-2** (sprint2_v6) | 2.5443 | −3.4% | 100/100 | 60.5s / 3.3s | Tuple conversion, refine pass extension, SA bounding |
| **Sprint 3** (analytical_v2) | 2.5289 | −0.6% | 100/100 | 53.0s / 2.4s | Analytical-target refinement pass |
| **Sprint 3** (analytical_v5) | 2.5218 | −0.3% | 100/100 | 68.5s / 3.4s | Generalized swap passes to all counts |
| **Sprint 3** (analytical_v6) | 2.5211 | −0.03% | 100/100 | 70.2s / 3.5s | Aggressive analytical refinement |
| **Sprint 3** (contour_v1) | **2.4690** | **−2.1%** | 100/100 | 66.9s / 3.2s | **Contour-based packer with analytical ordering** |
| Sprint 3 (contour_v5) | 2.4690 | 0% | 100/100 | 65.0s / 3.2s | SA relocation moves, increased SA budget (no effect) |
| Sprint 3 (quadratic_v1) | 2.4658 | −0.13% | 100/100 | 69.0s / 3.2s | Quadratic placement via conjugate gradient |
| **Sprint 4** (portfolio_v1) | 2.3977 | **−2.8%** | 100/100 | 313.8s / 14.7s | **Parallel multi-start portfolio** |
| Sprint 4 (portfolio_v4) | 2.4030 | +0.2% | 100/100 | 231.4s / 10.2s | Abacus-style legalization in portfolio |
| Sprint 4 (portfolio_v5) | 2.4029 | −0.004% | 100/100 | 243s / 10.2s | Aspect-ratio dead-space fitting (no effect) |
| Sprint 4 (portfolio_v7) | **2.4167** | +0.6% | 100/100 | 278s / 10.2s | **De-overfitted per-count tuning (§7)** |

**Total improvement (from original):** 2.6326 → 2.4167 = **−8.2%** (validation)
**Best validation score:** 2.4029 (portfolio_v5, with per-count tuning)
**De-overfitted score:** 2.4167 (portfolio_v7, should transfer better to hidden test)

---

## Verified Score Decomposition (contour_v1, exp(n/12) weighting)

| Band | Weight | Avg Cost | HPWL Gap | Area Gap | V_rel |
|------|--------|----------|----------|----------|-------|
| 21–40 | 0.1% | 2.66 | 1.08 | 0.92 | 0.142 |
| 41–60 | 0.5% | 2.85 | 1.27 | 1.11 | 0.130 |
| 61–80 | 2.9% | 2.79 | 1.29 | 1.24 | 0.103 |
| 81–100 | 15.3% | 2.74 | 1.29 | 1.07 | 0.112 |
| **101–115** | **47.0%** | **2.61** | **1.24** | **0.95** | **0.109** |
| **116–120** | **34.1%** | **2.15** | **0.98** | **0.63** | **0.087** |

**Weighted quality factor:** 2.0052 (81% of score)  
**Weighted soft factor:** 1.2301 (19% of score)  
**Dominant cost driver:** HPWL gap (~1.24 on 101-115 band, 47% of total weight)

---

## Top 15 Weighted Cases (analytical_v6)

| Contrib% | ID | n | Cost | HPWL Gap | Area Gap | V_rel | Runtime |
|----------|----|---|------|----------|----------|-------|---------|
| 16.35% | 99 | 120 | 2.04 | 0.84 | 0.68 | 0.075 | 2.2s |
| 16.03% | 98 | 119 | 2.18 | 1.15 | 0.59 | 0.077 | 1.0s |
| 14.70% | 97 | 118 | 2.17 | 0.80 | 0.64 | 0.117 | 1.6s |
| 13.56% | 95 | 116 | 2.37 | 1.31 | 0.63 | 0.091 | 1.8s |
| 13.14% | 89 | 110 | 3.78 | 1.74 | 1.65 | 0.170 | 3.5s |
| 12.94% | 92 | 113 | 2.90 | 1.80 | 0.89 | 0.107 | 1.0s |
| 12.80% | 96 | 117 | 2.05 | 0.80 | 0.61 | 0.092 | 1.0s |
| 12.38% | 93 | 114 | 2.55 | 1.09 | 1.26 | 0.081 | 0.8s |
| 11.40% | 94 | 115 | 2.16 | 1.18 | 0.77 | 0.046 | 1.1s |
| 9.94% | 90 | 111 | 2.63 | 1.13 | 0.76 | 0.152 | 0.7s |
| 9.83% | 91 | 112 | 2.39 | 1.06 | 0.70 | 0.121 | 1.1s |
| 8.74% | 88 | 109 | 2.73 | 1.13 | 0.98 | 0.143 | 0.8s |
| 8.44% | 86 | 107 | 3.12 | 1.76 | 1.25 | 0.109 | 2.9s |
| 8.17% | 87 | 108 | 2.78 | 1.37 | 0.84 | 0.140 | 2.0s |
| 6.40% | 85 | 106 | 2.57 | 1.19 | 1.20 | 0.078 | 2.3s |

**Worst case:** test_id=89 (n=110, cost=3.78, hpwl_gap=1.74, area_gap=1.65). This single case is 13.14% of the total score.

---

## What Was Tried — Detailed Log

### Sprint 1-2 (Previous Agent)

| Approach | Score | Verdict | Notes |
|----------|-------|---------|-------|
| Tuple conversion for ALL block counts | 2.5443 | ✅ Helped | 80-99 band was iterating torch tensors with float() per element. Now uses lightweight tuples. |
| Extend refine passes to 80-99 band | 2.5443 | ✅ Helped | 80-99 band had cost 5-9, now ~2.8. |
| Simplify 100+ refine paths | 2.5443 | ✅ Helped | All 100+ cases now get full treatment. |
| SA budget bounding | 2.5443 | ✅ Helped | Reduced from min(8, max(2, n*0.05)) to min(3, max(1, n*0.02)). |
| Interior obstacles for 80+ | 2.5443 | ✅ Helped | Interior packer now avoids preplaced blocks. |
| BFS ordering | — | ❌ Reverted | Disrupted degree-based ordering. |
| Multi-start SA | — | ❌ Reverted | Same convergence, doubled runtime. |
| Force-directed refinement | — | ❌ Reverted | Blocks couldn't move due to overlap constraints. |
| Centroid sorting | — | ❌ Reverted | Shelf packer's row structure dominates. |
| Real contest cost for variant selection | — | ❌ Reverted | Variant selection was tuned for proxy cost. |
| Position swaps | — | ❌ Reverted | Swapping different-dimension blocks creates bad layouts. |
| Two-axis compaction | — | ❌ Reverted | Created overlaps, cost 5.15. |

### Sprint 3 (This Session)

| Approach | Score | Verdict | Notes |
|----------|-------|---------|-------|
| Analytical global placement (centroid relaxation) | 2.5289 | ✅ Helped | Computes wirelength-optimal positions, but only used as targets for refinement. |
| Analytical-target refinement pass | 2.5289 | ✅ Helped | Moves free blocks toward analytical centers. Main driver of 2.5443→2.5289. |
| Replacing shelf packer with analytical ordering | — | ❌ Failed | Created 25-36 soft violations (vs 5-8 for shelf). Cluster grouping broken. |
| Compaction toward origin | — | ❌ Failed | Created 11-25 boundary violations. |
| Generalized equal-shape swaps (all ≥50) | 2.5218 | ✅ Helped | Was only 117/119/120, now all ≥50. Max swaps 2→5. |
| Generalized boundary wire swaps (all ≥50) | 2.5218 | ✅ Helped | Was only 116-119, now all ≥50. |
| Aggressive analytical refinement (cluster members) | 2.5211 | ✅ Helped | 20 more cases improved, 0 worsened. |
| Real contest cost in SA | 2.5211 | ≈ Neutral | SA moves too limited to exploit different cost landscape. |
| SA temperature normalization | 2.5211 | ≈ Neutral | Normalized delta by current cost for real-cost SA. |
| **Contour-based packer with analytical ordering** | **2.4690** | **✅ Helped** | **−2.1%. Same degree/area ordering as shelf (preserves grouping) but contour placement finds lowest y at analytical x. 80 improved, 6 worsened.** |
| Unbounded contour packer (analytical x-ordering) | — | ❌ Failed | 41-58 soft violations. Analytical x-ordering breaks cluster grouping. |

---

## Why the Algorithm Plateaued at ~2.47

The contour packer + refinement has a **structural ceiling**:

1. **Contour packer helps but can't fix analytical ordering.** The contour packer finds the lowest y at the analytical x-position, but uses degree/area ordering (not analytical ordering) to preserve grouping. This means connected blocks may still be placed far apart if they have different degrees.

2. **Local refinement converges in 1-2 iterations.** `_refine_free_block_shifts` and `_refine_toward_analytical` move blocks toward centroids, but overlap constraints prevent significant movement.

3. **SA moves are too limited.** Only swap (exchange positions of same-area blocks) and shift (move by 10% of size). No relocation (block → empty region), no row changes.

4. **The 101-115 band is the bottleneck.** 47% of the score, avg cost 2.61, avg HPWL gap 1.24. The 116-120 band is already at 2.15 and can't improve much more.

---

## Architecture: Current Optimizer (3,123 lines, 87 methods)

```
MyOptimizer(FloorplanOptimizer)
├── solve()                                    # Entry point: guardrail (shelf vs analytical)
├── _construct_layout()                        # Shelf path (existing, degree-based ordering)
│   ├── _choose_dimensions()                   # Handle fixed/preplaced/MIB dims
│   ├── _pack_interior_units()                 # Shelf pack interior blocks (degree ordering)
│   ├── _place_boundary_items()                # Perimeter frame for boundary blocks
│   ├── _refine_*()                            # 8+ refinement passes
│   └── _sa_post_optimization()                # SA with swap/shift moves
├── _analytical_construct_layout()             # Analytical path (shelf + centroid relaxation)
│   ├── _analytical_global_placement()         # Gauss-Seidel centroid relaxation
│   ├── _pack_interior_units()                 # Same packer as shelf path
│   ├── _place_boundary_items()                # Same boundary handling
│   ├── _refine_toward_analytical()            # Move free blocks toward analytical centers
│   ├── _refine_analytical_aggressive()        # Move cluster members toward centers
│   ├── _refine_*()                            # Same refinement passes as shelf path
│   └── _sa_post_optimization()                # SA with real contest cost
├── _layout_variants()                         # Per-count tuned parameters
├── _selection_cost()                          # Proxy cost for variant selection
└── _soft_violation_count()                    # O(n²) soft violation counter
```

---

## What's Next — Remaining Levers (Status: Exhausted for Incremental Approach)

### What Was Tried (All Failed to Improve Beyond 2.4658)

| Approach | Result | Why It Failed |
|----------|--------|---------------|
| Analytical ordering for non-cluster blocks | 2.4891 (worse) | Disrupts packer's row structure |
| 200 relaxation sweeps | 2.4700 (same) | Already converged at 50 sweeps |
| SA relocation moves | 2.4690 (same) | Moves rejected — creates overlaps or increases cost |
| SA time budget increase (3s → 5s) | 2.4690 (same) | SA moves too limited to exploit more time |
| **Quadratic placement (CG)** | **2.4658** | **−0.13%. Marginal improvement. 17 improved, 23 worsened.** |
| Quadratic + centroid relaxation hybrid | 2.4722 (worse) | CG solution disrupted by relaxation |

### What Would Actually Break Past 2.47

The current approach (shelf/contour packer + refinement + SA) has hit a **structural ceiling**. To reach 1.5-1.8, you need a fundamentally different placement algorithm:

1. **Quadratic Placement** — Build the connectivity Laplacian, solve `Lx = b` per axis with conjugate gradient. This produces wirelength-optimal positions that respect the full graph structure (not just local centroids). Then legalize with the contour packer.

2. **Sequence-Pair SA** — Replace the shelf packer with a sequence-pair representation. SA over sequence-pairs can explore a much larger solution space (any non-overlapping placement) while the cost function directly optimizes the contest objective.

3. **Force-Directed with Legalization** — Proper force-directed placement (not just centroid relaxation) that considers repulsion forces between blocks to avoid overlaps, then legalize.

### Current Score Summary

| Metric | Value |
|--------|-------|
| **Total Score** | **2.4690** |
| Quality factor | 2.0052 |
| Soft factor | 1.2301 |
| HPWL gap (101-115) | 1.24 |
| Area gap (101-115) | 0.95 |
| 100/100 feasible | ✅ |
| Runtime sum | 66.9s |
| Max single case | 3.2s |

---

## Commands

```bash
cd /home/ubuntu/EDA

# Copy optimizer to FloorSet
cp contest_solution/my_optimizer.py external/FloorSet/iccad2026contest/

# Run full evaluation
cd external/FloorSet/iccad2026contest
PYTHONPATH=.. /usr/bin/python3 iccad2026_evaluate.py \
    --evaluate my_optimizer.py \
    --output /home/ubuntu/EDA/results/next.json

# Run single case (debug)
PYTHONPATH=.. /usr/bin/python3 iccad2026_evaluate.py \
    --evaluate my_optimizer.py --test-id 80 --verbose

# Analyze results
cd /home/ubuntu/EDA
python3 scripts/analyze_results.py results/next.json --top 20

# Commit
git add -A && git commit -m "message"
```

---

## Key Files

| File | Purpose |
|------|---------|
| `contest_solution/my_optimizer.py` | Optimizer (3,123 lines, 87 methods) |
| `results/analytical_v6.json` | Latest verified result (2.5211) |
| `results/sprint2_v6.json` | Sprint 2 baseline (2.5443) |
| `results/_baselines.json` | Precomputed hpwl/area baselines for all 100 cases |
| `NEXT_STEPS.md` | Original plan (partially executed) |
| `REPORT.md` | Sprint 1-2 report |
| `SPRINT3_REPORT.md` | Sprint 3 report |
| `PLAN_EXECUTION_LOG.md` | This file |

---

## Constraint Reference

**Hard constraints** (violation → cost 10.0):
1. No overlaps (touching edges OK, tolerance 1e-6)
2. Soft-block area: `|w*h - target| / target ≤ 0.01` (symmetric)
3. Fixed-shape blocks: exact (w,h) from input (tolerance 1e-4)
4. Preplaced blocks: exact (x,y,w,h) from input (tolerance 1e-4)

**Soft constraints** (penalized via `exp(2 * V_rel)`):
1. Boundary: block must touch specified bounding-box edge/corner (bitmask: 1=left, 2=right, 4=top, 8=bottom)
2. Grouping: blocks in same cluster must abut (share edge)
3. MIB: blocks in same MIB group must have identical (w,h)

**Constraint tensor columns:** [fixed, preplaced, mib_id, cluster_id, boundary_bitmask]

**Scoring formula:**
```
Cost = (1 + 0.5*(max(0, hpwl_gap) + max(0, area_gap))) * exp(2*V_rel) * max(0.7, RuntimeFactor^0.3)
     = 10.0 if infeasible

Total = Σ Cost[i] * exp(n_i/12) / Σ exp(n_j/12)
```

---

## Standing Warnings

- **Overfitting:** The `_layout_variants` `tuned{}` table has bespoke parameters for ~50 block counts. These are fit to THESE 100 validation cases. Final ranking is a DIFFERENT hidden 100 (same 21-120 range). Once the general placer beats 2.52, strip the per-count tuning.
- **Runtime is a live leaderboard risk.** Local eval forces RuntimeFactor=1.0; the real one is `your_rt / cross-submission_median`, penalty uncapped. Keep max per-case ≤ ~1-2s.
- **The stale summary.json** still exists in results/ (claims 1.50 for a 9.69 file). Delete it or regenerate from analytical_v6.
