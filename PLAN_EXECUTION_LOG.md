# FloorSet ICCAD-2026 — Plan Execution Log

**Started:** Sprint 1 (pre-existing agent work)  
**Latest update:** Sprint 3 (Opus 4.8 session)  
**Current score:** 2.5211 (verified, internally consistent)  
**Target:** ≤ 1.5 (winning-tier)

---

## Score Trajectory

| Sprint | Score | Δ | Feasible | Runtime (sum/max) | Key change |
|--------|-------|---|----------|--------------------|------------|
| **Baseline** (tuned52) | 2.6326 | — | 100/100 | 102.8s / 14.6s | Original optimizer, per-count tuned |
| **Sprint 1-2** (sprint2_v6) | 2.5443 | −3.4% | 100/100 | 60.5s / 3.3s | Tuple conversion, refine pass extension, SA bounding |
| **Sprint 3** (analytical_v2) | 2.5289 | −0.6% | 100/100 | 53.0s / 2.4s | Analytical-target refinement pass |
| **Sprint 3** (analytical_v5) | 2.5218 | −0.3% | 100/100 | 68.5s / 3.4s | Generalized swap passes to all counts |
| **Sprint 3** (analytical_v6) | **2.5211** | −0.03% | 100/100 | 70.2s / 3.5s | Aggressive analytical refinement |

**Total improvement:** 2.6326 → 2.5211 = **−4.2%**

---

## Verified Score Decomposition (analytical_v6, exp(n/12) weighting)

| Band | Weight | Avg Cost | HPWL Gap | Area Gap | V_rel |
|------|--------|----------|----------|----------|-------|
| 21–40 | 0.1% | 2.74 | 1.14 | 0.98 | 0.142 |
| 41–60 | 0.5% | 2.98 | 1.34 | 1.26 | 0.128 |
| 61–80 | 2.9% | 2.81 | 1.32 | 1.27 | 0.100 |
| 81–100 | 15.3% | 2.73 | 1.27 | 1.08 | 0.114 |
| **101–115** | **47.0%** | **2.70** | **1.32** | **1.07** | **0.103** |
| **116–120** | **34.1%** | **2.16** | **0.98** | **0.63** | **0.090** |

**Weighted quality factor:** 2.0531 (81% of score)  
**Weighted soft factor:** 1.2265 (19% of score)  
**Dominant cost driver:** HPWL gap (~1.3 on 101-115 band, 47% of total weight)

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

---

## Why the Algorithm Plateaued at ~2.52

The shelf packer + local refinement has a **structural ceiling** that cannot be broken by more refinement passes:

1. **Shelf packing ignores geometry.** Blocks are placed in rows by degree/area ordering. Connected blocks end up in different rows, creating high HPWL (~1.3 gap on 101-115 band).

2. **Local refinement converges in 1-2 iterations.** `_refine_free_block_shifts` and `_refine_toward_analytical` move blocks toward centroids, but overlap constraints prevent significant movement. The blocks are "stuck" in the shelf structure.

3. **Replacing the packer breaks soft constraints.** The shelf packer's degree-based ordering is essential for keeping cluster members adjacent (grouping constraint). Changing the ordering to analytical positions creates 25-36 soft violations.

4. **SA moves are too limited.** Only swap (exchange positions of same-area blocks) and shift (move by 10% of size). No relocation (block → empty region), no row changes.

5. **The 101-115 band is the bottleneck.** 47% of the score, avg cost 2.70, avg HPWL gap 1.32. The 116-120 band is already at 2.16 and can't improve much more.

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

## What's Next — The Remaining Lever

### Replace Shelf Packer with Contour-Based Packer (Target: 2.52 → ~1.5-1.8)

The analytical global placement produces wirelength-optimal positions, but the shelf packer can't use them without breaking grouping. The solution is a **contour-based packer** that:

1. **Builds cluster macros first** (same as shelf packer, preserves grouping)
2. **Sorts ALL units by analytical x-position** (not degree/area)
3. **Places each unit using contour tracking**: find the lowest y where the unit fits without overlapping, closest to the analytical target
4. **Handles boundary by pre-reserving edge space** before interior packing

This preserves grouping (cluster macros stay contiguous) while using analytical geometry for global ordering. The HPWL gap should drop from ~1.3 to ~0.5-0.7.

### Implementation Strategy

```python
def _contour_pack_with_analytics(self, interior, dims, constraints, area_targets,
                                   b2b_edges, p2b_edges, centers, start_x, start_y, obstacles):
    # 1. Build cluster macros (identical to _pack_interior_units)
    # 2. Sort units by analytical x-position (using centers)
    # 3. Contour-based placement:
    #    - Maintain a skyline (list of (x_end, y_top) segments)
    #    - For each unit, find lowest y where it fits
    #    - Place at that y, update skyline
    # 4. Return positions
```

### Risk Mitigation

- **Guardrail already exists**: `solve()` runs both shelf and analytical paths, keeps the better one. The contour packer replaces the packer in the analytical path only.
- **Fallback exists**: if the contour packer produces overlaps, the existing fallback shelf-packs from scratch.
- **Test on single case first**: before full evaluation, test on case 80 (n=101) and case 99 (n=120).

### Expected Impact

| Metric | Current | Target | Driver |
|--------|---------|--------|--------|
| HPWL gap (101-115) | 1.32 | 0.5-0.7 | Contour pack preserves analytical geometry |
| Area gap (101-115) | 1.07 | 0.6-0.8 | Compaction works better with contour layout |
| Soft violations | ~0.10 | ≤0.10 | Cluster macros preserved |
| **Total score** | **2.5211** | **~1.5-1.8** | Quality factor drops from 2.05 to ~1.45 |

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
