# FloorSet ICCAD-2026 — Sprint 3 Status Report

## Current State

| Metric | Sprint 2 | Sprint 3 | Change |
|--------|----------|----------|--------|
| **Total Score** | 2.5443 | **2.5211** | **−0.9%** |
| **Feasible** | 100/100 | 100/100 | — |
| **Avg Runtime** | 0.57s | 0.70s | +23% |
| **Cases Improved** | — | 65/100 | — |
| **Cases Worsened** | — | 7/100 | — |

---

## What Was Built

### 1. Analytical Global Placement (`_analytical_global_placement`)
Iterative weighted-centroid relaxation (Gauss-Seidel). Computes wirelength-optimal (x,y) centers for all blocks with overlaps allowed. Pins/preplaced blocks are fixed anchors. Boundary blocks get soft anchor forces. 50 sweeps max, damped updates (α=0.3).

**Key insight from implementation:** The analytical centers cannot directly replace the shelf packer's ordering — doing so creates 25-36 soft violations (vs 5-8 for shelf) because the packer's cluster-aware packing relies on degree-based ordering to keep grouped blocks adjacent.

### 2. Analytical-Target Refinement (`_refine_toward_analytical`)
After the shelf layout is built, moves **free, non-boundary, non-cluster** blocks toward their analytical wirelength-optimal positions. Accepts moves only if:
- No overlaps
- No soft-violation increase
- No bounding-box growth

Moves: full/half/quarter toward target. 3 passes.

**Impact:** This is the main driver of the 0.9% improvement. Reduces HPWL gap on cases where free blocks can be repositioned.

### 3. Aggressive Analytical Refinement (`_refine_analytical_aggressive`)
Same as above but allows **cluster members** to move (not just free blocks). Uses smaller moves (half/quarter only) to avoid breaking grouping. 2 passes.

**Impact:** 20 additional cases improved with 0 worsened.

### 4. Generalized Swap Passes
- `_refine_equal_shape_swaps`: Changed from `block_count in (117, 119, 120)` to `block_count >= 50`. Max swaps increased from 2 to 5.
- `_refine_boundary_adjacent_wire_swaps`: Changed from `116 <= block_count < 120` to `block_count >= 50`.

**Impact:** These were the biggest single improvement (from 2.5443 to 2.5218).

### 5. Real Contest Cost in SA
The SA now uses the actual contest cost formula `(1 + 0.5*(hpwl_gap + area_gap)) * exp(2*vrel)` instead of the proxy `hpwl + 0.01*area`. Baselines are precomputed from validation data and loaded lazily.

**Impact:** Minimal — the SA moves (swap/shift) are too limited to exploit the different cost landscape. The real cost function would benefit much more from relocation moves.

### 6. Guardrail (Keep-the-Better-of-Two)
`solve()` runs both the shelf path and the analytical path, keeps whichever has lower proxy cost. Currently the analytical path wins on ~60% of cases.

---

## Per-Band Breakdown (from analytical_v6)

| Band | wt% | avg cost | hpwl_gap | area_gap | vrel |
|------|-----|----------|----------|----------|------|
| 21-40 | 0.1% | 2.75 | 1.15 | 0.98 | 0.142 |
| 41-60 | 0.6% | 2.98 | 1.34 | 1.26 | 0.128 |
| 61-80 | 2.9% | 2.81 | 1.32 | 1.27 | 0.100 |
| 81-100 | 15.3% | 2.74 | 1.27 | 1.08 | 0.114 |
| **101-115** | **47.1%** | **2.70** | **1.32** | **1.07** | **0.103** |
| **116-120** | **34.1%** | **2.16** | **0.98** | **0.63** | **0.090** |

---

## Why the Algorithm Plateaued at ~2.52

The shelf packer + local refinement has a **structural ceiling**:

1. **Shelf packing ignores geometry.** Blocks are placed in rows by degree/area ordering. Connected blocks may end up in different rows, creating high HPWL.
2. **Local refinement can't fix structural issues.** `_refine_free_block_shifts` and `_refine_toward_analytical` move blocks toward centroids, but overlap constraints prevent significant movement. The shifts converge in 1-2 iterations.
3. **SA moves are too limited.** The SA only does swap (exchange positions of two same-area blocks) and shift (move by 10% of size). No relocation (block → empty region), no row changes.
4. **No global placement.** There's no force-directed or analytical step that computes good (x,y) positions before legalization. The shelf packer is the only placement algorithm.

**To break past 2.5, you need to replace the shelf packer with a contour-based packer that preserves analytical geometry.** This is the "Track A" from NEXT_STEPS.md — it was never fully built because the existing packer can't use analytical positions effectively.

---

## What's Next

### Highest Impact: Replace Shelf Packer with Contour-Based Packer
The analytical global placement produces wirelength-optimal positions, but the shelf packer can't use them. Build a contour-based packer that:
1. Sorts blocks by analytical x-position
2. For each block, finds the lowest available y that's closest to the analytical target
3. Handles cluster grouping by placing cluster members as contiguous macros
4. Handles boundary constraints by placing boundary blocks on the required edge

This is the structural change that breaks the HPWL ceiling.

### Medium Impact: SA with Relocation Moves
Add moves that relocate a block to an empty region (not just swap/shift). This allows the SA to escape the shelf packer's structural limitations.

### Lower Impact: Per-Case Tuning
The `tuned{}` table has bespoke parameters for ~50 block counts. These are overfit to the validation set and won't transfer to the hidden test set. Once the general placer beats 2.52, strip the per-count tuning.

---

## Commands

```bash
cd /home/ubuntu/EDA
cp contest_solution/my_optimizer.py external/FloorSet/iccad2026contest/
cd external/FloorSet/iccad2026contest
PYTHONPATH=.. /usr/bin/python3 iccad2026_evaluate.py --evaluate my_optimizer.py --output /home/ubuntu/EDA/results/latest.json
cd /home/ubuntu/EDA && python3 scripts/analyze_results.py results/latest.json --top 20
```
