# FloorSet ICCAD-2026 — Detailed Status Report for Next Iteration

## Current State

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Total Score** | 2.6326 | 2.5425 | -3.4% |
| **Feasible** | 100/100 | 100/100 | — |
| **Avg Runtime** | ~0.35s | 0.30s | -14% |
| **Avg Cost** | 2.88 | 2.80 | -2.8% |

**Repo:** `https://github.com/Lawrence-eth/opc-eda-2026` (branch `main`, 6 commits ahead)

---

## Per-Band Breakdown (from `sprint1_v14.json`)

| Band | Cases | Avg Cost | Score Contribution | % of Total | Avg HPWL Gap | Avg Area Gap | Avg Soft |
|------|-------|----------|-------------------|------------|--------------|--------------|----------|
| 21-40 | 20 | 2.74 | 0.003 | 0.11% | 1.15 | 0.96 | 0.14 |
| 41-60 | 20 | 3.02 | 0.016 | 0.62% | 1.38 | 1.29 | 0.13 |
| 61-80 | 20 | 2.83 | 0.084 | 3.31% | 1.37 | 1.28 | 0.10 |
| 81-100 | 20 | 2.80 | 0.424 | 16.7% | 1.35 | 1.11 | 0.11 |
| **101-120** | **20** | **2.60** | **2.016** | **79.3%** | **1.27** | **0.96** | **0.10** |

**Key insight:** 101-120 band = 79.3% of total score. 81-100 band = 16.7%. Together = 96%.

---

## Top 10 Worst Weighted Cases

```
test_id  blocks  cost    hpwl_gap  area_gap  soft    runtime
99       120     2.045   0.841     0.682     0.075   0.80s
98       119     2.178   1.148     0.588     0.077   0.40s
97       118     2.168   0.800     0.634     0.117   0.55s
95       116     2.355   1.357     0.690     0.076   0.49s
89       110     3.804   1.771     1.646     0.170   1.74s  <- WORST
92       113     2.911   1.813     0.885     0.107   0.25s
96       117     2.055   0.804     0.613     0.092   0.42s
93       114     2.560   1.100     1.257     0.081   0.25s
94       115     2.163   1.177     0.767     0.046   0.33s
90       111     2.641   1.141     0.761     0.152   0.29s
```

**Dominant quality issue: HPWL** (avg 1.27 on top cases). Area is secondary (avg 0.96). Soft violations are small (avg 0.10).

---

## What Changed (6 commits)

### Commit 1: Tuple conversion for ALL block counts

```python
# Before (line 36-41):
if block_count >= 100:
    b2b_edges = self._b2b_edges(b2b_connectivity)
    p2b_edges = self._p2b_edges(p2b_connectivity)
else:
    b2b_edges = b2b_connectivity  # raw torch tensor!
    p2b_edges = p2b_connectivity  # raw torch tensor!

# After:
b2b_edges = self._b2b_edges(b2b_connectivity)  # always tuples
p2b_edges = self._p2b_edges(p2b_connectivity)  # always tuples
```

**Why it helps:** The 80-99 band was iterating torch tensors with `float()` per element in every refine pass (~100x slower). Now uses lightweight `(int, int, float)` tuples.

**Impact:** Runtime on 80-99 band dropped significantly. Score unchanged.

### Commit 2: Extended refine passes to 80-99 band

```python
# Before: only ran for block_count >= 100
# After: runs for block_count >= 80
if block_count < 100:
    self._refine_group_translations(...)
    self._refine_free_block_shifts(...)
    self._refine_boundary_edge_inward_compactions(...)
    self._refine_equal_shape_swaps(...)
    self._refine_boundary_adjacent_wire_swaps(...)
    if block_count >= 80:
        self._refine_free_block_shifts(...)  # second pass
        self._refine_boundary_line_shifts_118(...)
```

**Why it helps:** The 80-99 band had cost 5-9 and was running no refinement at all. Now gets the same passes as 100+.

**Impact:** 80-99 band avg cost dropped from ~7 to ~2.8.

### Commit 3: Simplified 100+ refine paths

```python
# Before: per-count gating
if 116 <= block_count <= 120:
    # 7 passes
elif 110 <= block_count <= 115:
    # 6 passes
elif 100 <= block_count <= 109:
    # 6 passes

# After: all passes for all 100+
if block_count >= 100:
    self._refine_group_translations(...)
    self._refine_free_block_shifts(...)
    self._refine_boundary_edge_inward_compactions(...)  # was 110+
    self._refine_boundary_line_shifts_118(...)
    self._refine_equal_shape_swaps(...)
    self._refine_boundary_adjacent_wire_swaps(...)  # was 116+
    self._refine_boundary_line_shifts_118(...)  # second pass
    self._refine_free_block_shifts(...)  # second pass
```

**Why it helps:** 100-115 band (33% of score) was getting fewer refinement passes than 116-120. Now gets the full treatment.

**Impact:** 100-115 band avg cost improved.

### Commit 4: SA budget bounding

```python
# Before:
max_sa_time = min(8.0, max(2.0, block_count * 0.05))
# n=99 -> 4.95s, n=120 -> 6.0s

# After:
max_sa_time = min(3.0, max(1.0, block_count * 0.02))
# n=99 -> 1.98s, n=120 -> 2.4s
```

**Why it helps:** SA was consuming 3-6s per case on 80-99 band. The SA optimizes a proxy cost (`hpwl + 0.01*area`), not the real contest cost, so long SA runs don't help much.

**Impact:** Runtime reduced, score unchanged.

### Commit 5: Interior obstacles for 80+

```python
# Before:
if (block_count in (109, 111, 113, 114, 115) or block_count >= 116) and placed_rects:
    interior_obstacles = placed_rects

# After:
if block_count >= 80 and placed_rects:
    interior_obstacles = placed_rects
```

**Why it helps:** Interior packer now avoids preplaced blocks for all 80+ cases, not just specific counts.

---

## What Didn't Work (tested and reverted)

| Approach | Score | Why it failed |
|----------|-------|---------------|
| **BFS ordering** | 2.60 (worse) | Disrupted degree-based ordering optimized for shelf packer |
| **Multi-start SA** | 2.54 (same) | SA converges to same local optimum regardless of seed; doubled runtime |
| **Force-directed refinement** | 2.54 (same) | Blocks couldn't move due to overlap constraints; converged in 1-2 iterations |
| **Centroid sorting** | 2.54 (same) | Shelf packer's row structure dominates; ordering doesn't matter |
| **Real contest cost function** | 2.55 (worse) | Variant selection was tuned for proxy cost; different landscape |
| **Position swaps** | 2.94 (much worse) | Swapping positions of different-dimension blocks creates bad layouts |
| **Tuned variants for 102/106/107/108** | 2.55 (worse) | Search picked worse variants; overfitting |
| **Two-axis compaction** | 5.15 (much worse) | Created overlaps and violated soft constraints |

---

## Root Cause: Why the Algorithm Plateaued at ~2.54

The shelf packer + local refinement has a fundamental limitation:

1. **Shelf packing ignores geometry.** Blocks are placed in rows by degree/area ordering. Connected blocks may end up in different rows, creating high HPWL.

2. **Local refinement can't fix structural issues.** The `_refine_free_block_shifts` moves blocks toward their connectivity-weighted centroids, but overlap constraints prevent significant movement. The shifts converge in 1-2 iterations.

3. **SA optimizes the wrong objective.** The SA uses `hpwl + 0.01*area` as cost, but the contest uses `(1 + 0.5*(hpwl_gap + area_gap)) * exp(2*v_rel)`. The SA also starts from the constructive layout and can only make small moves.

4. **No global placement.** There's no force-directed or analytical step that computes good (x,y) positions from connectivity before legalization. The shelf packer is the only placement algorithm.

---

## Remaining Opportunities (for next iteration)

### Highest Impact: Analytical Placement -> Legalize (Section 3.1)

Replace the shelf packer initial placement with:

1. **Force-directed global placement:** Compute (x,y) centroids from b2b + p2b connectivity using iterative weighted averaging (Jacobi/Gauss-Seidel). Pins act as fixed anchors.
2. **Legalization:** Sort blocks by x-position, pack using contour-based algorithm that respects the analytical positions (minimal displacement).
3. **Compaction:** Push everything toward origin to shrink bbox area.
4. **Detailed SA:** Use the real contest cost function with strict time budget.

This directly attacks HPWL (the dominant quality issue) by placing connected blocks near each other.

### Medium Impact: SA on Real Contest Objective (Section 3.2)

Replace the SA's proxy cost with the actual contest formula:

```python
# Current proxy:
cost = hpwl + 0.01 * area

# Real contest cost (need baseline metrics):
cost = (1 + 0.5 * max(0, hpwl_gap) + 0.5 * max(0, area_gap)) * exp(2 * v_rel)
```

The baseline metrics can be precomputed from the validation data.

### Lower Impact: Per-Case Tuning

Add tuned variants for the worst cases:

- test_id=89 (110 blocks, cost 3.80) -- worst weighted case
- test_id=83 (104 blocks, cost 3.31)
- test_id=86 (107 blocks, cost 3.12)
- test_id=92 (113 blocks, cost 2.91)

---

## Code Architecture

```
my_optimizer.py (2559 lines)
+-- MyOptimizer(FloorplanOptimizer)
|   +-- solve()                           # entry point, converts edges to tuples, runs variants
|   +-- _construct_layout()               # main placement pipeline
|   |   +-- _choose_dimensions()          # handle fixed/preplaced/MIB dims
|   |   +-- _pack_interior_units()        # shelf pack interior blocks
|   |   +-- _place_boundary_items()       # perimeter frame for boundary blocks
|   |   +-- _refine_*()                   # 8 refinement passes
|   |   +-- _compact_both_axes()          # (added but reverted - caused overlaps)
|   |   +-- _sa_post_optimization()       # simulated annealing
|   +-- _layout_variants()                # tuned parameters per block count
|   +-- _selection_cost()                 # proxy cost for variant selection
+-- Helper methods (1500+ lines of refinement passes)
    +-- _refine_group_translations()      # move group components to abut
    +-- _refine_free_block_shifts()       # move blocks toward connectivity centroids
    +-- _refine_boundary_edge_inward_compactions()  # compact boundary blocks inward
    +-- _refine_boundary_line_shifts_118()  # shift boundary lines
    +-- _refine_equal_shape_swaps()       # swap same-shape blocks' positions
    +-- _refine_boundary_adjacent_wire_swaps()  # swap adjacent boundary blocks
    +-- _refine_top_boundary_compaction()  # compact top boundary down
    +-- _sa_post_optimization()           # SA with shift moves
```

---

## Scoring Formula Reference

```
Cost = (1 + 0.5 * max(0, hpwl_gap) + 0.5 * max(0, area_gap)) * exp(2 * V_rel) * max(0.7, RuntimeFactor^0.3)
     = 10.0 if infeasible

Total Score = SUM(Cost[i] * exp(n_i / 12)) / SUM(exp(n_j / 12))
```

- `hpwl_gap`, `area_gap`: relative gaps vs baseline, clamped >= 0
- `V_rel` in [0, 1]: normalized soft violations (boundary + grouping + MIB)
- `RuntimeFactor`: your_runtime / median_runtime (local eval = 1.0)
- `exp(n/12)` weighting: n=116-120 = 34% of total, n=101-115 = 33%, n=81-100 = 15%
- Lower is better. Perfect = 1.0, max speed bonus = 0.70, infeasible = 10.0

---

## Constraint Reference

**Hard constraints** (violation -> Cost = 10.0):

1. No block overlaps (touching edges OK, tolerance 1e-6)
2. Soft-block area: `|w*h - target| / target <= 0.01` (symmetric)
3. Fixed-shape blocks: exact (w,h) from input (tolerance 1e-4)
4. Preplaced blocks: exact (x,y,w,h) from input (tolerance 1e-4)

**Soft constraints** (penalized via `exp(2 * V_rel)`):

1. **Boundary:** Block must touch specified bounding-box edge/corner
   - Bitmask: 1=left, 2=right, 4=top, 8=bottom
   - Corners: 5=top-left, 6=top-right, 9=bottom-left, 10=bottom-right
2. **Grouping:** Blocks in same cluster must abut (share edge)
3. **MIB:** Blocks in same MIB group must have identical (w,h)

**Constraint tensor columns:** [fixed, preplaced, mib_id, cluster_id, boundary_bitmask]

---

## Evaluation Commands

```bash
cd /home/ubuntu/EDA

# Copy optimizer to FloorSet
cp contest_solution/my_optimizer.py external/FloorSet/iccad2026contest/

# Run full evaluation
cd external/FloorSet/iccad2026contest
PYTHONPATH=.. /home/ubuntu/EDA/.venv/bin/python3 iccad2026_evaluate.py \
    --evaluate my_optimizer.py --verbose \
    --output /home/ubuntu/EDA/results/latest.json

# Run single case (for debugging)
PYTHONPATH=.. /home/ubuntu/EDA/.venv/bin/python3 iccad2026_evaluate.py \
    --evaluate my_optimizer.py --test-id 72 --verbose

# Validate submission format
PYTHONPATH=.. /home/ubuntu/EDA/.venv/bin/python3 iccad2026_evaluate.py \
    --validate my_optimizer.py

# Analyze results
cd /home/ubuntu/EDA
python3 scripts/analyze_results.py results/latest.json --top 20

# Push to GitHub
cd /home/ubuntu/EDA
git add -A && git commit -m "message" && git push origin main
```

---

## Summary for Next Claude Session

**Goal:** Reduce score from 2.5425 toward 2.0 or lower.

**Biggest lever:** 101-120 band (79% of score, avg cost 2.60, avg HPWL 1.27).

**What to try:**

1. Force-directed global placement -> contour legalization -> compaction
2. SA on real contest cost function (not proxy)
3. Multi-variant search with real cost function for selection

**What NOT to try (already tested, hurt):**

- BFS ordering, centroid sorting, force-directed refinement, position swaps, compaction, multi-start SA, real cost function for variant selection

**Key constraint:** Shelf packer is the bottleneck. Any improvement must either replace it or work around its limitations.
