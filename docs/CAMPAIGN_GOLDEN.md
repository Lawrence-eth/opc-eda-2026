# CAMPAIGN: GOLDEN — the active plan to a winning score

**Standing orders (operator, 2026-07-07):** the only goal is to win. No time
limit. Full rebuilds allowed. The verified submission package is the safe
floor and must never regress; everything new integrates behind per-case
best-of gates on exact contest cost.

**Status (2026-07-07, round 3): official 1.8074, 100/100, rt avg 0.18s.
radj@1s 1.353 vs v9's 2.110 (−36%). G3 V_rel gate met (n≥100 vr 0.110).
Trajectory: 2.7182 → 2.1204 → 1.9573 → 1.9352 → 1.8975 → 1.8074.
Open leads: ag 0.25 (band free-width heights; case 70 outlier), hg 0.86
(ordering), util 0.79→0.90, cluster 58 / MIB 117 residuals, preplaced-with-
boundary codes are inherent violations (golden pays them too). Package
refreshed after each engine change (§G5).**
Every milestone below records its real, measured result when it lands —
nothing in this file is aspirational; if a number is missing, it hasn't run.

---

## 1. Where the score is (verified 2026-07-07, fresh environment)

Weighted decomposition of v9 (`results/repro_run1.json` == `v9_locked.json`):

| Scenario | RF=1 | at 0.7 floor |
|---|---|---|
| v9 as-is (hg 1.33, ag 1.03, V_rel 0.109) | 2.718 | 1.903 |
| v9 + area_gap→0 | 2.077 | 1.454 |
| v9 + hpwl_gap→0 | 1.887 | 1.321 |
| v9 + V_rel→0 | 2.180 | 1.526 |
| both gaps→0 (keep V_rel) | 1.246 | 0.872 |
| golden-equivalent (its own V_rel) | 1.108 | 0.776 |
| absolute bound | 1.000 | 0.700 |

The two quality gaps are worth ~1.0 runtime-adjusted; V_rel alone ~0.38.
Winning requires attacking hpwl_gap AND area_gap together without giving up
feasibility or (much) runtime.

## 2. Evidence that reframes the problem (new, 2026-07-07)

Mined from the golden data (validation labels + 1M training layouts):

1. **Golden layouts are near-perfect tessellations**: utilization ≈0.97,
   soft-block areas equal their targets **exactly** (not ±1%), all dims and
   coordinates are integers. The ~3% whitespace comes from a few "floating"
   blocks (golden y above the compacted contour), not from uniform slack.
2. **`tree_sol` in the training labels is a B\*-tree over blocks**
   (side 0 = child right-adjacent: x_c = x_p + w_p; side 1 = child stacked:
   x_c = x_p). Verified against golden coordinates: x-relations hold 100%.
   It appears to be *derived from* the layout, not the generative order —
   replaying it through standard contour packing reproduces x exactly but not
   every y (the floaters), and gives util 0.50–0.88, so **the tree alone is
   not the secret; the dims+layout structure is.**
3. **Retrieval is impossible**: all 1,008,000 training instances scanned — zero
   overlap with validation (area-multiset signature). The hidden set must be
   *solved*, not looked up. (`results/retrieval_scan.json`)
4. **Golden itself violates soft constraints** (90/100 validation cases,
   mean V_rel 0.051): perfect play ≈0.776 at the floor, not 0.70.
   (`results/golden_scored.json`)
5. Validation MIB groups have **equal area targets** (community finding,
   FloorSet issue #12, confirmed on validation): identical dims are exactly
   realizable. Training MIB groups are mostly broken (dataset packaging bug) —
   don't tune MIB handling on training data.

## 3. Why previous approaches lost (one sentence each)

Rigid near-square rectangles can't tile (shelf/skyline cap ~0.6 util);
free topological search (SP-SA) packs to 0.74–0.83 but shreds cluster
abutment/boundary (V_rel 0.24–0.71); constraining SP contiguity crushes util
(0.28–0.42); local search can't escape the shelf optimum; interior-only
replacement can't move the bbox (set by the frame); ML-per-block prediction
was packer-bound. Full detail: `MASTER_PLAYBOOK.md`, `PLAN_EXECUTION_LOG.md`.

## 4. The campaign: exact-area dissection engine

**Core idea:** stop packing rectangles; *dissect* the die. Build the layout as
a guillotine/slicing structure whose cut positions are chosen so every
soft-block leaf realizes its target area **exactly** (soft blocks have free
aspect — the evaluator only checks area). Then:

- **util ≈ 1.0 by construction** (area_gap ≈ 0 modulo fixed/preplaced slack) —
  the single biggest lever, never before exploited;
- **overlap-free by construction** (it's a partition of the plane);
- **clusters = contiguous subtrees ⇒ members tile a connected region ⇒
  abutment satisfied by construction** (the exact property every packer lost);
- **boundary blocks = peripheral leaves** of the dissection on their required
  side;
- **MIB groups**: equal target areas ⇒ assign identical (w,h) and place as
  siblings in a common strip;
- **fixed-shape blocks**: rigid leaves absorbed with local slack (golden also
  carries ~3% whitespace — budget exists);
- **preplaced blocks**: carve the dissection around their pinned rectangles
  (the leftover region splits into rectangles; dissect each);
- **HPWL**: choose the recursive partition by connectivity (min-cut style) and
  polish with SA over sibling swaps / subtree transplants / cut flips —
  wirelength-aware slicing is a classic, well-behaved search space.

Runtime: construction is O(n log n); SA is budgeted. Comfortably inside the
runtime floor even with the executable spawn overhead (~0.11s).

## 5. Milestones and gates (each committed; engine stays dormant until it wins)

- **G1 — Mine golden structure**: cluster-as-subtree conventions, boundary
  placement, preplaced embedding, aspect distributions, floater statistics.
  Artifact: `results/golden_structure.json` + findings recorded in §6.
- **G2 — Exact dissector** ✅ (2026-07-07): strip-of-rows exact-fill engine
  (`contest_solution/dissect.py`). Area gap collapsed: weighted ag 1.03 → ~0.2
  (util n≥100 ≈ 0.80 with fixed/preplaced slack; movable regions fill exactly).
  Note: gate was re-scoped from standalone util to measured area_gap, which is
  what actually scores.
- **G3 — Constraints** ◐ (2026-07-07): feasibility part PASSED (100/100 hard-
  feasible: exact-fill rows, obstacle carving, one-row bands, L/R row-end
  injection, cluster lanes, MIB slots). V_rel part OPEN: 0.176 vs target
  ≤0.109 (worst on small n: L/R queues exceed row count; some cluster lanes).
  util n≥100 0.80 vs 0.90 target (obstacle-segment remainders + fixed slack).
- **G4 — HPWL ordering + best-of integration** ✅ (2026-07-07): barycenter
  vertical ordering (b2b + pin anchors) + within-row x-pull; die width matched
  to obstacle-forced height; wf∈{0.85,1.0,1.15} portfolio; integrated into
  `my_optimizer.solve()` behind a feasibility-gated self-normalized selector
  (`_select_candidate`; gaps SIGNED when golden baselines absent — clamping
  against a non-golden reference censors improvements, found the hard way).
  GATE PASSED: official 2.1204 vs 2.7182; runtime-adjusted @{1,2,3}s =
  1.849/1.584/1.508 vs v9's 2.110/1.924/1.903; 93/100 improved; 51/51 tests.
- **G5 — Repackage + verify** ✅ (2026-07-07): dissect.py bundled; stub gained
  elementwise comparisons; wrapper+binary = **2.120411, 100/100, 0 position
  diffs**, avg 0.33s/case incl. spawn; fuzz 400/400. Package current.
- **G6 — Polish (open-ended, gated)**: SA over the dissection (sibling swaps,
  subtree transplants, strip re-partitions) under exact cost; aspect-bound
  tuning; floater insertion for fixed-block slack recovery; per-case portfolio
  (dissection vs v9 — already the integration form).
- **G7 — ML ordering (optional, needs GPU)**: learn the recursive partition
  order / strip assignment from the 1M golden trees (supervision exists!);
  inference is a permutation prior, decode stays exact. Revisit after G6.

## 6. G1 findings

(recorded here as they are measured — see `results/golden_structure.json`)

- Aspect ratios (prior mining, `PLAN_EXECUTION_LOG.md` Part IV): golden median
  1.45, p90 2.5, max 3.0; utilization 0.966–0.977.
- tree_sol semantics verified (§2.2). Remaining to mine: cluster-subtree
  contiguity rates, boundary-block tree positions, preplaced cut alignment,
  fixed-block slack absorption, floater statistics.

## 7. Progress log

- 2026-07-07: campaign opened. Evidence base gathered and verified (§2).
- 2026-07-07: dissection engine v2 built (`contest_solution/dissect.py`):
  exact-fill rows; frame = one-row bottom/top bands + L/R row-end injection;
  obstacle slabs; cluster lanes; MIB slots; barycenter ordering. Iterations:
  5.54 → 3.33 (frame) → 2.69 (bands/columns→rows) → 2.50 (die sizing to
  obstacle height, right-alignment) → 2.25 (barycenter + x-pull) → fixed two
  overlap bugs found by full-100 eval (retouch stale snapshot; band right-
  corner reservation) → dissect-only 2.2391 (88/100 wins, 100/100 feasible).
- 2026-07-07: integrated behind best-of gate → **official 2.1204, 100/100,
  avg 0.27s** (`results/integrated_v2.json`). Selector lesson recorded in G4.
- Open leads (G3/G6): V_rel 0.176→≤0.11 (small-n L/R overflow, cluster-lane
  edge blocks, T-misses after spill); hg 0.89→lower (ordering refinement, W
  portfolio widening, in-row iteration); util 0.80→0.90 (segment remainders).
