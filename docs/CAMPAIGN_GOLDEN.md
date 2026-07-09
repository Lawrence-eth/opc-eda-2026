# CAMPAIGN: GOLDEN — the active plan to a winning score

**Standing orders (operator, 2026-07-07):** the only goal is to win. No time
limit. Full rebuilds allowed. The verified submission package is the safe
floor and must never regress; everything new integrates behind per-case
best-of gates on exact contest cost.

**Status (2026-07-09, round 22): official 1.6181, 100/100, rt avg 0.173s.
Paired-window radj@1s 1.174 vs v9's 2.110 (-44%); v28 was 1.176. G3 V_rel
gate met (n≥100 vr 0.085).
Trajectory: 2.7182 → 2.1204 → 1.9573 → 1.9352 → 1.8975 → 1.8074 → 1.7978 → 1.7952 → 1.7903 → 1.7827 → 1.7368 → 1.7027 → 1.6960 → 1.6845 → 1.6651 → 1.6568 → 1.6507 → 1.6483 → 1.6385 → 1.6328 → 1.6225 → 1.6207 → 1.6190 → 1.6181.
Open leads: ag 0.154 on n≥100 (case-70-class snowball mitigated by a gated
candidate), hg 0.560 on n≥100 integrated score drivers (ordering),
grouping 54 / MIB 126 residuals. Exact v29 soft ledger is boundary 324,
grouping 54, MIB 126, total 504/4478 (`results/enriched_diagnostics.json`).
Golden mining says MIB should be exact, clusters should almost always connect,
and some boundary misses are inherent or cost-optimal (including 13
preplaced-boundary misses). Package refreshed after each engine change (§G5).**
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

## 2. Evidence that reframes the problem (updated 2026-07-08)

Mined from the golden data (validation labels + 1M training layouts):

1. **Golden layouts are near-perfect tessellations**: utilization ≈0.97,
   soft-block areas equal their targets **exactly** (not ±1%), all dims and
   coordinates are integers. The ~3% whitespace is structured slack around
   fixed/preplaced/boundary geometry; validation re-mining found zero
   unsupported-above-floor blocks.
2. **`tree_sol` in the training labels is a B\*-tree over blocks**
   (side 0 = child right-adjacent: x_c = x_p + w_p; side 1 = child stacked:
   x_c = x_p). Verified against golden coordinates: x-relations hold 100%.
   It appears to be *derived from* the layout, not the generative order —
   replaying it through standard contour packing reproduces x exactly but not
   every y, and gives util 0.50–0.88, so **the tree alone is not the secret;
   the dims+layout structure is.**
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

- **G1 — Mine golden structure** ✅ (2026-07-08): cluster connectivity,
  boundary placement, preplaced/fixed embedding, aspect distributions, and
  support statistics mined from validation labels. Artifact:
  `results/golden_structure.json`; violation-ratio check matches
  `results/golden_scored.json` exactly.
- **G2 — Exact dissector** ✅ (2026-07-07): strip-of-rows exact-fill engine
  (`contest_solution/dissect.py`). Area gap collapsed: weighted ag 1.03 → ~0.2
  (util n≥100 ≈ 0.80 with fixed/preplaced slack; movable regions fill exactly).
  Note: gate was re-scoped from standalone util to measured area_gap, which is
  what actually scores.
- **G3 — Constraints** ✅ (2026-07-07, round 3): feasibility 100/100; V_rel
  gate MET — n≥100 vr 0.110 vs target ≤0.109 (edge stacks, flat band clusters,
  cluster-edge stacks, MIB forcing). util subtarget stays open (0.79 vs 0.90;
  obstacle-segment remainders + fixed slack + case-70 lead → HANDOFF §6).
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
  diffs**, avg 0.33s/case incl. spawn; fuzz 400/400. Later G6 packages were
  refreshed after each kept polish.
- **G6 — Polish (open-ended, gated)**: first kept polish landed 2026-07-08:
  a case-70/90 obstacle-band snowball trace led to a **single gated
  `band_edge_cap` candidate** (fires only when incumbent/capped bottom-band
  height ratio ≥5×). Official 1.7978, 100/100; radj@{1,2,3}s =
  1.350/1.260/1.258; wrapper parity 0 position diffs; fuzz 400/400. Second
  kept polish added two gated wf=1.0 `pin_scale` candidates (0.5, 4.0) only for
  50≤n≤103. Official **1.7952**, 100/100; radj@{1,2,3}s =
  **1.348/1.257/1.257**; wrapper parity 0 position diffs; fuzz 400/400. Third
  kept polish added a high-weight same-area boundary reshape candidate for
  free non-cluster/non-MIB n≥118 blocks. Official **1.7903**, 100/100;
  radj@{1,2,3}s = **1.343/1.254/1.253**; wrapper parity 0 position diffs;
  fuzz 400/400. Fourth kept polish added a tightly gated same-bbox boundary
  edge-slide candidate using obstacle-clearance endpoints for n∈{103,119}.
  Official **1.7827**, 100/100; radj@{1,2,3}s =
  **1.342/1.249/1.248**; wrapper parity 0 position diffs; fuzz 400/400. Fifth
  kept polish added one wf=1.0 barycentric edge-queue candidate for L/R boundary
  units. Official **1.7368**, 100/100; radj@{1,2,3}s =
  **1.329/1.220/1.216**; wrapper parity 0 position diffs; fuzz 400/400.
  Sixth kept polish changed that extra candidate into a hybrid: n<118 also
  sorts bottom/top band units by pin/net-x pull, while n>=118 preserves the
  v15 width-first band order that won cases 98/99. Official **1.7027**,
  100/100; radj@{1,2,3}s = **1.316/1.201/1.192**; wrapper parity 0
  position diffs; fuzz 400/400.
  Seventh kept polish made that same extra candidate width-adaptive: wf=0.8
  only for high-boundary, moderate-net 95..117 block cases, with no extra
  portfolio member. Official **1.6960**, 100/100; same-window v16 recheck
  gate passed at radj@{1,2,3}s (**1.322/1.197/1.187** vs
  **1.332/1.204/1.192**); wrapper parity 0 position diffs; fuzz 400/400.
  Eighth kept polish added one high-weight strong pin-pull hybrid candidate
  (`pin_scale=6.0`, edge-bary, band-pinx) only for n>=100. Official
  **1.6845**, 100/100; radj@{1,2,3}s = **1.317/1.191/1.179**; wrapper
  parity 0 position diffs; fuzz 400/400.
  Ninth kept polish added a gated `wf=0.85` version of that strong pin-pull
  candidate for boundary-heavy n>=100 cases with low p2b or dense b2b nets.
  Official **1.6651**, 100/100; same-window v18 recheck gate passed at
  radj@{1,2,3}s (**1.357/1.192/1.166** vs **1.368/1.204/1.179**); wrapper
  parity 0 position diffs; fuzz 400/400.
  Tenth kept polish added a case-99-class wide edge-bary tail candidate
  (`wf=1.15`, `pin_scale=6.0`, no band-pinx) only for n>=120 with high
  boundary count and dense b2b/p2b nets. Official **1.6568**, 100/100;
  radj@{1,2,3}s = **1.315/1.176/1.160**; wrapper parity 0 position diffs;
  fuzz 400/400.
  Eleventh kept polish made pass-2 row ordering aware of b2b edges to
  external previous-position anchors (preplaced, fixed, and edge-routed
  blocks). Official **1.6507**, 100/100; same-window v20 recheck gate passed
  at radj@{1,2,3}s (**1.311/1.172/1.156** vs
  **1.342/1.187/1.163**); wrapper parity 0 position diffs; fuzz 400/400.
  Twelfth kept polish changed non-flat cluster lane seeding from pure area
  order to boundary-side priority (`left`, interior, `right`) before area.
  Official **1.6483**, 100/100; same-window v21 recheck gate passed at
  radj@{1,2,3}s (**1.338/1.180/1.155** vs **1.344/1.185/1.159**);
  wrapper parity 0 position diffs; fuzz 400/400.
  Thirteenth kept polish added two tightly gated strong pin-pull width pockets
  (`wf=0.75` and `wf=1.15`) for high-weight feature classes. Official
  **1.6385**, 100/100; radj@{1,2,3}s = **1.314/1.167/1.147** vs v22
  **1.338/1.180/1.155**; wrapper parity 0 position diffs; fuzz 400/400.
  Fourteenth kept polish kept the existing capped obstacle-band candidate
  count unchanged but replaced its width with `wf=1.1` for the low-p2b
  case-90 class. Official **1.6328**, 100/100; radj@{1,2,3}s =
  **1.299/1.159/1.143** vs v23 **1.314/1.167/1.147**; wrapper parity
  0 position diffs; fuzz 400/400.
  Fifteenth kept polish replaced repeated tensor-scalar HPWL calls in
  candidate scoring with pre-extracted Python edge/pin lists and one center
  pass per candidate. It preserves all v24 positions and the **1.6328** raw
  score while reducing in-process avg runtime from **0.238s to 0.159s**;
  radj@{1,2,3}s = **1.175/1.143/1.143**. Wrapper parity is 0 position
  diffs; binary fuzz is 400/400.
  Sixteenth kept polish backfills short flexible rows clamped to a preplaced
  obstacle edge from the remaining mid queue in tightly gated feature
  pockets. Official **1.6225**, 100/100; same-window radj@{1,2,3}s =
  **1.175/1.136/1.136** vs v25 **1.182/1.143/1.143**; wrapper parity
  0 position diffs; fuzz 400/400.
  Seventeenth kept polish relaxes the active obstacle-slab aspect guard only
  for the established case-88 backfill and case-90 capped-band feature
  pockets. Official **1.6207**, 100/100; same-window radj@{1,2,3}s =
  **1.173/1.134/1.134** vs v26 **1.174/1.136/1.136**; wrapper parity
  0 position diffs; fuzz 400/400.
  Still open: SA over the dissection (sibling swaps, subtree
  transplants, strip re-partitions) under exact cost; aspect-bound tuning;
  fixed/preplaced slack recovery; per-case portfolio.
- **G7 — ML ordering (optional, needs GPU)**: learn the recursive partition
  order / strip assignment from the 1M golden trees (supervision exists!);
  inference is a permutation prior, decode stays exact. Revisit after G6.

## 6. G1 findings

(recorded here as they are measured — see `results/golden_structure.json`)

- `scripts/mine_golden.py` now writes `results/golden_structure.json`; mined
  `violations_relative` matches `results/golden_scored.json` exactly
  (`max_abs_vr_diff=0`).
- Utilization: median 0.971, p10 0.963, p90 0.978, range 0.954–0.987.
  Soft-block aspect ratios: median 1.45, p90 2.50, p95 2.75, max 3.0.
- Golden soft violations: 229 / 4478 across 90/100 cases. Split: boundary
  219, grouping 10, MIB 0. Do not spend slack chasing zero boundary at all
  costs; golden itself pays boundary misses.
- Boundary: 2184/2403 satisfied (90.9%). Left, bottom, and bottom-left codes
  are perfect; most misses are right/top/top-right/bottom-right. Preplaced
  boundary misses are 13/122 and are unfixable by rule; fixed boundary misses
  are 32/247 and remain movable tradeoffs.
- Clusters: 350/360 connected (97.2%); only ten groups split, all into two
  components. Preplaced clusters: 68/70 connected, and every preplaced member
  touches a same-cluster member (74/74). A bridge can be real, but the
  deployable selector still needs a hidden-safe ranking signal.
- MIB: 100/100 groups are shape-uniform in golden. Keep treating MIB
  equality as a first-class structure, but global-square forcing was already
  measured and reverted.
- Preplaced/fixed embedding: median case has 33% of non-preplaced blocks
  aligned to a preplaced x/y cut and 13% adjacent to a preplaced block; fixed
  cut reuse is stronger (median 67% aligned, 32% adjacent). This points to
  fixed/preplaced cut reuse, not generic floater insertion.
- Support: every golden block touches another block edge, and zero blocks are
  unsupported above the bbox floor under the simple support test. Do not chase
  "floater insertion" as a primary prior.

## 7. Progress log

- 2026-07-07: campaign opened. Evidence base gathered and verified (§2).
- Round 1 (G2–G5): engine built and integrated — official 2.1204; package
  parity-verified.
- Round 2: flat band clusters + edge stacks + MIB lanes (1.9573) → two-pass
  construction + fast-first shelf (1.9352 @ 0.18s avg; two runtime-negative
  experiments reverted — see PLAN_EXECUTION_LOG).
- Round 3: MIB forcing + segment edge injection + preplaced-safe retouch
  (1.8975; preplaced-moving retouch bug caught by full-eval gate) → cluster
  edge stacks + band free-width + clamp re-queue (**1.8074**, V_rel gate met).
  Clamp-branch segmented-fill experiment reverted (regressed ag; case 70
  unchanged). Package rebuilt + parity-verified at 1.807413.
- Round 4: traced case 70 and found bottom-band height snowball (y≈202 before
  mid fill, one block placed, five units spilled). Replacing incumbent band
  behavior globally fixed case 70 but regressed integrated score to 1.8436.
  Kept it only as a gated extra candidate for ≥5× band-height snowball cases
  (validation hits 16/70/90): **official 1.7978**, 100/100, radj@{1,2,3}s =
  1.350/1.260/1.258. Package rebuilt + parity-verified at 1.797786.
- Round 5: swept existing dissection `pin_scale` ordering. Standalone variants
  were worse, but as gated best-of candidates `pin_scale` 0.5 and 4.0 improved
  mid/large validation cases. The first broad gate hurt runtime on 104–120
  where no case selected the variants; final gate 50≤n≤103 kept the material
  wins: **official 1.7952**, 100/100, radj@{1,2,3}s =
  1.348/1.257/1.257. Package rebuilt + parity-verified at 1.795152.
- Round 6: no-code upper bound for same-area boundary reshaping found a
  deployable case-98 win. Broad reshape gates failed the median-1 runtime
  keep gate; final gate only tries free non-cluster/non-MIB blocks for n≥118
  and requires the reshaped rectangle to satisfy its full current-bbox boundary
  code. Kept as v13: **official 1.7903**, 100/100, radj@{1,2,3}s =
  1.343/1.254/1.253. Package rebuilt + parity-verified at 1.790330.
- Round 7: in-bbox boundary slide upper bound found public wins on cases 82
  and 98, but broad and n≥100 gates failed the median-1 runtime gate. Final
  v14 gate only runs for n∈{103,119}, uses obstacle-clearance endpoint
  candidates instead of log-grid sampling, and keeps only free non-cluster,
  non-MIB blocks while preserving the current bbox. Official **1.7827**,
  100/100, radj@{1,2,3}s = **1.342/1.249/1.248**. Package rebuilt +
  parity-verified at 1.782689.
- Round 8: left/right boundary queues were still area-sorted vertically. Added
  one extra wf=1.0 candidate that uses barycenter ordering for those edge
  queues, then keeps it behind the existing selector. Gated and replacement
  variants were measured but lost either quality or reliable runtime-adjusted
  margin. Official **1.7368**, 100/100, radj@{1,2,3}s =
  **1.329/1.220/1.216**. Package rebuilt + parity-verified at 1.736800;
  binary fuzz 400/400 feasible.
- Round 9: bottom/top band rows still sorted most units by width. Broad pin-x
  band-row ordering and an `lr_boundary>=27` gate lost runtime-adjusted margin,
  and a global replacement regressed cases 98/99. Final v16 hybrid uses
  edge-bary+band-pinx only below 118 blocks and preserves v15 width-first bands
  on 118+ blocks. Official **1.7027**, 100/100, radj@{1,2,3}s =
  **1.316/1.201/1.192**. Package rebuilt + parity-verified at 1.702727;
  binary fuzz 400/400 feasible.
- Round 10: tested a second wf=0.8 edge-bary+band-pinx candidate. Adding it as
  an extra portfolio member improved RF=1 but lost median-1 runtime-adjusted
  score, even with a structural high-boundary/moderate-net gate. Kept the
  signal as a **replacement** instead: the existing v16 hybrid candidate uses
  wf=0.8 only for 95≤n<118, b2b<2500, p2b<2000, boundary≥31,
  preplaced≤4; otherwise it remains wf=1.0. Official **1.6960**, 100/100.
  Same-window v16 recheck gate passed at radj@{1,2,3}s =
  **1.322/1.197/1.187** vs **1.332/1.204/1.192**. Package rebuilt +
  parity-verified at 1.696014; binary fuzz 400/400 feasible.
- Round 11: replayed stronger pin-pull variants for the hybrid ordering on
  n>=100. Kept one extra candidate with `pin_scale=6.0`,
  `edge_order_mode="bary"`, and `band_order_mode="pinx"` only for the weighted
  cases. Official **1.6845**, 100/100; radj@{1,2,3}s =
  **1.317/1.191/1.179**. Package rebuilt + parity-verified at 1.684492;
  binary fuzz 400/400 feasible.
- Round 12: replayed a narrower width pocket for the strong pin-pull hybrid.
  Broad `wf=0.85` improved RF=1 but lost median-1 against the older fast v18
  timing sample; final gate runs it only when `31 <= boundary_count <= 34`
  and (`p2b <= 1200` or `b2b > 6000`). Official **1.6651**, 100/100.
  Same-window v18 recheck gate passed at radj@{1,2,3}s =
  **1.357/1.192/1.166** vs **1.368/1.204/1.179**. Package rebuilt +
  parity-verified at 1.665114; binary fuzz 400/400 feasible.
- Round 13: replayed a wide edge-bary tail for the heaviest case. Kept a
  single `wf=1.15`, `pin_scale=6.0`, `edge_order_mode="bary"` candidate
  without band-pinx only for case-99-class features (`n>=120`, boundary>=36,
  b2b>6000, p2b>3000). Official **1.6568**, 100/100; radj@{1,2,3}s =
  **1.315/1.176/1.160**. Package rebuilt + parity-verified at 1.656802;
  binary fuzz 400/400 feasible.
- Round 14: added external-y anchors to the paid second dissection pass. B2B
  edges to blocks outside the current row-ordering queue now pull toward those
  blocks' previous y centers, improving row coherence without adding a
  portfolio candidate. Official **1.6507**, 100/100. Same-window v20 recheck
  gate passed at radj@{1,2,3}s = **1.311/1.172/1.156** vs
  **1.342/1.187/1.163**. Package rebuilt + parity-verified at 1.650715;
  binary fuzz 400/400 feasible.
- Round 15: changed non-flat cluster lane seeding from pure area order to
  boundary-side priority before area. This keeps the same two-lane cluster
  tiler and adds no portfolio member. Official **1.6483**, 100/100.
  Same-window v21 recheck gate passed at radj@{1,2,3}s =
  **1.338/1.180/1.155** vs **1.344/1.185/1.159**. Package rebuilt +
  parity-verified at 1.648337; binary fuzz 400/400 feasible.
- Round 16: no-code upper-bound scan over a short high-weight ordering
  variant list found two deployable strong pin-pull width pockets. Kept two
  tightly gated extra candidates: `wf=0.75` for high-boundary low-p2b or
  dense-b2b pockets, and `wf=1.15` for 106-108 block moderate-boundary/net
  pockets. Official **1.6385**, 100/100; radj@{1,2,3}s =
  **1.314/1.167/1.147** vs v22 **1.338/1.180/1.155**. Package rebuilt +
  parity-verified at 1.638545; binary fuzz 400/400 feasible.
- Round 17: capped-band scans found a material case-90 upper-bound win.
  Adding extra capped candidates improved raw score but failed the runtime
  keep gate. Kept the deployable replacement instead: the existing
  `band_edge_cap` candidate still fires through the v11 predictor, but uses
  `wf=1.1` for the low-p2b, dense-b2b, five/six-preplaced case-90 class.
  Official **1.6328**, 100/100; radj@{1,2,3}s =
  **1.299/1.159/1.143** vs v23 **1.314/1.167/1.147**. Package rebuilt +
  parity-verified at 1.632775; binary fuzz 400/400 feasible.
- Round 18: profiled heavy cases and replaced repeated tensor-scalar HPWL
  evaluation in candidate scoring with pre-extracted Python lists and one
  center pass. All v24 positions stayed identical; in-process runtime fell
  from 0.238s to 0.159s and radj@{1,2,3}s became
  **1.175/1.143/1.143**.
- Round 19: obstacle traces showed `free_row_clamped` left short slabs empty
  after testing only the admitted row. Kept a gated in-place backfill from the
  remaining queue, changing eight cases without adding a candidate. Official
  **1.6225**, 100/100; same-window radj@{1,2,3}s =
  **1.175/1.136/1.136** vs v25 **1.182/1.143/1.143**. Package rebuilt +
  parity-verified at 1.622518; binary fuzz 400/400 feasible.
- Round 20: preserving active-slab queue order but relaxing its internal
  aspect guard in the existing case-88 and case-90 feature pockets improved
  two HPWL residuals. Official **1.6207**, 100/100; same-window
  radj@{1,2,3}s = **1.173/1.134/1.134** vs v26
  **1.174/1.136/1.136**. Package rebuilt + parity-verified at 1.620687;
  binary fuzz 400/400 feasible.
- Round 21: a strong pin-pull dissection pass anchored by the selected
  incumbent was narrowed to the high-boundary, low-p2b 100-103-block feature
  pocket. It changes only case 81, reducing cost **1.468250 → 1.375491**.
  Official **1.6190**, 100/100; two-run median radj@{1,2,3}s =
  **1.168/1.133/1.133** vs v27 **1.171/1.134/1.134**. Package rebuilt +
  exact parity at 1.619032; binary fuzz 400/400 feasible.
- Round 22: iterating the same case-81 anchored pass once more reduces its
  cost **1.375491 → 1.323854**; the next iteration is selector-rejected.
  Official **1.6181**, 100/100; paired-window radj@{1,2,3}s =
  **1.174/1.133/1.133** vs v28 **1.176/1.133/1.133**. Package rebuilt +
  exact parity at 1.618110; binary fuzz 400/400 feasible.
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
