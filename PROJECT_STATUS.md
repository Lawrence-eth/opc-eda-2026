# Project Status

**Updated 2026-07-11. For the full handoff, read [`HANDOFF.md`](HANDOFF.md).**

## Current result (official evaluator, 100 validation cases)

| Metric | Current (`integrated_v32`) | Pre-campaign (v9) |
|---|---|---|
| Score (RF=1) | **1.615379** | 2.7182 |
| Feasible | 100/100 | 100/100 |
| Runtime avg | 0.196s paired (v31 control: 0.196s) | 0.18s |
| Runtime-adjusted @ median 1s / 2s / 3s | **1.1951 / 1.1312 / 1.1308** (paired v31: 1.1967 / 1.1341 / 1.1316) | 2.110 / 1.924 / 1.903 |
| Packaged binary | AMD64 Debian 13 build; exact 100-case wrapper parity | — |

## What the solver is

`contest_solution/my_optimizer.py` = v9's constructive shelf (fast, no-SA
reference) + a portfolio of **exact-area dissection** layouts
(`contest_solution/dissect.py`, docs/CAMPAIGN_GOLDEN.md) selected per case by
a feasibility-gated exact-cost-shaped comparison. v11 adds one gated
obstacle-band cap candidate for the measured case-70/90 failure mode; v12 adds
two gated pin-scale ordering candidates for n=50..103 HPWL/area wins; v13 adds
a high-weight same-area boundary reshape candidate for free n>=118 blocks; v14
adds a tightly gated same-bbox boundary edge-slide polish for the n=103/119
classes; v15 adds one barycentric left/right edge-queue ordering candidate for
HPWL wins on high-weight cases; v16 switches that extra candidate to pin/net-x
band ordering for n<118 while preserving v15 width-first bands on n>=118; v17
uses wf=0.8 for that hybrid candidate on high-boundary, moderate-net 95..117
block cases, without adding another candidate; v18 adds one high-weight
strong pin-pull hybrid ordering candidate (`pin_scale=6.0`) only for n>=100.
v19 adds a gated narrower-width version of that candidate (`wf=0.85`) only for
boundary-heavy n>=100 cases with low p2b edge count or very dense b2b nets.
v20 adds one narrow case-99-class tail candidate (`wf=1.15`,
`pin_scale=6.0`, `edge_order_mode="bary"`) for n>=120, boundary-heavy, dense
b2b/p2b cases. v21 adds an external-y anchor to the second dissection pass:
b2b edges to blocks outside the current queue pull row ordering toward those
blocks' previous y centers, improving HPWL without adding a portfolio member.
v22 changes non-flat cluster lane seeding from pure area order to left-boundary,
interior, right-boundary priority before area, improving grouping/boundary
interactions without adding a portfolio member.
v23 adds two tightly gated strong pin-pull width pockets (`wf=0.75` and
`wf=1.15`) for high-weight feature classes, changing only four validation
cases and keeping the result selector-gated.
v24 keeps the existing `band_edge_cap` candidate count unchanged but chooses
`wf=1.1` for the low-p2b case-90 obstacle-band class, capturing the capped-band
scan signal without the runtime cost of an extra candidate.
v25 preserves every v24 placement while replacing repeated tensor-scalar HPWL
evaluation in candidate scoring with pre-extracted Python edge/pin lists and
one block-center pass per candidate.
v26 backfills short rows clamped to preplaced-obstacle edges from the remaining
mid queue in tightly gated feature pockets, improving eight cases without
adding a portfolio member.
v27 relaxes the active obstacle-slab aspect guard from 12:1 to 18:1 for the
case-88 backfill class and to 40:1 for the existing case-90 capped-band class,
improving HPWL on two cases without changing portfolio size.
v28 runs one incumbent-anchored strong pin-pull dissection pass only for a
high-boundary, low-p2b 100-103-block feature pocket. The selector accepts the
case-81 ordering gain and changes no other validation placement.
v29 repeats that pass once when selected and stops at the first rejection,
capturing the next case-81 ordering gain without broadening the feature gate.
v30 reuses an existing portfolio pass with a relaxed aspect guard for one
preplaced-heavy feature pocket. v31 makes grouping selection match exact
Shapely contact and applies a topology-preserving weighted-median HPWL polish
only for n≤90; 65 public cases improve and none regress.
v32 retains the first pass already computed inside the unconditional n≥100
strong-pin candidate, then compares that saved layout against the fully
repaired incumbent at the final selector. It adds no dissection solve. Across
five source-disjoint folds it improves clean **1.778134 → 1.767565** and raw
**1.848364 → 1.834588**, with all 1,050 evaluations feasible; the sealed clean
and raw fold-4 deltas are −0.010418 and −0.012594.
The dissection family wins most cases; the shelf covers the rest. Deterministic,
CPU-only, stdlib-only in the packaged binary.

Key structural properties (why it beats everything previous):
- soft-block dims are derived from the row structure ⇒ areas exact, overlap
  impossible, utilization ≈1 in unobstructed regions;
- clusters tile contiguous regions ⇒ abutment by construction (the historical
  blocker);
- boundary demands satisfied by frame structure (bands / row-end slots /
  edge stacks), preplaced blocks carved around and never moved.

## Verification state (2026-07-11)

- The v32 source tests, official evaluation, and release gates pass. The
  manifest binds solver commit `d8f2c19`, the promoted result, the AMD64
  archive, and pinned FloorSet commit `aadddcc`.
- Submission package rebuilt for AMD64 and parity-verified: 100/100 feasible,
  all 28,200 position scalars and every quality metric exact through the
  organizers' `op_wrapper.py` path. The v31 target binary also passed 100/100
  random training instances under QEMU (timings nonrepresentative); the latest
  400-case native binary fuzz was v29. v31 additionally passed a 140/140
  MIB-clean low-size source holdout.
- Cross-hardware determinism verified (bit-identical results reproduced on a
  fresh 2-core VM vs the original 48-core box, for the v9 baseline).

## Where the remaining score is

Current n≥100 averages: hpwl_gap 0.560, area_gap 0.154, V_rel 0.085.
Score-weighted averages: hpwl_gap 0.560, area_gap 0.145, V_rel 0.085.
Exact v31 soft ledger: boundary 323, grouping 55, MIB 126, total 504/4478
(`results/enriched_diagnostics.json`). Score-weighted soft counts in the
top-20 focus band are boundary 3.117, MIB 1.127, grouping 0.773.
Ranked open leads with evidence: `HANDOFF.md` §6. Golden-equivalent
play would score 1.108 (RF=1) / 0.776 (at the runtime floor). Golden-structure
mining is now in `results/golden_structure.json`: golden pays boundary misses
(219 total, 13 preplaced), but is MIB-uniform on every group and cluster-
connected on 350/360 groups.

## History

The v9-era feature list and sprint history live in `MASTER_PLAYBOOK.md` and
`PLAN_EXECUTION_LOG.md`; superseded plans in `docs/archive/`.
