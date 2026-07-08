# Project Status

**Updated 2026-07-08. For the full handoff, read [`HANDOFF.md`](HANDOFF.md).**

## Current result (official evaluator, 100 validation cases)

| Metric | Current (`integrated_v20`) | Pre-campaign (v9) |
|---|---|---|
| Score (RF=1) | **1.6568** | 2.7182 |
| Feasible | 100/100 | 100/100 |
| Runtime avg / max | 0.240s / 0.871s | 0.18s / 0.9s |
| Runtime-adjusted @ median 1s / 2s / 3s | **1.315 / 1.176 / 1.160** | 2.110 / 1.924 / 1.903 |
| Packaged binary (official command, incl. spawn) | 1.656802, 0 position diffs, avg 0.239s | — |

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
b2b/p2b cases.
The dissection family wins most cases; the shelf covers the rest. Deterministic,
CPU-only, stdlib-only in the packaged binary.

Key structural properties (why it beats everything previous):
- soft-block dims are derived from the row structure ⇒ areas exact, overlap
  impossible, utilization ≈1 in unobstructed regions;
- clusters tile contiguous regions ⇒ abutment by construction (the historical
  blocker);
- boundary demands satisfied by frame structure (bands / row-end slots /
  edge stacks), preplaced blocks carved around and never moved.

## Verification state (2026-07-08)

- 51/51 tests pass; result audit PASS; release gate PASS
  (defaults: `results/integrated_v20.json`, max-score 1.66).
- Submission package rebuilt + parity-verified (0/100 position diffs through
  the organizers' op_wrapper path); 400/400 training-instance binary fuzz.
- Cross-hardware determinism verified (bit-identical results reproduced on a
  fresh 2-core VM vs the original 48-core box, for the v9 baseline).

## Where the remaining score is

Current n≥100 averages: hpwl_gap 0.609, area_gap 0.157, V_rel 0.089.
Weighted worst-case averages: hpwl_gap 0.630, area_gap 0.164, V_rel 0.093.
Exact v20 soft ledger: boundary 327, grouping 57, MIB 122, total 506/4478
(`results/enriched_diagnostics.json`). Score-weighted soft counts in the
top-20 focus band are boundary 3.221, MIB 1.055, grouping 0.910.
Ranked open leads with evidence: `HANDOFF.md` §6. Golden-equivalent
play would score 1.108 (RF=1) / 0.776 (at the runtime floor). Golden-structure
mining is now in `results/golden_structure.json`: golden pays boundary misses
(219 total, 13 preplaced), but is MIB-uniform on every group and cluster-
connected on 350/360 groups.

## History

The v9-era feature list and sprint history live in `MASTER_PLAYBOOK.md` and
`PLAN_EXECUTION_LOG.md`; superseded plans in `docs/archive/`.
