# Project Status

**Updated 2026-07-07. For the full handoff, read [`HANDOFF.md`](HANDOFF.md).**

## Current result (official evaluator, 100 validation cases)

| Metric | Current (`integrated_v10`) | Pre-campaign (v9) |
|---|---|---|
| Score (RF=1) | **1.8074** | 2.7182 |
| Feasible | 100/100 | 100/100 |
| Runtime avg / max | 0.18s / 0.68s | 0.18s / 0.9s |
| Runtime-adjusted @ median 1s / 2s / 3s | **1.353 / 1.266 / 1.265** | 2.110 / 1.924 / 1.903 |
| Packaged binary (official command, incl. spawn) | 1.807413, 0 position diffs, avg 0.198s | — |

## What the solver is

`contest_solution/my_optimizer.py` = v9's constructive shelf (fast, no-SA
reference) + a portfolio of **exact-area dissection** layouts
(`contest_solution/dissect.py`, docs/CAMPAIGN_GOLDEN.md) selected per case by
a feasibility-gated exact-cost-shaped comparison. The dissection wins 91/100
cases; the shelf covers the rest. Deterministic, CPU-only, stdlib-only in the
packaged binary.

Key structural properties (why it beats everything previous):
- soft-block dims are derived from the row structure ⇒ areas exact, overlap
  impossible, utilization ≈1 in unobstructed regions;
- clusters tile contiguous regions ⇒ abutment by construction (the historical
  blocker);
- boundary demands satisfied by frame structure (bands / row-end slots /
  edge stacks), preplaced blocks carved around and never moved.

## Verification state (2026-07-07)

- 51/51 tests pass; result audit PASS; release gate PASS
  (defaults: `results/integrated_v10.json`, max-score 1.81).
- Submission package rebuilt + parity-verified (0/100 position diffs through
  the organizers' op_wrapper path); 400/400 training-instance binary fuzz.
- Cross-hardware determinism verified (bit-identical results reproduced on a
  fresh 2-core VM vs the original 48-core box, for the v9 baseline).

## Where the remaining score is

Dissect-only, n≥100 weighted: hpwl_gap 0.86, area_gap 0.25, V_rel 0.110,
util 0.79. Ranked open leads with evidence: `HANDOFF.md` §6. Golden-equivalent
play would score 1.108 (RF=1) / 0.776 (at the runtime floor).

## History

The v9-era feature list and sprint history live in `MASTER_PLAYBOOK.md` and
`PLAN_EXECUTION_LOG.md`; superseded plans in `docs/archive/`.
