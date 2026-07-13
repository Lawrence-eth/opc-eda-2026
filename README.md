# ICCAD 2026 FloorSet Challenge — Problem C

Deterministic heuristic optimizer for data-driven SoC floorplanning (21–120 blocks).

> **📊 Presentation / overview: [`docs/SUMMARY.md`](docs/SUMMARY.md)** — problem, solution, results, methodology, key insights, and honest assessment (each section ≈ one slide). Start here.
>
> **📦 Submission: [`SUBMISSION_PLAN.md`](SUBMISSION_PLAN.md)** — organizer requirements (PyInstaller executable via `op_wrapper.py`), the verified torch-free package (`packaging/`, build with `packaging/build_submission.sh`), and the rebuild gate.
>
> **🤝 Handoff: [`HANDOFF.md`](HANDOFF.md)** — complete state, workflows, and ranked open leads for whoever continues the work.
>
> **🏆 Winning strategy: [`docs/WINNING_PLAN.md`](docs/WINNING_PLAN.md)** —
> authoritative beta-to-final research plan, promotion gates, architecture,
> and execution calendar. This supersedes narrower tactical conclusions.

## Current Result (submission candidate)

Current best = **v32: v31 plus a zero-solve-cost final gate that reuses the
already-computed first pass of the heavy strong-pin candidate**. The rest of
the solver remains the exact-area dissection portfolio, constraint-preserving
polish, and feasibility-gated selector developed through v31 —
`contest_solution/my_optimizer.py` + `contest_solution/dissect.py` +
`contest_solution/topology_polish.py`; result
snapshot `results/integrated_v32.json`. v32 exposes an intermediate layout that
the incumbent previously discarded and compares it only after all primary
selection and boundary repairs. It adds no dissection pass.
Previous locked entry: `sprint5_v9` (2.7182, `results/v9_locked.json`) —
still the per-case fallback inside solve().

| Metric | Value |
|--------|-------|
| Validation score (RF=1.0) | **1.615379** (v31: 1.616638; v9: 2.7182) |
| Feasible cases | **100 / 100** |
| Average runtime | **0.196s/case paired** (v31 control: 0.196s) |
| Runtime-adjusted total @ median 1s | **1.1951** (paired v31: 1.1967) |

**Note on the score:** the contest cost is *runtime-adjusted* —
`cost · max(0.7, (rt/median)^0.3)` with a hard 0.7× floor for being ≥~3×
faster than the field median. The current solver has 40% better raw quality
than v9. v32 improves all five clean and all five raw source-disjoint heavy
folds: pooled clean **1.778134 → 1.767565** and raw **1.848364 → 1.834588**,
with 1,050/1,050 feasible. Its paired runtime-adjusted score improves at every
tested field median from 0.25s through 3s.
Runtime discipline is a standing rule: changes are judged runtime-adjusted at
median ∈ {0.25,0.5,1,2,3}s (`HANDOFF.md` §5.3), never on the raw score alone.

*Validation-set results only. Final ranking uses hidden test data.*

Frozen pre-beta release: **[v32-prebeta-20260711](https://github.com/Lawrence-eth/opc-eda-2026/releases/tag/v32-prebeta-20260711)**
(tagged source plus the verified AMD64 submission archive).

## Repository Structure

```
contest_solution/
├── my_optimizer.py          # THE solver: shelf + dissection portfolio + selector
├── dissect.py               # Exact-area dissection engine (CAMPAIGN_GOLDEN)
├── topology_polish.py       # Fixed-topology weighted-median HPWL polish (n≤90)
├── learned_order.py         # Input-only, permutation-equivariant order features
├── order_model_v5b.py       # Generated deployment artifact for learned ordering
├── golden_plus_repair.py    # Fail-closed fixed-topology MIB repair
├── sequence_pair_sa.py      # Dormant SP-SA floorplanner (historical)
├── iccad2026_evaluate.py    # Adapted local convenience copy (not the official source of truth)
└── *dataset*.py             # Dataset loaders + tests
packaging/                   # Submission executable sources + build script + organizers' op_wrapper
scripts/                     # Maintained and legacy tools; see scripts/README.md
tests/                       # Regression + unit tests
results/                     # Curated evidence and retention policy; see results/README.md
├── folds/                   # Immutable heavy folds, baselines, selector audits
└── models/                  # Reproducible learned-model artifacts and provenance
external/FloorSet/           # Contest framework + datasets (gitignored; see HANDOFF bootstrap)
docs/
├── README.md               # Documentation index and source-of-truth map
├── WINNING_PLAN.md          # AUTHORITATIVE first-place strategy + execution gates
├── CAMPAIGN_GOLDEN.md       # Prior dissection campaign: evidence and milestones
├── extracted/               # Problem statement v10 + Q&A 2026-06-18 + submission guidelines
└── archive/                 # Superseded plans/reports (history)

HANDOFF.md                   # Complete handoff: state, workflows, ranked open leads
CLAUDE.md                    # Agent onboarding + hard rules
SUBMISSION_PLAN.md           # Submission mechanics + package verification + rebuild gate
MASTER_PLAYBOOK.md           # History (exact failed variants remain dead ends)
PLAN_EXECUTION_LOG.md        # Chronological experiment log (newest at top)
PROJECT_STATUS.md            # Frozen v32 status snapshot (historical link target)
```

The tracked `contest_solution/iccad2026_evaluate.py` is an adapted convenience
copy. Official verification uses the evaluator in the pinned organizer checkout
at `external/FloorSet/iccad2026contest/iccad2026_evaluate.py`.

## Quick Start

`my_optimizer.py` targets the FloorSet contest framework — it is **not** a standalone module (it imports `iccad2026_evaluate` → `cost`/`utils`, which live in the FloorSet checkout). Run it by copying it into the official contest dir, with the FloorSet root on `PYTHONPATH` (the `PYTHONPATH=..` below). One-shot setup + eval: `scripts/setup_and_evaluate.sh`.

```bash
# Validate
cp contest_solution/my_optimizer.py contest_solution/dissect.py \
   contest_solution/topology_polish.py contest_solution/learned_order.py \
   contest_solution/order_model_v5b.py \
   contest_solution/golden_plus_repair.py \
   /path/to/FloorSet/iccad2026contest/
cd /path/to/FloorSet/iccad2026contest
PYTHONPATH=.. python iccad2026_evaluate.py --validate my_optimizer.py --quick

# Evaluate
PYTHONPATH=.. python iccad2026_evaluate.py --evaluate my_optimizer.py
```
(The live source set is registered in `scripts/solver_components.py` and
includes `my_optimizer.py`, `dissect.py`, `topology_polish.py`,
`learned_order.py`, `order_model_v5b.py`, and `golden_plus_repair.py`;
`sequence_pair_sa.py` is only
needed if the dormant SP-SA path is re-enabled. Full environment bootstrap and
all workflows: `HANDOFF.md` §5.)

## Release Checks

```bash
.venv/bin/python -m pytest                          # all tests pass (torch-dependent tests skip if torch absent)
.venv/bin/python scripts/check_public_release.py    # PASS (defaults + hashes: results/release_manifest.json)
```

## Documentation

| Doc | Purpose |
|-----|---------|
| [`HANDOFF.md`](HANDOFF.md) | **Complete handoff** — state, workflows, ranked open leads (start here) |
| [`docs/WINNING_PLAN.md`](docs/WINNING_PLAN.md) | **Authoritative winning plan** — research tracks, kill gates, beta/final calendar |
| [`docs/README.md`](docs/README.md) | Documentation index — active sources of truth vs snapshots/history |
| [`docs/CAMPAIGN_GOLDEN.md`](docs/CAMPAIGN_GOLDEN.md) | Prior dissection campaign — evidence, measured milestones |
| [`docs/SUMMARY.md`](docs/SUMMARY.md) | Presentation overview — problem, solution, results, insights |
| [`results/README.md`](results/README.md) | Evidence catalog — release, rollback, folds, timing, model provenance, retention |
| [`scripts/README.md`](scripts/README.md) | Tool catalog — supported workflows and deprecated legacy utilities |
| [`MASTER_PLAYBOOK.md`](MASTER_PLAYBOOK.md) | Historical strategy and decision trail |
| [`PLAN_EXECUTION_LOG.md`](PLAN_EXECUTION_LOG.md) | Chronological experiment log |
| [`PROJECT_STATUS.md`](PROJECT_STATUS.md) | Frozen v32 status snapshot; current status is `HANDOFF.md` |
| `docs/extracted/` | Contest problem statement + official Q&A |
| `docs/archive/` | Superseded plans/reports (history) |

## Contest

- Problem: [FloorSet Challenge](https://www.iccad-contest.org/Problems.html)
- Repository: [IntelLabs/FloorSet](https://github.com/IntelLabs/FloorSet)
- Contest dir: `iccad2026contest/`
- Scoring: `Cost = (1 + 0.5*(HPWL_gap + Area_gap)) * exp(2*V_rel) * max(0.7, RuntimeFactor^0.3)`
