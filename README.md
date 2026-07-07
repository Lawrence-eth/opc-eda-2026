# ICCAD 2026 FloorSet Challenge — Problem C

Deterministic heuristic optimizer for data-driven SoC floorplanning (21–120 blocks).

> **📊 Presentation / overview: [`docs/SUMMARY.md`](docs/SUMMARY.md)** — problem, solution, results, methodology, key insights, and honest assessment (each section ≈ one slide). Start here.
>
> **📦 Submission: [`SUBMISSION_PLAN.md`](SUBMISSION_PLAN.md)** — the June-2026 organizer requirements (PyInstaller executable via `op_wrapper.py`), the verified torch-free package (`packaging/`, build with `packaging/build_submission.sh`), full verification results, and the plan to the ≈2026-07-13 deadline.

## Current Result (submission candidate)

Current best = **v9 + exact-area dissection portfolio** (CAMPAIGN_GOLDEN G4) —
`contest_solution/my_optimizer.py` + `contest_solution/dissect.py`; result
snapshot `results/integrated_v8.json`. Previous locked entry: `sprint5_v9`
(2.7182, `results/v9_locked.json`) — still the per-case fallback inside solve().

| Metric | Value |
|--------|-------|
| Validation score (RF=1.0) | **1.9352** (v9: 2.7182) |
| Feasible cases | **100 / 100** |
| Average runtime | **~0.18s/case** (max ~0.6s) |
| Runtime-adjusted total @ median 1s | **1.45** (v9: 2.11; baseline `quadratic_v1`: 2.65) |

**Note on the score:** the contest cost is *runtime-adjusted* — `cost · max(0.7, (rt/median)^0.3)`. v9 trades a little raw quality for large speed, which dominates at the contest's likely runtime range (it beats every earlier solution for assumed median ≤ ~3.5s). So the raw 2.7182 is *higher* (worse) than older raw scores (e.g. the 2.6326 baseline) but the *runtime-adjusted* score is better. A higher-raw-quality, still-fast alternative (`quadratic_v1`, 2.466 @ 0.69s) is retained as a hedge for high-median scenarios. Full reasoning + history: `MASTER_PLAYBOOK.md`.

*Validation-set results only. Final ranking uses hidden test data.*

## Repository Structure

```
contest_solution/
├── my_optimizer.py          # Optimizer — THE submission source
├── sequence_pair_sa.py      # Experimental SP-SA floorplanner (dormant; see playbook)
├── iccad2026_evaluate.py    # Official evaluator (working copy)
└── *dataset*.py             # Dataset loaders + tests
scripts/                     # analyze / audit / compare / release-check / benchmark
tests/                       # Regression + unit tests
results/                     # Result artifacts (v9_locked.json = current; others gitignored)
external/FloorSet/           # Contest dataset + official evaluator
docs/
├── extracted/               # Contest problem statement + Q&A
└── archive/                 # Superseded plans/reports (history)
logs/                        # Per-session execution logs

MASTER_PLAYBOOK.md           # Strategy & decision playbook (authoritative current state at top)
PLAN_EXECUTION_LOG.md        # Chronological experiment log
PROJECT_STATUS.md            # Status summary
```

## Quick Start

`my_optimizer.py` targets the FloorSet contest framework — it is **not** a standalone module (it imports `iccad2026_evaluate` → `cost`/`utils`, which live in the FloorSet checkout). Run it by copying it into the official contest dir, with the FloorSet root on `PYTHONPATH` (the `PYTHONPATH=..` below). One-shot setup + eval: `scripts/setup_and_evaluate.sh`.

```bash
# Validate
cp contest_solution/my_optimizer.py /path/to/FloorSet/iccad2026contest/
cd /path/to/FloorSet/iccad2026contest
PYTHONPATH=.. python iccad2026_evaluate.py --validate my_optimizer.py --quick

# Evaluate
PYTHONPATH=.. python iccad2026_evaluate.py --evaluate my_optimizer.py
```
(The submission is self-contained as the single file `my_optimizer.py`; `sequence_pair_sa.py` is only needed if the dormant SP-SA path is re-enabled.)

## Release Checks

```bash
.venv/bin/python -m pytest                          # all tests pass (torch-dependent tests skip if torch absent)
.venv/bin/python scripts/check_public_release.py    # PASS (defaults: results/v9_locked.json, --max-score 2.72)
```

## Documentation

| Doc | Purpose |
|-----|---------|
| [`docs/SUMMARY.md`](docs/SUMMARY.md) | **Presentation overview** — problem, solution, results, insights (start here) |
| [`MASTER_PLAYBOOK.md`](MASTER_PLAYBOOK.md) | Full strategy & decision trail (authoritative current state at top) |
| [`PLAN_EXECUTION_LOG.md`](PLAN_EXECUTION_LOG.md) | Chronological experiment log |
| [`PROJECT_STATUS.md`](PROJECT_STATUS.md) | Status summary |
| `docs/extracted/` | Contest problem statement + official Q&A |
| `docs/archive/` | Superseded plans/reports (history) |
| `logs/` | Per-session execution logs |

## Contest

- Problem: [FloorSet Challenge](https://www.iccad-contest.org/Problems.html)
- Repository: [IntelLabs/FloorSet](https://github.com/IntelLabs/FloorSet)
- Contest dir: `iccad2026contest/`
- Scoring: `Cost = (1 + 0.5*(HPWL_gap + Area_gap)) * exp(2*V_rel) * max(0.7, RuntimeFactor^0.3)`
