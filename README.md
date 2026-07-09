# ICCAD 2026 FloorSet Challenge — Problem C

Deterministic heuristic optimizer for data-driven SoC floorplanning (21–120 blocks).

> **📊 Presentation / overview: [`docs/SUMMARY.md`](docs/SUMMARY.md)** — problem, solution, results, methodology, key insights, and honest assessment (each section ≈ one slide). Start here.
>
> **📦 Submission: [`SUBMISSION_PLAN.md`](SUBMISSION_PLAN.md)** — organizer requirements (PyInstaller executable via `op_wrapper.py`), the verified torch-free package (`packaging/`, build with `packaging/build_submission.sh`), and the rebuild gate.
>
> **🤝 Handoff: [`HANDOFF.md`](HANDOFF.md)** — complete state, workflows, and ranked open leads for whoever continues the work. **🏁 Active plan: [`docs/CAMPAIGN_GOLDEN.md`](docs/CAMPAIGN_GOLDEN.md).**

## Current Result (submission candidate)

Current best = **v9 + exact-area dissection portfolio + gated obstacle-band cap
+ gated pin-scale ordering candidates + high-weight boundary reshape + gated
boundary edge-slide polish + width-adaptive hybrid pin-x/barycentric edge
ordering candidate + high-weight strong pin-pull hybrid ordering candidate
+ case-99 edge-bary tail candidate + external-y anchored second-pass ordering
+ cluster lane edge-ordering + gated strong-width pin-pull pockets
+ band-cap width replacement + list-based HPWL candidate scoring**
(CAMPAIGN_GOLDEN G6 polish) —
`contest_solution/my_optimizer.py` + `contest_solution/dissect.py`; result
snapshot `results/integrated_v25.json`.
Previous locked entry: `sprint5_v9` (2.7182, `results/v9_locked.json`) —
still the per-case fallback inside solve().

| Metric | Value |
|--------|-------|
| Validation score (RF=1.0) | **1.6328** (v9: 2.7182) |
| Feasible cases | **100 / 100** |
| Average runtime | **~0.159s/case** (max ~0.52s) |
| Runtime-adjusted total @ median 1s | **1.175** (v24: 1.299; v9: 2.11) |

**Note on the score:** the contest cost is *runtime-adjusted* —
`cost · max(0.7, (rt/median)^0.3)` with a hard 0.7× floor for being ≥~3×
faster than the field median. The current solver has 40% better raw quality
than v9, and v25's output-preserving HPWL scoring path improves v24 at medians
{1,2}s while tying at 3s because both reach the hard runtime floor.
Runtime discipline is a standing rule: changes are judged runtime-adjusted at
median ∈ {1,2,3}s (`HANDOFF.md` §5.3), never on the raw score alone.

*Validation-set results only. Final ranking uses hidden test data.*

## Repository Structure

```
contest_solution/
├── my_optimizer.py          # THE solver: shelf + dissection portfolio + selector
├── dissect.py               # Exact-area dissection engine (CAMPAIGN_GOLDEN)
├── sequence_pair_sa.py      # Dormant SP-SA floorplanner (historical)
├── iccad2026_evaluate.py    # Official evaluator (working copy)
└── *dataset*.py             # Dataset loaders + tests
packaging/                   # Submission executable sources + build script + organizers' op_wrapper
scripts/                     # dissect_eval / fuzz_binary / analyze / audit / release-check / retrieval scan
tests/                       # Regression + unit tests (53)
results/                     # Curated result artifacts (integrated_v25.json = current)
external/FloorSet/           # Contest framework + datasets (gitignored; see HANDOFF bootstrap)
docs/
├── CAMPAIGN_GOLDEN.md       # ACTIVE plan: evidence, milestones, open leads
├── extracted/               # Problem statement v10 + Q&A 2026-06-18 + submission guidelines
└── archive/                 # Superseded plans/reports (history)

HANDOFF.md                   # Complete handoff: state, workflows, ranked open leads
CLAUDE.md                    # Agent onboarding + hard rules
SUBMISSION_PLAN.md           # Submission mechanics + package verification + rebuild gate
MASTER_PLAYBOOK.md           # Historical strategy (dead-end list still binding)
PLAN_EXECUTION_LOG.md        # Chronological experiment log (newest at top)
PROJECT_STATUS.md            # One-page status summary
```

## Quick Start

`my_optimizer.py` targets the FloorSet contest framework — it is **not** a standalone module (it imports `iccad2026_evaluate` → `cost`/`utils`, which live in the FloorSet checkout). Run it by copying it into the official contest dir, with the FloorSet root on `PYTHONPATH` (the `PYTHONPATH=..` below). One-shot setup + eval: `scripts/setup_and_evaluate.sh`.

```bash
# Validate
cp contest_solution/my_optimizer.py contest_solution/dissect.py /path/to/FloorSet/iccad2026contest/
cd /path/to/FloorSet/iccad2026contest
PYTHONPATH=.. python iccad2026_evaluate.py --validate my_optimizer.py --quick

# Evaluate
PYTHONPATH=.. python iccad2026_evaluate.py --evaluate my_optimizer.py
```
(The solver is `my_optimizer.py` + `dissect.py`; `sequence_pair_sa.py` is only
needed if the dormant SP-SA path is re-enabled. Full environment bootstrap and
all workflows: `HANDOFF.md` §5.)

## Release Checks

```bash
.venv/bin/python -m pytest                          # all tests pass (torch-dependent tests skip if torch absent)
.venv/bin/python scripts/check_public_release.py    # PASS (defaults: results/integrated_v25.json, --max-score 1.635)
```

## Documentation

| Doc | Purpose |
|-----|---------|
| [`HANDOFF.md`](HANDOFF.md) | **Complete handoff** — state, workflows, ranked open leads (start here) |
| [`docs/CAMPAIGN_GOLDEN.md`](docs/CAMPAIGN_GOLDEN.md) | Active campaign — evidence, measured milestones |
| [`docs/SUMMARY.md`](docs/SUMMARY.md) | Presentation overview — problem, solution, results, insights |
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
