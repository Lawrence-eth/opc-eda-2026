# ICCAD 2026 FloorSet Optimizer

Repository for the ICCAD 2026 FloorSet Challenge Problem C optimizer. The public source of truth is `contest_solution/my_optimizer.py`; copies in official FloorSet checkouts are derived validation artifacts.

This repository currently contains a deterministic heuristic optimizer and supporting validation, audit, and analysis tooling. Local validation results are not hidden-test or leaderboard results. Do not claim contest placement from local artifacts alone.

## Repository Structure

```text
.
├── contest_solution/          # Canonical optimizer submission source
│   ├── my_optimizer.py
│   └── test_my_optimizer.py
├── scripts/                   # Benchmark, audit, comparison, and analysis tools
├── tests/                     # Public regression tests
├── results/                   # Curated validation artifacts only
├── docs/                      # Contest notes and extracted references
├── PLAN/                      # Historical planning notes
└── ANALYSIS/                  # Historical analysis notes
```

`external/FloorSet/` and `/workspace/eda/FloorSet/` are local official-checkout mirrors used for validation. They should not be committed as project source.

## Current Validation Snapshot

Latest local official-evaluator run against the synced contest checkout:

- Result file: `results/tuned35_official_full.json`
- Validation cases: 100
- Feasible cases: 100 / 100
- Total score: 2.7494
- Average runtime: 0.91s
- Official quick validator: passed
- Public release guard: passed against `results/tuned35_official_full.json`

These are validation-set results. Final contest ranking uses hidden test data and official leaderboard evaluation.

## Quick Start

Install dependencies in the project virtual environment, then copy the optimizer into an official FloorSet checkout for validation:

```bash
cp contest_solution/my_optimizer.py /path/to/FloorSet/iccad2026contest/my_optimizer.py
cd /path/to/FloorSet/iccad2026contest
PYTHONPATH=.. python iccad2026_evaluate.py --validate my_optimizer.py --quick
PYTHONPATH=.. python iccad2026_evaluate.py --evaluate my_optimizer.py --verbose
```

From this repository, the local benchmark helper can run the 100 validation cases:

```bash
.venv/bin/python scripts/benchmark_ml_score.py --num-cases 100 --output results/benchmark_ml_score_100cases_local.json
```

That helper JSON is for local tuning. Release checks expect official evaluator JSON with top-level `total_score` and `test_results`.

## Release Checks

Before publishing optimizer changes, run:

```bash
.venv/bin/python -m pytest -q tests/test_optimizer_soft_constraints.py tests/test_sp_labels.py
.venv/bin/python scripts/check_public_release.py \
  --result results/tuned35_official_full.json \
  --max-score 2.7493747535028557 \
  --contest-optimizer "$CONTEST_OPTIMIZER"
```

Set `CONTEST_OPTIMIZER` to the active `my_optimizer.py` inside your local FloorSet checkout. The release guard checks result integrity, public-safe wording, and optimizer-copy synchronization.

## Artifact Policy

- Keep curated official-evaluator artifacts in `results/` only when they are useful for review or release validation.
- Treat `results/probe_*.json`, `results/benchmark_ml_score_*_local.json`, and sweep outputs as local experiment artifacts. They are ignored by default.
- Do not commit local external checkouts, generated solution dumps, logs, models, or private environment files.

## Contest Notes

- Problem: Data-driven SoC floorplanning with 21-120 blocks.
- Validation data: 100 public Lite validation cases.
- Final ranking: hidden test data evaluated by the contest infrastructure.
- Scoring: `Cost = (1 + 0.5*(HPWL_gap + Area_gap)) * exp(2*V_rel) * max(0.7, RuntimeFactor^0.3)`.

## License

Contest submission workspace for ICCAD 2026 Problem C.
