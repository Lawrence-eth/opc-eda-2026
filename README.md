# ICCAD 2026 FloorSet Challenge — Problem C

Deterministic heuristic optimizer for data-driven SoC floorplanning (21–120 blocks).

## Current Score

| Metric | Value |
|--------|-------|
| Validation score | **2.7132** |
| Feasible cases | 100 / 100 |
| Average runtime | 0.98s |

*Validation-set results only. Final ranking uses hidden test data.*

## Repository Structure

```
contest_solution/
├── my_optimizer.py          # Optimizer (submission source)
├── test_my_optimizer.py     # Optimizer tests
├── iccad2026_evaluate.py    # Official evaluator
├── lite_dataset.py          # Dataset loader
└── lite_dataset_test.py     # Dataset tests
scripts/
├── benchmark_ml_score.py    # Local benchmark
├── audit_results.py         # Result audit
├── check_public_release.py  # Release guard
├── analyze_results.py       # Analysis tools
├── compare_results.py       # Result comparison
└── setup_and_evaluate.sh    # Setup helper
tests/                       # Regression tests
results/                     # Curated validation artifacts
docs/                        # Contest reference docs
```

## Quick Start

```bash
# Validate
cp contest_solution/my_optimizer.py /path/to/FloorSet/iccad2026contest/
cd /path/to/FloorSet/iccad2026contest
PYTHONPATH=.. python iccad2026_evaluate.py --validate my_optimizer.py --quick

# Evaluate
PYTHONPATH=.. python iccad2026_evaluate.py --evaluate my_optimizer.py
```

## Release Checks

```bash
.venv/bin/python -m pytest -q tests/test_optimizer_soft_constraints.py tests/test_sp_labels.py
.venv/bin/python scripts/check_public_release.py \
  --result results/tuned37_official_full.json \
  --max-score 2.7132479301385657 \
  --contest-optimizer "$CONTEST_OPTIMIZER"
```

## Contest

- Problem: [FloorSet Challenge](https://www.iccad-contest.org/Problems.html)
- Repository: [IntelLabs/FloorSet](https://github.com/IntelLabs/FloorSet/tree/main/iccad2026contest)
- Scoring: `Cost = (1 + 0.5*(HPWL_gap + Area_gap)) * exp(2*V_rel) * max(0.7, RuntimeFactor^0.3)`
