#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [ ! -d external/FloorSet ]; then
  mkdir -p external
  git clone https://github.com/IntelLabs/FloorSet.git external/FloorSet
fi
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
# Install the official contest dependencies (torch, numpy, shapely, matplotlib,
# tqdm, requests) plus pytest, rather than an ad-hoc list.
pip install -r external/FloorSet/iccad2026contest/requirements.txt pytest
cp contest_solution/my_optimizer.py external/FloorSet/iccad2026contest/my_optimizer.py
cp contest_solution/test_my_optimizer.py external/FloorSet/iccad2026contest/test_my_optimizer.py
cd external/FloorSet
PYTHONPATH=. python lite_dataset_test.py
cd iccad2026contest
PYTHONPATH=.. ../../../.venv/bin/python -m pytest test_my_optimizer.py -q
PYTHONPATH=.. ../../../.venv/bin/python iccad2026_evaluate.py --validate my_optimizer.py
mkdir -p "$ROOT/results"
PYTHONPATH=.. ../../../.venv/bin/python iccad2026_evaluate.py --evaluate my_optimizer.py --verbose --save-solutions --output "$ROOT/results/v9_locked.json"
cd "$ROOT"
python scripts/audit_results.py results/v9_locked.json --expected-cases 100 --require-positions
