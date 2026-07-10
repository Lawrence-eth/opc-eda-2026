#!/bin/bash
# Quick smoke test: run the optimizer on a few representative cases.
# Portable — resolves paths relative to this script (no machine-specific paths).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
CONTEST="$ROOT/external/FloorSet/iccad2026contest"

# Make the optimizer available in the contest framework dir, then run from there.
cp "$ROOT/contest_solution/my_optimizer.py" \
   "$ROOT/contest_solution/dissect.py" \
   "$ROOT/contest_solution/topology_polish.py" \
   "$ROOT/contest_solution/sequence_pair_sa.py" \
   "$CONTEST/"
cd "$CONTEST" || exit 1
PYTHON="$ROOT/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    PYTHON=python3
fi
for test_id in 0 50 99; do
    PYTHONPATH=.. "$PYTHON" iccad2026_evaluate.py --evaluate my_optimizer.py \
        --test-id "$test_id" --verbose 2>&1
done
echo "EXIT=$?"
