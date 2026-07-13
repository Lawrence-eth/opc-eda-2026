#!/bin/bash
# Quick smoke test: run the optimizer on a few representative cases.
# Portable — resolves paths relative to this script (no machine-specific paths).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
CONTEST="$ROOT/external/FloorSet/iccad2026contest"
PYTHON="$ROOT/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    PYTHON=python3
fi

# Make the optimizer available in the contest framework dir, then run from there.
LIVE_SOLVER_COMPONENT_TEXT="$("$PYTHON" "$ROOT/scripts/solver_components.py")"
mapfile -t LIVE_SOLVER_COMPONENTS <<< "$LIVE_SOLVER_COMPONENT_TEXT"
for component in "${LIVE_SOLVER_COMPONENTS[@]}"; do
    cp "$ROOT/contest_solution/$component" "$CONTEST/$component"
done
cp "$ROOT/contest_solution/sequence_pair_sa.py" "$CONTEST/sequence_pair_sa.py"
cd "$CONTEST" || exit 1
for test_id in 0 50 99; do
    PYTHONPATH=.. "$PYTHON" iccad2026_evaluate.py --evaluate my_optimizer.py \
        --test-id "$test_id" --verbose 2>&1
done
echo "EXIT=$?"
