#!/bin/bash
# Quick smoke test: run the optimizer on a few representative cases.
# Portable — resolves paths relative to this script (no machine-specific paths).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
CONTEST="$ROOT/external/FloorSet/iccad2026contest"

# Make the optimizer available in the contest framework dir, then run from there.
cp "$ROOT/contest_solution/my_optimizer.py" "$ROOT/contest_solution/sequence_pair_sa.py" "$CONTEST/" 2>/dev/null || true
cd "$CONTEST" || exit 1
PYTHONPATH=.. python3 iccad2026_evaluate.py --evaluate my_optimizer.py \
    --test-id 0 --test-id 50 --test-id 99 --verbose 2>&1
echo "EXIT=$?"
