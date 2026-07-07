#!/usr/bin/env bash
# Build the ICCAD 2026 Problem C submission package.
#
# Produces:
#   submission/src/                     torch-free solver sources (shim layout)
#   submission/dist/my_optimizer/       PyInstaller --onedir executable
#   submission/iccad2026_submission.tar.gz   the uploadable package
#
# The executable must NOT bundle real torch (that is the whole point), so the
# PyInstaller build runs in a dedicated venv that has pyinstaller only.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PKG="$ROOT/packaging"
SUB="$ROOT/submission"
SRC="$SUB/src"

echo "== assemble shim sources =="
rm -rf "$SRC" "$SUB/dist" "$SUB/build"
mkdir -p "$SRC"
cp "$ROOT/contest_solution/my_optimizer.py" "$SRC/my_optimizer.py"
cp "$ROOT/contest_solution/dissect.py"      "$SRC/dissect.py"
cp "$PKG/torch_stub.py"                     "$SRC/torch.py"
cp "$PKG/eval_stub.py"                      "$SRC/iccad2026_evaluate.py"
cp "$PKG/solver_main.py"                    "$SRC/solver_main.py"

echo "== build venv (pyinstaller only, no torch) =="
if [ ! -x "$SUB/.buildvenv/bin/pyinstaller" ]; then
  python3 -m venv "$SUB/.buildvenv"
  "$SUB/.buildvenv/bin/pip" install --quiet --upgrade pip
  "$SUB/.buildvenv/bin/pip" install --quiet pyinstaller
fi

echo "== pyinstaller (--onedir for fast startup) =="
cd "$SUB"
"$SUB/.buildvenv/bin/pyinstaller" --noconfirm --onedir --name my_optimizer \
  --distpath "$SUB/dist" --workpath "$SUB/build" --specpath "$SUB/build" \
  --paths "$SRC" \
  --hidden-import my_optimizer \
  --hidden-import dissect \
  --hidden-import iccad2026_evaluate \
  --hidden-import torch \
  "$SRC/solver_main.py" >/dev/null

# Real torch must not have leaked into the bundle.
if find "$SUB/dist/my_optimizer" -name "libtorch*" -o -name "*torch/_C*" | grep -q .; then
  echo "ERROR: real torch leaked into the bundle" >&2
  exit 1
fi
du -sh "$SUB/dist/my_optimizer"

echo "== smoke test the binary =="
"$SUB/dist/my_optimizer/my_optimizer" <<'EOF'
{"block_count": 3, "area_targets": [100.0, 25.0, 4.0], "b2b_connectivity": [[0,1,1.0],[1,2,2.0]], "p2b_connectivity": [], "pins_pos": [], "constraints": [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]], "target_positions": null}
EOF
echo

echo "== assemble tar.gz =="
STAGE="$SUB/iccad2026_submission"
rm -rf "$STAGE"
mkdir -p "$STAGE"
cp -r "$SUB/dist" "$STAGE/dist"
cp "$PKG/op_wrapper.py" "$STAGE/op_wrapper.py"          # organizers' wrapper, verbatim
cp "$PKG/README_SUBMISSION.md" "$STAGE/README.md"
cp -r "$SRC" "$STAGE/source_fallback"                    # source fallback per guidelines
printf '%s\n' \
  '# The executable is fully self-contained (PyInstaller --onedir).' \
  '# Nothing to install for the executable path.' \
  '# The source_fallback/ directory is pure Python 3 (stdlib only).' \
  > "$STAGE/requirements.txt"
tar -czf "$SUB/iccad2026_submission.tar.gz" -C "$SUB" iccad2026_submission
ls -la "$SUB/iccad2026_submission.tar.gz"
echo "OK"
