#!/usr/bin/env bash
# Build the ICCAD 2026 Problem C submission package.
#
# Produces:
#   submission/src/                     torch-free solver sources (shim layout)
#   submission/dist/my_optimizer/       x86-64 PyInstaller --onedir executable
#   submission/iccad2026_submission.tar.gz   the uploadable package
#
# The executable must NOT bundle real torch (that is the whole point), so the
# PyInstaller build runs in a dedicated venv that has pyinstaller only.
# The official host is an Intel Xeon running Debian 13.  On a non-x86 build
# host we re-enter this script in an amd64 Debian 13/Python 3.13 container;
# shipping a native ARM PyInstaller binary would fail before solve() starts.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PKG="$ROOT/packaging"
SUB="$ROOT/submission"
SRC="$SUB/src"
TARGET_MACHINE="x86_64"

if [ "${ICCAD_AMD64_BUILD_CONTAINER:-0}" != "1" ] && [ "$(uname -m)" != "$TARGET_MACHINE" ]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: an x86-64 build host or Docker with linux/amd64 support is required" >&2
    exit 1
  fi
  echo "== re-enter amd64 Debian 13 / Python 3.13 build container =="
  docker run --rm --platform linux/amd64 \
    --user "$(id -u):$(id -g)" \
    -e ICCAD_AMD64_BUILD_CONTAINER=1 \
    -e HOME=/tmp \
    -v "$ROOT:/workspace" \
    -w /workspace \
    python:3.13-trixie \
    bash packaging/build_submission.sh
  exit 0
fi

if [ "$(uname -m)" != "$TARGET_MACHINE" ]; then
  echo "ERROR: build environment is $(uname -m), expected $TARGET_MACHINE" >&2
  exit 1
fi

echo "== assemble shim sources =="
rm -rf "$SRC" "$SUB/dist" "$SUB/build"
mkdir -p "$SRC"
cp "$ROOT/contest_solution/my_optimizer.py" "$SRC/my_optimizer.py"
cp "$ROOT/contest_solution/dissect.py"      "$SRC/dissect.py"
cp "$ROOT/contest_solution/topology_polish.py" "$SRC/topology_polish.py"
cp "$PKG/torch_stub.py"                     "$SRC/torch.py"
cp "$PKG/eval_stub.py"                      "$SRC/iccad2026_evaluate.py"
cp "$PKG/solver_main.py"                    "$SRC/solver_main.py"

BUILD_VENV="$SUB/.buildvenv-amd64"

echo "== build venv (pyinstaller only, no torch) =="
if [ ! -x "$BUILD_VENV/bin/pyinstaller" ]; then
  python3 -m venv "$BUILD_VENV"
  "$BUILD_VENV/bin/pip" install --quiet --upgrade pip
  "$BUILD_VENV/bin/pip" install --quiet pyinstaller
fi

echo "== pyinstaller (--onedir for fast startup) =="
cd "$SUB"
"$BUILD_VENV/bin/pyinstaller" --noconfirm --onedir --name my_optimizer \
  --distpath "$SUB/dist" --workpath "$SUB/build" --specpath "$SUB/build" \
  --paths "$SRC" \
  --hidden-import my_optimizer \
  --hidden-import dissect \
  --hidden-import topology_polish \
  --hidden-import iccad2026_evaluate \
  --hidden-import torch \
  "$SRC/solver_main.py" >/dev/null

# Real torch must not have leaked into the bundle.  Use the build interpreter
# for this check; a find|grep pipeline proved unreliable under qemu binfmt.
python3 - "$SUB/dist/my_optimizer" <<'PY'
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
bad = [p for p in root.rglob("*")
       if p.name.startswith("libtorch") or "torch/_C" in p.as_posix()]
if bad:
    raise SystemExit("ERROR: real torch leaked into the bundle: "
                     + ", ".join(map(str, bad[:5])))
PY

# ELF e_machine 62 is AMD x86-64.  Check the actual artifact, not uname alone.
python3 - "$SUB/dist/my_optimizer/my_optimizer" <<'PY'
import pathlib
import struct
import sys

path = pathlib.Path(sys.argv[1])
header = path.read_bytes()[:20]
if header[:4] != b"\x7fELF" or len(header) < 20:
    raise SystemExit(f"ERROR: {path} is not an ELF executable")
endian = "<" if header[5] == 1 else ">"
machine = struct.unpack(endian + "H", header[18:20])[0]
if machine != 62:
    raise SystemExit(
        f"ERROR: {path} has ELF e_machine={machine}; expected 62 (AMD x86-64)"
    )
print(f"verified AMD x86-64 ELF: {path}")
PY
du -sh "$SUB/dist/my_optimizer"

echo "== smoke test the binary =="
"$SUB/dist/my_optimizer/my_optimizer" <<'EOF'
{"block_count": 3, "area_targets": [100.0, 25.0, 4.0], "b2b_connectivity": [[0,1,1.0],[1,2,2.0]], "p2b_connectivity": [], "pins_pos": [], "constraints": [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]], "target_positions": null}
EOF
echo

# PyInstaller imports the shim sources during analysis; do not ship bytecode
# cache files in the documented source fallback.
find "$SRC" -type d -name __pycache__ -prune -exec rm -rf {} +

echo "== assemble tar.gz =="
STAGE="$SUB/iccad2026_submission"
rm -rf "$STAGE"
mkdir -p "$STAGE"
cp -r "$SUB/dist" "$STAGE/dist"
cp "$PKG/op_wrapper.py" "$STAGE/op_wrapper.py"          # organizers' wrapper, verbatim
cp "$PKG/README_SUBMISSION.md" "$STAGE/README.md"
cp -r "$SRC" "$STAGE/source_fallback"                    # source fallback per guidelines
printf '%s\n' \
  '# The x86-64 executable is fully self-contained (PyInstaller --onedir).' \
  '# Nothing to install for the executable path.' \
  '# The source_fallback/ directory is pure Python 3 (stdlib only).' \
  > "$STAGE/requirements.txt"

# Normalize permissions before archiving.  In particular, the organizer's
# downloaded wrapper can inherit mode 0600; a root-owned extraction followed
# by an unprivileged evaluator would then be unable to import it.
find "$STAGE" -type d -exec chmod 0755 {} +
find "$STAGE" -type f -exec chmod 0644 {} +
chmod 0755 "$STAGE/dist/my_optimizer/my_optimizer"

tar --owner=0 --group=0 --numeric-owner \
  -czf "$SUB/iccad2026_submission.tar.gz" -C "$SUB" iccad2026_submission
ls -la "$SUB/iccad2026_submission.tar.gz"
echo "OK"
