#!/usr/bin/env python3
"""Export a validated learned-order JSON artifact as a stdlib Python module."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def export_model(source: Path, output: Path):
    model = json.loads(source.read_bytes())
    payload_sha256 = model.get("payload_sha256")
    if not isinstance(payload_sha256, str) or len(payload_sha256) != 64:
        raise ValueError("source model lacks a valid payload_sha256")
    canonical = json.dumps(model, sort_keys=True, separators=(",", ":"), allow_nan=False)
    module = (
        '"""Generated learned-order model; do not edit by hand.\n\n'
        f"Source: {source.relative_to(ROOT)}\n"
        f"Payload SHA-256: {payload_sha256}\n"
        '"""\n\n'
        "import json as _json\n\n"
        f"MODEL = _json.loads({canonical!r})\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(module, encoding="utf-8")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "results" / "models" / "order_ridge_v5b_clean_raw.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "contest_solution" / "order_model_v5b.py",
    )
    args = parser.parse_args()
    export_model(args.source, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
