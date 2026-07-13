#!/usr/bin/env python3
"""Audit which legacy standard dissection width wins the deployed selector.

This is a structural portfolio-redundancy audit.  It runs the solver with the
learned replacement disabled and records only the identity of the selected
standard width-factor candidate.  Training labels are used solely to recreate
the fixed/preplaced optimizer input; no golden cost is computed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_training_holdout import (  # noqa: E402
    OFFICIAL_ROOT,
    SOLUTION_DIR,
    _file_sha256,
    _git_state,
    _load_optimizer,
    _optimizer_targets,
    _positions_from_training_label,
    _resolve_manifest_cases,
    _solver_component_hashes,
)
from lite_dataset import FloorplanDatasetLite  # noqa: E402


SLOTS = (0.8, 0.9, 1.0, 1.1, 1.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=OFFICIAL_ROOT)
    parser.add_argument("--solver-dir", type=Path, default=SOLUTION_DIR)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset = FloorplanDatasetLite(str(args.data_root))
    selected, manifest = _resolve_manifest_cases(
        dataset, args.data_root, args.fold_manifest, args.fold
    )
    optimizer = _load_optimizer(args.solver_dir)
    if not hasattr(optimizer, "_learned_order_enabled"):
        raise RuntimeError("solver does not expose the learned-control switch")
    optimizer._learned_order_enabled = False
    rows = []
    for ordinal, (sample_index, sample, identity) in enumerate(selected, 1):
        area, b2b, p2b, pins, constraints = sample["input"]
        _tree, fp_sol, _metrics = sample["label"]
        n = int((area != -1).sum().item())
        golden = _positions_from_training_label(fp_sol, n)
        targets = _optimizer_targets(constraints, golden, n)
        optimizer.solve(
            n,
            area.clone(),
            b2b.clone(),
            p2b.clone(),
            pins.clone(),
            constraints.clone(),
            targets.clone(),
        )
        selected_wf = getattr(optimizer, "_debug_selected_base_wf", None)
        rows.append(
            {
                "case_id": identity["case_id"],
                "sample_index": sample_index,
                "block_count": n,
                "selected_standard_wf": selected_wf,
            }
        )
        if ordinal % 10 == 0 or ordinal == len(selected):
            print(f"audited {ordinal}/{len(selected)}", flush=True)

    by_size = defaultdict(Counter)
    for row in rows:
        key = "other" if row["selected_standard_wf"] is None else str(
            row["selected_standard_wf"]
        )
        by_size[row["block_count"]][key] += 1
    result = {
        "schema_version": 1,
        "mode": "legacy_deployed_selector_standard_slot_usage",
        "config": {
            "solver_dir": str(args.solver_dir),
            "fold_manifest": str(args.fold_manifest),
            "fold": args.fold,
            "manifest": manifest,
            "learned_enabled": False,
            "golden_cost_computed": False,
        },
        "provenance": {
            "harness_sha256": _file_sha256(Path(__file__)),
            "solver_components": _solver_component_hashes(args.solver_dir),
            "solver_git": _git_state(args.solver_dir),
        },
        "slots": list(SLOTS),
        "counts_by_size": {
            str(n): dict(counts) for n, counts in sorted(by_size.items())
        },
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["counts_by_size"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
