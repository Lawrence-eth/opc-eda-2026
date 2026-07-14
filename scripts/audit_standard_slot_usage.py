#!/usr/bin/env python3
"""Audit final-output fidelity when each legacy standard slot is removed.

This is a structural portfolio-redundancy audit.  It runs the solver with the
learned replacement disabled, snapshots the complete deployed v32 output, then
reruns with one paid standard width suppressed at a time.  A slot is removable
only when its absence preserves both the final positions and all inference-
visible metrics after every downstream/path-dependent stage.  Training labels
are used solely to recreate the fixed/preplaced optimizer input; no golden cost
is computed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
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


def _snapshot(positions, n):
    if positions is None or len(positions) != n:
        raise ValueError("solver returned malformed positions")
    rows = tuple(tuple(float(value) for value in row) for row in positions)
    if any(len(row) != 4 for row in rows):
        raise ValueError("solver returned malformed rectangle")
    if not all(math.isfinite(value) for row in rows for value in row):
        raise ValueError("solver returned non-finite rectangle")
    return rows


def _position_sha256(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(struct.pack("!4d", *row))
    return digest.hexdigest()


def _visible_metrics(optimizer, solver_globals, positions, area, b2b, p2b, pins, constraints, targets):
    b2b_edges = optimizer._b2b_edges(b2b)
    p2b_edges = optimizer._p2b_edges(p2b)
    pins_l = pins.tolist() if pins is not None else []
    return {
        "feasible": bool(optimizer._is_feasible(positions, constraints, area, targets)),
        "hpwl": float(solver_globals["_calculate_hpwl_edges"](
            positions, b2b_edges, p2b_edges, pins_l
        )),
        "area": float(solver_globals["calculate_bbox_area"](positions)),
        "soft_violations": int(optimizer._soft_violation_count(positions, constraints)),
    }


def _metrics_equal(left, right):
    return (
        left["feasible"] == right["feasible"]
        and left["soft_violations"] == right["soft_violations"]
        and math.isclose(left["hpwl"], right["hpwl"], rel_tol=0.0, abs_tol=1e-9)
        and math.isclose(left["area"], right["area"], rel_tol=0.0, abs_tol=1e-9)
    )


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
    if not hasattr(optimizer, "_debug_disabled_standard_wf"):
        raise RuntimeError("solver does not expose the standard-slot audit switch")
    optimizer._learned_order_enabled = False
    solver_globals = optimizer._select_candidate.__func__.__globals__
    rows = []
    for ordinal, (sample_index, sample, identity) in enumerate(selected, 1):
        area, b2b, p2b, pins, constraints = sample["input"]
        _tree, fp_sol, _metrics = sample["label"]
        n = int((area != -1).sum().item())
        golden = _positions_from_training_label(fp_sol, n)
        targets = _optimizer_targets(constraints, golden, n)
        optimizer._debug_disabled_standard_wf = None
        baseline_positions = _snapshot(optimizer.solve(
            n,
            area.clone(),
            b2b.clone(),
            p2b.clone(),
            pins.clone(),
            constraints.clone(),
            targets.clone(),
        ), n)
        baseline_metrics = _visible_metrics(
            optimizer, solver_globals, baseline_positions, area, b2b, p2b, pins,
            constraints, targets,
        )
        selected_wf = getattr(optimizer, "_debug_selected_base_wf", None)
        removals = {}
        for wf in SLOTS:
            optimizer._debug_disabled_standard_wf = wf
            removed_positions = _snapshot(optimizer.solve(
                n,
                area.clone(),
                b2b.clone(),
                p2b.clone(),
                pins.clone(),
                constraints.clone(),
                targets.clone(),
            ), n)
            removed_metrics = _visible_metrics(
                optimizer, solver_globals, removed_positions, area, b2b, p2b, pins,
                constraints, targets,
            )
            positions_equal = removed_positions == baseline_positions
            metrics_equal = _metrics_equal(removed_metrics, baseline_metrics)
            removals[str(wf)] = {
                "final_positions_sha256": _position_sha256(removed_positions),
                "final_positions_equal": positions_equal,
                "visible_metrics": removed_metrics,
                "visible_metrics_equal": metrics_equal,
                "final_preserved": positions_equal and metrics_equal,
            }
        optimizer._debug_disabled_standard_wf = None
        rows.append(
            {
                "case_id": identity["case_id"],
                "sample_index": sample_index,
                "block_count": n,
                "selected_standard_wf": selected_wf,
                "baseline_final_positions_sha256": _position_sha256(baseline_positions),
                "baseline_visible_metrics": baseline_metrics,
                "removals": removals,
            }
        )
        if ordinal % 10 == 0 or ordinal == len(selected):
            print(f"audited {ordinal}/{len(selected)}", flush=True)

    preserved_by_size = {}
    for n in sorted({row["block_count"] for row in rows}):
        same_size = [row for row in rows if row["block_count"] == n]
        preserved_by_size[str(n)] = {
            str(wf): sum(
                row["removals"][str(wf)]["final_preserved"] for row in same_size
            )
            for wf in SLOTS
        }
    result = {
        "schema_version": 2,
        "mode": "legacy_v32_final_output_slot_removal_fidelity",
        "config": {
            "solver_dir": str(args.solver_dir),
            "fold_manifest": str(args.fold_manifest),
            "fold": args.fold,
            "manifest": manifest,
            "learned_enabled": False,
            "golden_cost_computed": False,
            "comparison_stage": "complete_deployed_solver_output",
            "final_preserved_requires": [
                "bit_exact_positions",
                "feasibility_equal",
                "hpwl_abs_delta_le_1e-9",
                "area_abs_delta_le_1e-9",
                "soft_violations_equal",
            ],
        },
        "provenance": {
            "harness_sha256": _file_sha256(Path(__file__)),
            "solver_components": _solver_component_hashes(args.solver_dir),
            "solver_git": _git_state(args.solver_dir),
        },
        "slots": list(SLOTS),
        "preserved_counts_by_size": preserved_by_size,
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["preserved_counts_by_size"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
