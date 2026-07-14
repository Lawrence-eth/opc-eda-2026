#!/usr/bin/env python3
"""Reject learned replacement slots that alter the public v32 final output.

This structural audit never computes public golden cost.  For each heavy public
case it runs learning-off v32, reruns learning-off with the preregistered slot
removed, and compares packed final positions plus inference-visible metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external" / "FloorSet" / "iccad2026contest"))

from audit_standard_slot_usage import (  # noqa: E402
    _metrics_equal,
    _position_sha256,
    _snapshot,
    _visible_metrics,
)
from evaluate_training_holdout import (  # noqa: E402
    OFFICIAL_ROOT,
    SOLUTION_DIR,
    _file_sha256,
    _git_state,
    _load_optimizer,
    _portable_path,
    _solver_component_hashes,
)
SLOTS = (0.8, 0.9, 1.0, 1.1, 1.2)


def parse_slot_map(payload):
    if not isinstance(payload, dict):
        raise ValueError("slot map payload must be an object")
    if "replacement_wf_by_size" in payload:
        if payload.get("schema_version") != 2 or payload.get("mode") != (
            "legacy_v32_final_output_preserving_slot_calibration"
        ):
            raise ValueError("slot-map artifact must use the frozen schema-2 contract")
        raw_map = payload["replacement_wf_by_size"]
    else:
        raw_map = payload
    if not isinstance(raw_map, dict) or not raw_map:
        raise ValueError("slot map must be a nonempty object")
    slot_map = {}
    for raw_n, raw_slot in raw_map.items():
        if not isinstance(raw_n, str) or not raw_n.isdigit() or str(int(raw_n)) != raw_n:
            raise ValueError("slot-map block counts must be canonical integer strings")
        if isinstance(raw_slot, bool) or not isinstance(raw_slot, (int, float)):
            raise ValueError("slot-map width factors must be numeric and non-boolean")
        slot = float(raw_slot)
        n = int(raw_n)
        if not math.isfinite(slot) or slot not in SLOTS:
            raise ValueError("slot-map width factor is not a standard paid slot")
        if not 100 <= n <= 120:
            raise ValueError("slot map contains a non-heavy size")
        slot_map[n] = slot
    return slot_map


def _optimizer_targets(constraints, golden_positions, n):
    targets = torch.full((n, 4), -1.0)
    if constraints is None or constraints.dim() <= 1:
        return targets
    columns = constraints.shape[1]
    for block in range(n):
        fixed = columns > 0 and constraints[block, 0] != 0
        preplaced = columns > 1 and constraints[block, 1] != 0
        if preplaced:
            targets[block] = torch.tensor(golden_positions[block])
        elif fixed:
            targets[block, 2] = golden_positions[block][2]
            targets[block, 3] = golden_positions[block][3]
    return targets


def _hard_target_positions_from_labels(labels, block_count):
    """Reconstruct rectangles without reading or computing golden metrics.

    The official public labels contain polygon coordinates followed by stored
    quality metrics. This audit needs polygon geometry only to recreate the
    fixed/preplaced hard-target input that the evaluator passes to solve().
    Keeping the extraction local prevents a rejection-only structural check
    from accidentally computing or consuming public golden HPWL/area values.
    """

    if not isinstance(labels, (tuple, list)) or not labels:
        raise ValueError("public label does not contain polygon geometry")
    polygons = labels[0]
    if len(polygons) < block_count:
        raise ValueError("public label contains too few block polygons")
    positions = []
    for block_index in range(block_count):
        block = polygons[block_index]
        if (
            not isinstance(block, torch.Tensor)
            or block.ndim != 2
            or block.shape[1] < 2
        ):
            raise ValueError(
                f"public block {block_index} has malformed polygon geometry"
            )
        coordinates = block[:, :2]
        padding = (coordinates[:, 0] == -1) & (coordinates[:, 1] == -1)
        mismatched_padding = (coordinates[:, 0] == -1) ^ (coordinates[:, 1] == -1)
        if bool(mismatched_padding.any()):
            raise ValueError(
                f"public block {block_index} has malformed polygon padding"
            )
        if not bool(torch.isfinite(coordinates[~padding]).all()):
            raise ValueError(
                f"public block {block_index} has non-finite polygon geometry"
            )
        valid = coordinates[~padding]
        if valid.shape[0] == 0:
            raise ValueError(
                f"public block {block_index} has no valid polygon vertices"
            )
        minimum = valid.min(dim=0).values
        maximum = valid.max(dim=0).values
        rectangle = (
            float(minimum[0]),
            float(minimum[1]),
            float(maximum[0] - minimum[0]),
            float(maximum[1] - minimum[1]),
        )
        if not all(math.isfinite(value) for value in rectangle):
            raise ValueError(
                f"public block {block_index} produced a non-finite rectangle"
            )
        if rectangle[2] <= 0.0 or rectangle[3] <= 0.0:
            raise ValueError(
                f"public block {block_index} produced nonpositive dimensions"
            )
        positions.append(rectangle)
    return positions


def main():
    # Parser and contract tests do not require an installed official checkout.
    # The actual audit imports the evaluator only when executing its CLI.
    from iccad2026_evaluate import ContestEvaluator

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=OFFICIAL_ROOT)
    parser.add_argument("--solver-dir", type=Path, default=SOLUTION_DIR)
    parser.add_argument("--slot-map", type=Path, required=True)
    parser.add_argument(
        "--expected-component",
        action="append",
        help="explicit research-only component registry override",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    map_payload = json.loads(args.slot_map.read_bytes())
    slot_map = parse_slot_map(map_payload)

    evaluator = ContestEvaluator(str(args.data_root), verbose=False)
    evaluator._load_dataset()
    optimizer = _load_optimizer(args.solver_dir)
    optimizer._learned_order_enabled = False
    optimizer._learned_order_mode = "off"
    optimizer._baselines_by_n = {}
    solver_globals = optimizer._select_candidate.__func__.__globals__
    rows = []
    seen_sizes = Counter()
    for test_id in range(len(evaluator.dataset)):
        sample = evaluator.dataset[test_id]
        inputs, labels = sample["input"], sample["label"]
        area, b2b, p2b, pins, constraints = inputs
        n = int((area != -1).sum().item())
        if n not in slot_map:
            continue
        hard_target_positions = _hard_target_positions_from_labels(labels, n)
        targets = _optimizer_targets(constraints, hard_target_positions, n)
        optimizer._debug_disabled_standard_wf = None
        control = _snapshot(optimizer.solve(
            n, area.clone(), b2b.clone(), p2b.clone(), pins.clone(),
            constraints.clone(), targets.clone(),
        ), n)
        control_metrics = _visible_metrics(
            optimizer, solver_globals, control, area, b2b, p2b, pins,
            constraints, targets,
        )
        optimizer._debug_disabled_standard_wf = slot_map[n]
        try:
            removed = _snapshot(optimizer.solve(
                n, area.clone(), b2b.clone(), p2b.clone(), pins.clone(),
                constraints.clone(), targets.clone(),
            ), n)
        finally:
            optimizer._debug_disabled_standard_wf = None
        removed_metrics = _visible_metrics(
            optimizer, solver_globals, removed, area, b2b, p2b, pins,
            constraints, targets,
        )
        control_sha = _position_sha256(control)
        removed_sha = _position_sha256(removed)
        positions_equal = control_sha == removed_sha
        metrics_equal = _metrics_equal(control_metrics, removed_metrics)
        rows.append({
            "test_id": test_id,
            "block_count": n,
            "removed_width_factor": slot_map[n],
            "control_positions_sha256": control_sha,
            "removed_positions_sha256": removed_sha,
            "final_positions_equal": positions_equal,
            "control_visible_metrics": control_metrics,
            "removed_visible_metrics": removed_metrics,
            "visible_metrics_equal": metrics_equal,
            "final_preserved": positions_equal and metrics_equal,
        })
        seen_sizes[n] += 1
    if any(seen_sizes[n] != 1 for n in slot_map):
        raise ValueError("public release must contain exactly one case per mapped size")
    rejected = sorted(row["block_count"] for row in rows if not row["final_preserved"])
    result = {
        "schema_version": 1,
        "mode": "public_v32_final_output_slot_removal_rejection",
        "config": {
            "solver_dir": _portable_path(args.solver_dir),
            "slot_map_sha256": hashlib.sha256(args.slot_map.read_bytes()).hexdigest(),
            "uses_golden_costs": False,
            "reads_stored_golden_metrics": False,
            "computes_golden_hpwl_or_area": False,
            "golden_geometry_use": "fixed/preplaced hard-target reconstruction only",
            "policy": "rejection_only_no_slot_retuning",
        },
        "provenance": {
            "harness_sha256": _file_sha256(Path(__file__)),
            "solver_components": _solver_component_hashes(
                args.solver_dir, args.expected_component
            ),
            "solver_git": _git_state(args.solver_dir),
        },
        "summary": {
            "mapped_cases": len(rows),
            "preserved_cases": len(rows) - len(rejected),
            "rejected_sizes": rejected,
        },
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
