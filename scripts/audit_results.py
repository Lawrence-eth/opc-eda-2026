#!/usr/bin/env python3
"""Audit FloorSet evaluator result JSON files before publication.

This script checks structural and metric integrity that score comparison alone
does not cover: duplicate case IDs, missing required fields, non-finite values,
summary mismatches, infeasible cases, and malformed saved rectangles.

Examples:
    python scripts/audit_results.py results/boundary_full.json
    python scripts/audit_results.py candidate_full.json --expected-cases 100 --max-score 2.0133
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REQUIRED_CASE_FIELDS = (
    "test_id",
    "block_count",
    "is_feasible",
    "cost",
    "hpwl_gap",
    "area_gap",
    "violations_relative",
    "runtime_seconds",
)
FINITE_NONNEGATIVE_FIELDS = (
    "cost",
    "hpwl_gap",
    "area_gap",
    "violations_relative",
    "runtime_seconds",
)
SUMMARY_AVERAGE_FIELDS = {
    "avg_cost": "cost",
    "avg_runtime": "runtime_seconds",
}
FLOAT_TOLERANCE = 1e-6
GEOMETRY_TOLERANCE = 1e-6


def _num(value: Any, default: float = math.nan) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _block_weight(block_count: int, max_blocks: int) -> float:
    return math.exp((block_count - max_blocks) / 12)


def load_result(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"{path} does not contain a JSON object")
    return data


def _is_finite_nonnegative(value: Any) -> bool:
    number = _num(value)
    return math.isfinite(number) and number >= 0.0


def _weighted_score(cases: list[dict[str, Any]]) -> float:
    """Reconstruct the evaluator's block-count weighted total score."""

    max_blocks = max(int(_num(case.get("block_count"))) for case in cases)
    raw_weights = [_block_weight(int(_num(case.get("block_count"))), max_blocks) for case in cases]
    total_weight = sum(raw_weights) or 1.0
    return sum(_num(case.get("cost")) * weight / total_weight for case, weight in zip(cases, raw_weights))


def _mean_case_field(cases: list[dict[str, Any]], field: str) -> float:
    values = [_num(case.get(field)) for case in cases]
    return sum(values) / len(values) if values else math.nan


def _audit_positions(
    case: dict[str, Any],
    errors: list[str],
    warnings: list[str],
    *,
    required: bool,
) -> None:
    test_id = case.get("test_id", "?")
    block_count = int(_num(case.get("block_count"), -1))
    positions = case.get("positions")
    if positions is None:
        message = f"case {test_id}: positions are absent; rerun evaluator with --save-solutions for deeper diagnostics"
        if required:
            errors.append(message)
        else:
            warnings.append(message)
        return
    if not isinstance(positions, list):
        errors.append(f"case {test_id}: positions must be a list")
        return
    if block_count >= 0 and len(positions) != block_count:
        errors.append(f"case {test_id}: positions length {len(positions)} does not match block_count {block_count}")
    valid_rects: list[tuple[int, float, float, float, float]] = []
    for idx, rect in enumerate(positions):
        if not isinstance(rect, (list, tuple)) or len(rect) != 4:
            errors.append(f"case {test_id}: position {idx} is not an [x, y, w, h] rectangle")
            continue
        values = [_num(v) for v in rect]
        if not all(math.isfinite(v) for v in values):
            errors.append(f"case {test_id}: position {idx} contains a non-finite value")
            continue
        if values[2] <= 0.0 or values[3] <= 0.0:
            errors.append(f"case {test_id}: position {idx} has non-positive width/height")
            continue
        valid_rects.append((idx, values[0], values[1], values[2], values[3]))
    for left_pos, (left_idx, x1, y1, w1, h1) in enumerate(valid_rects):
        for right_idx, x2, y2, w2, h2 in valid_rects[left_pos + 1:]:
            x_overlap = min(x1 + w1, x2 + w2) - max(x1, x2)
            y_overlap = min(y1 + h1, y2 + h2) - max(y1, y2)
            if x_overlap > GEOMETRY_TOLERANCE and y_overlap > GEOMETRY_TOLERANCE:
                errors.append(
                    f"case {test_id}: positions {left_idx} and {right_idx} overlap "
                    f"by {x_overlap:.6g} x {y_overlap:.6g}"
                )


def audit_result(
    data: dict[str, Any],
    *,
    expected_cases: int | None = None,
    require_full_feasible: bool = True,
    max_score: float | None = None,
    require_positions: bool = False,
) -> tuple[bool, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    total_score = _num(data.get("total_score"))
    if "total_score" not in data or not math.isfinite(total_score) or total_score < 0.0:
        errors.append("top-level total_score must be a finite nonnegative number")
    elif max_score is not None and total_score > max_score:
        errors.append(f"total_score {total_score:.6f} exceeds maximum allowed score {max_score:.6f}")

    cases = data.get("test_results")
    if not isinstance(cases, list) or not cases:
        errors.append("top-level test_results must be a non-empty list")
        return False, errors, warnings
    if expected_cases is not None and len(cases) != expected_cases:
        errors.append(f"test_results has {len(cases)} cases, expected {expected_cases}")

    summary = data.get("summary")
    if not isinstance(summary, dict):
        warnings.append("top-level summary is absent or not an object")
        summary = {}
    summary_num_tests = summary.get("num_tests")
    if summary_num_tests is not None and int(_num(summary_num_tests, -1)) != len(cases):
        errors.append(f"summary.num_tests={summary_num_tests} does not match {len(cases)} cases")

    seen_ids: set[int] = set()
    feasible_count = 0
    for idx, case in enumerate(cases):
        if not isinstance(case, dict):
            errors.append(f"case index {idx}: entry is not an object")
            continue
        for field in REQUIRED_CASE_FIELDS:
            if field not in case:
                errors.append(f"case index {idx}: missing required field {field}")
        test_id = int(_num(case.get("test_id"), idx))
        if test_id in seen_ids:
            errors.append(f"duplicate test_id {test_id}")
        seen_ids.add(test_id)
        block_count = _num(case.get("block_count"))
        if not math.isfinite(block_count) or block_count <= 0 or int(block_count) != block_count:
            errors.append(f"case {test_id}: block_count must be a positive integer")
        for field in FINITE_NONNEGATIVE_FIELDS:
            if field in case and not _is_finite_nonnegative(case[field]):
                errors.append(f"case {test_id}: {field} must be finite and nonnegative")
        if case.get("error") not in (None, ""):
            errors.append(f"case {test_id}: evaluator error is present: {case.get('error')}")
        if case.get("is_feasible") is True:
            feasible_count += 1
        elif require_full_feasible:
            errors.append(f"case {test_id}: is_feasible is not true")
        if require_positions or "positions" in case:
            _audit_positions(case, errors, warnings, required=require_positions)

    summary_feasible = summary.get("num_feasible")
    if summary_feasible is not None and int(_num(summary_feasible, -1)) != feasible_count:
        errors.append(f"summary.num_feasible={summary_feasible} does not match {feasible_count} feasible cases")
    if require_full_feasible and feasible_count != len(cases):
        errors.append(f"result is not fully feasible: {feasible_count}/{len(cases)} cases feasible")

    if not errors:
        reconstructed_score = _weighted_score(cases)
        if not math.isclose(total_score, reconstructed_score, rel_tol=0.0, abs_tol=FLOAT_TOLERANCE):
            errors.append(
                f"total_score {total_score:.12f} does not match reconstructed weighted score "
                f"{reconstructed_score:.12f}"
            )

    if isinstance(summary, dict):
        for summary_field, case_field in SUMMARY_AVERAGE_FIELDS.items():
            if summary_field not in summary:
                continue
            expected = _mean_case_field(cases, case_field)
            actual = _num(summary[summary_field])
            if not math.isfinite(actual):
                errors.append(f"summary.{summary_field} must be finite")
            elif not math.isclose(actual, expected, rel_tol=0.0, abs_tol=FLOAT_TOLERANCE):
                errors.append(
                    f"summary.{summary_field}={actual:.12f} does not match per-case average "
                    f"{expected:.12f}"
                )

    return not errors, errors, warnings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json", type=Path, help="Evaluator result JSON to audit")
    parser.add_argument("--expected-cases", type=int, default=None, help="Require an exact number of cases")
    parser.add_argument("--max-score", type=float, default=None, help="Fail if total_score exceeds this value")
    parser.add_argument("--allow-infeasible", action="store_true", help="Do not require every case to be feasible")
    parser.add_argument("--require-positions", action="store_true", help="Require saved [x, y, w, h] rectangles for every case")
    args = parser.parse_args()

    data = load_result(args.result_json)
    ok, errors, warnings = audit_result(
        data,
        expected_cases=args.expected_cases,
        require_full_feasible=not args.allow_infeasible,
        max_score=args.max_score,
        require_positions=args.require_positions,
    )

    print(f"Result audit: {args.result_json}")
    print(f"Status: {'PASS' if ok else 'FAIL'}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    if errors:
        print("Errors:")
        for error in errors:
            print(f"  - {error}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
