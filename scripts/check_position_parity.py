#!/usr/bin/env python3
"""Require exact saved-output parity between two full evaluator runs.

Runtime fields are deliberately ignored.  Both inputs are audited, every
non-runtime case/summary quality field must match, and each coordinate is
compared and hashed as an IEEE-754 binary64 value.  Both runs must save
solutions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import audit_results


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load evaluator result {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"evaluator result {path} must contain a JSON object")
    return value


def _indexed_cases(result: dict[str, Any], label: str) -> dict[Any, dict[str, Any]]:
    cases = result.get("test_results")
    if not isinstance(cases, list):
        raise ValueError(f"{label} test_results must be a list")
    indexed: dict[Any, dict[str, Any]] = {}
    for offset, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"{label} case {offset} must be an object")
        test_id = case.get("test_id")
        if isinstance(test_id, bool) or not isinstance(test_id, (int, str)):
            raise ValueError(f"{label} case {offset} has an invalid test_id")
        if test_id in indexed:
            raise ValueError(f"{label} contains duplicate test_id {test_id!r}")
        indexed[test_id] = case
    return indexed


def _positions(case: dict[str, Any], label: str, test_id: Any) -> list[list[float]]:
    positions = case.get("positions")
    block_count = case.get("block_count")
    if (
        isinstance(block_count, bool)
        or not isinstance(block_count, int)
        or block_count < 1
    ):
        raise ValueError(f"{label} case {test_id!r} has an invalid block_count")
    if not isinstance(positions, list) or len(positions) != block_count:
        raise ValueError(
            f"{label} case {test_id!r} positions do not match block_count"
        )
    normalized = []
    for block, row in enumerate(positions):
        if (
            not isinstance(row, list)
            or len(row) != 4
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in row
            )
        ):
            raise ValueError(
                f"{label} case {test_id!r} block {block} is not four finite numbers"
            )
        normalized.append([float(value) for value in row])
    return normalized


def _exact_value_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without collapsing distinct IEEE-754 encodings."""

    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, float) or isinstance(right, float):
        return (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
            and struct.pack("!d", float(left)) == struct.pack("!d", float(right))
        )
    if type(left) is not type(right):
        return False
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _exact_value_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _exact_value_equal(left[key], right[key]) for key in left
        )
    return left == right


def _position_hash_start() -> Any:
    return hashlib.sha256(b"opc-eda-exact-positions-v1\0")


def _position_hash_case(digest: Any, test_id: Any, positions: list[list[float]]) -> None:
    encoded_id = json.dumps(
        test_id, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    digest.update(struct.pack("!I", len(encoded_id)))
    digest.update(encoded_id)
    digest.update(struct.pack("!I", len(positions)))
    for row in positions:
        for value in row:
            digest.update(struct.pack("!d", value))


def check_position_parity(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    expected_cases: int | None = 100,
) -> tuple[bool, list[str]]:
    """Return exact coordinate-parity status and concise diagnostics."""

    try:
        reference_cases = _indexed_cases(reference, "reference")
        candidate_cases = _indexed_cases(candidate, "candidate")
    except ValueError as exc:
        return False, [str(exc)]
    messages: list[str] = []
    for label, result in (("reference", reference), ("candidate", candidate)):
        audited, errors, _warnings = audit_results.audit_result(
            result,
            expected_cases=expected_cases,
            require_full_feasible=True,
            require_positions=True,
        )
        if not audited:
            messages.extend(f"{label} audit: {error}" for error in errors)
    if expected_cases is not None and (
        len(reference_cases) != expected_cases
        or len(candidate_cases) != expected_cases
    ):
        messages.append(
            f"expected {expected_cases} cases; reference has {len(reference_cases)} "
            f"and candidate has {len(candidate_cases)}"
        )
    reference_ids = set(reference_cases)
    candidate_ids = set(candidate_cases)
    if reference_ids != candidate_ids:
        missing = sorted(reference_ids - candidate_ids, key=str)
        extra = sorted(candidate_ids - reference_ids, key=str)
        if missing:
            messages.append(f"candidate is missing case IDs: {missing}")
        if extra:
            messages.append(f"candidate has extra case IDs: {extra}")

    compared_coordinates = 0
    reference_digest = _position_hash_start()
    candidate_digest = _position_hash_start()
    for test_id in sorted(reference_ids & candidate_ids, key=str):
        reference_case = reference_cases[test_id]
        candidate_case = candidate_cases[test_id]
        if reference_case.get("block_count") != candidate_case.get("block_count"):
            messages.append(f"case {test_id!r} block_count differs")
            continue
        try:
            reference_positions = _positions(reference_case, "reference", test_id)
            candidate_positions = _positions(candidate_case, "candidate", test_id)
        except ValueError as exc:
            messages.append(str(exc))
            continue
        _position_hash_case(reference_digest, test_id, reference_positions)
        _position_hash_case(candidate_digest, test_id, candidate_positions)

        ignored_case_fields = {"test_id", "positions"}
        quality_fields = (
            set(reference_case) | set(candidate_case)
        ) - ignored_case_fields
        quality_fields = {
            field for field in quality_fields if "runtime" not in field.casefold()
        }
        for field in sorted(quality_fields):
            if (
                field not in reference_case
                or field not in candidate_case
                or not _exact_value_equal(
                    reference_case.get(field), candidate_case.get(field)
                )
            ):
                messages.append(
                    f"case {test_id!r} non-runtime field {field!r} differs"
                )
        for block, (left, right) in enumerate(
            zip(reference_positions, candidate_positions)
        ):
            for coordinate, (left_value, right_value) in enumerate(zip(left, right)):
                compared_coordinates += 1
                if struct.pack("!d", left_value) != struct.pack("!d", right_value):
                    axis = ("x", "y", "width", "height")[coordinate]
                    messages.append(
                        f"case {test_id!r} block {block} {axis} differs: "
                        f"{left_value!r} != {right_value!r}"
                    )
                    if len(messages) >= 20:
                        messages.append("additional position differences omitted")
                        return False, messages

    if not _exact_value_equal(reference.get("total_score"), candidate.get("total_score")):
        messages.append("top-level total_score differs")
    reference_summary = reference.get("summary")
    candidate_summary = candidate.get("summary")
    if isinstance(reference_summary, dict) and isinstance(candidate_summary, dict):
        summary_fields = {
            field
            for field in set(reference_summary) | set(candidate_summary)
            if "runtime" not in field.casefold()
        }
        for field in sorted(summary_fields):
            if (
                field not in reference_summary
                or field not in candidate_summary
                or not _exact_value_equal(
                    reference_summary.get(field), candidate_summary.get(field)
                )
            ):
                messages.append(f"summary non-runtime field {field!r} differs")

    reference_sha = reference_digest.hexdigest()
    candidate_sha = candidate_digest.hexdigest()
    if reference_sha != candidate_sha:
        messages.append(
            f"packed position SHA-256 differs: {reference_sha} != {candidate_sha}"
        )
    if messages:
        return False, messages
    return True, [
        f"exact position parity: {len(reference_cases)} cases, "
        f"{compared_coordinates} coordinates, position_sha256={reference_sha}"
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--expected-cases",
        type=int,
        default=100,
        help="required case count for both results (default: 100)",
    )
    args = parser.parse_args()
    if args.expected_cases < 1:
        parser.error("--expected-cases must be positive")

    try:
        reference = _load(args.reference)
        candidate = _load(args.candidate)
    except ValueError as exc:
        print(f"Position parity: FAIL\n  {exc}")
        sys.exit(1)
    ok, messages = check_position_parity(
        reference,
        candidate,
        expected_cases=args.expected_cases,
    )
    print(f"Position parity: {'PASS' if ok else 'FAIL'}")
    for message in messages:
        print(f"  {message}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
