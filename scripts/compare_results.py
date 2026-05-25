#!/usr/bin/env python3
"""Compare a candidate FloorSet result JSON against a baseline.

This is a publication guard for score-focused experiments.  It fails when a
candidate is infeasible, evaluates fewer cases than the baseline, or does not
strictly improve the total score unless explicitly allowed.

Examples:
    python scripts/compare_results.py results/boundary_full.json candidate.json
    python scripts/compare_results.py results/boundary_full.json candidate.json --allow-equal
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

FLOAT_TOLERANCE = 1e-6


def _num(value: Any, default: float = 0.0) -> float:
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
    if "test_results" not in data or not isinstance(data["test_results"], list):
        raise SystemExit(f"No test_results list found in {path}")
    return data


def feasible_count(data: dict[str, Any]) -> int:
    return sum(1 for case in data["test_results"] if case.get("is_feasible") is True)


def summary_feasible_count(data: dict[str, Any]) -> int | None:
    summary = data.get("summary") or {}
    if "num_feasible" in summary:
        return int(_num(summary["num_feasible"]))
    return None


def total_score(data: dict[str, Any]) -> float:
    if "total_score" not in data:
        raise SystemExit("Result JSON is missing total_score")
    return _num(data["total_score"])


def reconstructed_total_score(data: dict[str, Any]) -> float:
    """Reconstruct the evaluator's weighted score from per-case metrics."""

    cases = data["test_results"]
    if not cases:
        return 0.0
    max_blocks = max(int(_num(case.get("block_count"))) for case in cases)
    raw_weights = [_block_weight(int(_num(case.get("block_count"))), max_blocks) for case in cases]
    total_weight = sum(raw_weights) or 1.0
    return sum(_num(case.get("cost")) * weight / total_weight for case, weight in zip(cases, raw_weights))


def case_count(data: dict[str, Any]) -> int:
    return len(data["test_results"])


def case_ids(data: dict[str, Any]) -> set[int]:
    return {
        int(_num(case.get("test_id"), idx))
        for idx, case in enumerate(data["test_results"])
    }


def duplicate_case_ids(data: dict[str, Any]) -> list[int]:
    seen: set[int] = set()
    duplicates: set[int] = set()
    for idx, case in enumerate(data["test_results"]):
        test_id = int(_num(case.get("test_id"), idx))
        if test_id in seen:
            duplicates.add(test_id)
        seen.add(test_id)
    return sorted(duplicates)


def avg_runtime(data: dict[str, Any]) -> float:
    summary = data.get("summary") or {}
    if "avg_runtime" in summary:
        return _num(summary["avg_runtime"])
    runtimes = [_num(case.get("runtime_seconds")) for case in data["test_results"]]
    return sum(runtimes) / len(runtimes) if runtimes else 0.0


def score_weights(cases: list[dict[str, Any]]) -> dict[int, float]:
    """Reconstruct official exponential case weights by block count."""

    if not cases:
        return {}
    max_blocks = max(int(_num(case.get("block_count"))) for case in cases)
    raw: list[tuple[int, float]] = []
    for idx, case in enumerate(cases):
        test_id = int(_num(case.get("test_id"), idx))
        raw.append((test_id, _block_weight(int(_num(case.get("block_count"))), max_blocks)))
    total = sum(weight for _, weight in raw) or 1.0
    return {test_id: weight / total for test_id, weight in raw}


def weighted_case_deltas(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    top: int = 5,
) -> tuple[list[str], list[str]]:
    """Return formatted largest weighted regressions and improvements."""

    baseline_cases = {
        int(_num(case.get("test_id"), idx)): case
        for idx, case in enumerate(baseline["test_results"])
    }
    candidate_cases = {
        int(_num(case.get("test_id"), idx)): case
        for idx, case in enumerate(candidate["test_results"])
    }
    weights = score_weights(list(baseline_cases.values()))

    rows = []
    for test_id in sorted(set(baseline_cases) & set(candidate_cases)):
        bcase = baseline_cases[test_id]
        ccase = candidate_cases[test_id]
        bcost = _num(bcase.get("cost"))
        ccost = _num(ccase.get("cost"))
        weighted_delta = (ccost - bcost) * weights.get(test_id, 0.0)
        rows.append(
            {
                "test_id": test_id,
                "block_count": int(_num(ccase.get("block_count"), _num(bcase.get("block_count")))),
                "baseline_cost": bcost,
                "candidate_cost": ccost,
                "cost_delta": ccost - bcost,
                "weighted_delta": weighted_delta,
                "hpwl_delta": _num(ccase.get("hpwl_gap")) - _num(bcase.get("hpwl_gap")),
                "area_delta": _num(ccase.get("area_gap")) - _num(bcase.get("area_gap")),
                "soft_delta": _num(ccase.get("violations_relative")) - _num(bcase.get("violations_relative")),
                "runtime_delta": _num(ccase.get("runtime_seconds")) - _num(bcase.get("runtime_seconds")),
            }
        )

    def fmt(row: dict[str, Any]) -> str:
        return (
            f"test_id={row['test_id']} blocks={row['block_count']} "
            f"cost_delta={row['cost_delta']:+.6f} "
            f"weighted_delta={row['weighted_delta']:+.6f} "
            f"candidate_cost={row['candidate_cost']:.6f} "
            f"baseline_cost={row['baseline_cost']:.6f} "
            f"hpwl_delta={row['hpwl_delta']:+.4f} "
            f"area_delta={row['area_delta']:+.4f} "
            f"soft_delta={row['soft_delta']:+.4f} "
            f"runtime_delta={row['runtime_delta']:+.4f}s"
        )

    regressions = [fmt(row) for row in sorted(rows, key=lambda r: r["weighted_delta"], reverse=True) if row["weighted_delta"] > 0]
    improvements = [fmt(row) for row in sorted(rows, key=lambda r: r["weighted_delta"]) if row["weighted_delta"] < 0]
    return regressions[:top], improvements[:top]


def compare(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    allow_equal: bool = False,
    require_full_feasible: bool = True,
    min_cases: int | None = None,
) -> tuple[bool, list[str]]:
    baseline_score = total_score(baseline)
    candidate_score = total_score(candidate)
    baseline_reconstructed = reconstructed_total_score(baseline)
    candidate_reconstructed = reconstructed_total_score(candidate)
    baseline_cases = case_count(baseline)
    candidate_cases = case_count(candidate)
    required_cases = baseline_cases if min_cases is None else min_cases
    candidate_feasible = feasible_count(candidate)
    candidate_summary_feasible = summary_feasible_count(candidate)
    baseline_ids = case_ids(baseline)
    candidate_ids = case_ids(candidate)
    candidate_duplicate_ids = duplicate_case_ids(candidate)

    messages = [
        f"baseline_score={baseline_score:.6f}",
        f"baseline_reconstructed_score={baseline_reconstructed:.6f}",
        f"candidate_score={candidate_score:.6f}",
        f"candidate_reconstructed_score={candidate_reconstructed:.6f}",
        f"delta={candidate_score - baseline_score:+.6f}",
        f"candidate_feasible={candidate_feasible}/{candidate_cases}",
        f"candidate_avg_runtime={avg_runtime(candidate):.4f}s",
    ]

    ok = True
    if not math.isclose(baseline_score, baseline_reconstructed, rel_tol=0.0, abs_tol=FLOAT_TOLERANCE):
        ok = False
        messages.append(
            f"FAIL: baseline total_score={baseline_score:.12f} does not match reconstructed score "
            f"{baseline_reconstructed:.12f}"
        )
    if not math.isclose(candidate_score, candidate_reconstructed, rel_tol=0.0, abs_tol=FLOAT_TOLERANCE):
        ok = False
        messages.append(
            f"FAIL: candidate total_score={candidate_score:.12f} does not match reconstructed score "
            f"{candidate_reconstructed:.12f}"
        )
    if candidate_cases < required_cases:
        ok = False
        messages.append(f"FAIL: candidate has {candidate_cases} cases, expected at least {required_cases}")
    if candidate_summary_feasible is not None and candidate_summary_feasible != candidate_feasible:
        ok = False
        messages.append(
            f"FAIL: candidate summary.num_feasible={candidate_summary_feasible} "
            f"does not match per-case feasible count {candidate_feasible}"
        )
    if candidate_duplicate_ids:
        ok = False
        preview = ", ".join(str(test_id) for test_id in candidate_duplicate_ids[:10])
        if len(candidate_duplicate_ids) > 10:
            preview += ", ..."
        messages.append(f"FAIL: candidate has duplicate test_id values: {preview}")
    missing_ids = sorted(baseline_ids - candidate_ids)
    if missing_ids:
        ok = False
        preview = ", ".join(str(test_id) for test_id in missing_ids[:10])
        if len(missing_ids) > 10:
            preview += ", ..."
        messages.append(f"FAIL: candidate is missing baseline test_id values: {preview}")
    if require_full_feasible and candidate_feasible != candidate_cases:
        ok = False
        messages.append("FAIL: candidate is not fully feasible")
    improved = candidate_score < baseline_score
    equal = candidate_score == baseline_score
    if not improved and not (allow_equal and equal):
        ok = False
        comparator = "equal to" if equal else "worse than"
        messages.append(f"FAIL: candidate score is {comparator} baseline")
    if ok:
        messages.append("PASS: candidate satisfies the publication guard")

    regressions, improvements = weighted_case_deltas(baseline, candidate)
    if regressions:
        messages.append("Top weighted regressions:")
        messages.extend(f"  {row}" for row in regressions)
    if improvements:
        messages.append("Top weighted improvements:")
        messages.extend(f"  {row}" for row in improvements)
    return ok, messages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_json", type=Path, help="Published baseline result JSON")
    parser.add_argument("candidate_json", type=Path, help="Candidate result JSON")
    parser.add_argument("--allow-equal", action="store_true", help="Allow equal total_score")
    parser.add_argument(
        "--allow-partial-feasible",
        action="store_true",
        help="Do not require all candidate cases to be feasible",
    )
    parser.add_argument(
        "--min-cases",
        type=int,
        default=None,
        help="Minimum candidate case count; defaults to baseline case count",
    )
    args = parser.parse_args()

    baseline = load_result(args.baseline_json)
    candidate = load_result(args.candidate_json)
    ok, messages = compare(
        baseline,
        candidate,
        allow_equal=args.allow_equal,
        require_full_feasible=not args.allow_partial_feasible,
        min_cases=args.min_cases,
    )
    for message in messages:
        print(message)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
