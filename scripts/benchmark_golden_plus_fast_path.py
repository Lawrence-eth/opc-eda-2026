#!/usr/bin/env python3
"""Isolate no-report overhead of the narrow MIB repair fast path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "contest_solution")]

from evaluate_golden_plus_repair import (  # noqa: E402
    _load_clean_fold0_cases,
    _load_official,
    _load_optimizer,
    _load_public_cases,
    _load_public_positions,
)
from golden_plus_repair import RepairConfig, repair_fixed_topology  # noqa: E402


def _repair(case, positions, config):
    return repair_fixed_topology(
        positions,
        case["areas"],
        case["b2b"],
        case["p2b"],
        case["pins"],
        case["constraints"],
        case["optimizer_targets"],
        config=config,
        return_report=False,
    )


def _plain_case(case):
    """Mirror the JSON/list inputs used by the packaged executable."""

    result = dict(case)
    for key in ("areas", "b2b", "p2b", "pins", "constraints", "optimizer_targets"):
        value = result[key]
        result[key] = value.tolist() if hasattr(value, "tolist") else value
    return result


def _percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _benchmark(cases, incumbents, config, repeats):
    cases = [_plain_case(case) for case in cases]
    expected = []
    for case, incumbent in zip(cases, incumbents):
        expected.append(_repair(case, incumbent, config))

    call_times = []
    by_case = [[] for _case in cases]
    sweep_times = []
    for _repeat in range(repeats):
        started_sweep = time.perf_counter()
        for index, (case, incumbent) in enumerate(zip(cases, incumbents)):
            started = time.perf_counter()
            actual = _repair(case, incumbent, config)
            elapsed = time.perf_counter() - started
            call_times.append(elapsed)
            by_case[index].append(elapsed)
            if actual != expected[index]:
                raise RuntimeError(f"repair is non-deterministic for {case['case_id']}")
        sweep_times.append(time.perf_counter() - started_sweep)

    canonical = lambda rows: [tuple(map(float, row)) for row in rows]
    changed = [
        case["case_id"]
        for case, incumbent, repaired in zip(cases, incumbents, expected)
        if canonical(repaired) != canonical(incumbent)
    ]
    return {
        "cases": len(cases),
        "repeats": repeats,
        "calls": len(call_times),
        "changed_cases": changed,
        "per_case": [
            {
                "case_id": case["case_id"],
                "block_count": case["block_count"],
                "changed": case["case_id"] in changed,
                "runtime_mean": statistics.fmean(values),
                "runtime_p95": _percentile(values, 0.95),
                "runtime_max": max(values),
            }
            for case, values in zip(cases, by_case)
        ],
        "runtime_seconds": {
            "call_mean": statistics.fmean(call_times),
            "call_p50": _percentile(call_times, 0.50),
            "call_p95": _percentile(call_times, 0.95),
            "call_p99": _percentile(call_times, 0.99),
            "call_max": max(call_times),
            "sweep_mean": statistics.fmean(sweep_times),
            "sweep_p95": _percentile(sweep_times, 0.95),
            "sweep_max": max(sweep_times),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--floorset-root", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--public-v32-result", type=Path, required=True)
    parser.add_argument("--solver-dir", type=Path, default=ROOT / "contest_solution")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    official = _load_official(args.floorset_root.resolve())
    public_cases = _load_public_cases(args.floorset_root.resolve(), official)
    clean_cases, _manifest = _load_clean_fold0_cases(
        args.floorset_root.resolve(), args.fold_manifest.resolve()
    )
    public_positions, _meta = _load_public_positions(
        args.public_v32_result.resolve()
    )
    public_incumbents = [
        public_positions[case["sample_index"]] for case in public_cases
    ]

    optimizer = _load_optimizer(args.solver_dir.resolve())
    clean_incumbents = []
    solver_times = []
    for case in clean_cases:
        started = time.perf_counter()
        clean_incumbents.append(
            optimizer.solve(
                case["block_count"],
                case["areas"],
                case["b2b"],
                case["p2b"],
                case["pins"],
                case["constraints"],
                case["optimizer_targets"],
            )
        )
        solver_times.append(time.perf_counter() - started)

    config = RepairConfig(
        enable_boundary=False, enable_mib=True, enable_grouping=False
    )
    result = {
        "schema_version": 1,
        "benchmark": "golden_plus_mib_safe_gate_no_report",
        "scope": {
            "public": True,
            "heavy_clean_v1_fold": 0,
            "sealed_v2_accessed": False,
        },
        "config": config.__dict__,
        "provenance": {
            "repair_source_sha256": hashlib.sha256(
                (ROOT / "contest_solution" / "golden_plus_repair.py").read_bytes()
            ).hexdigest(),
            "fold_manifest_sha256": hashlib.sha256(
                args.fold_manifest.read_bytes()
            ).hexdigest(),
            "public_v32_result_sha256": hashlib.sha256(
                args.public_v32_result.read_bytes()
            ).hexdigest(),
        },
        "panels": {
            "public": _benchmark(
                public_cases, public_incumbents, config, args.repeats
            ),
            "heavy_clean_v1_fold0": _benchmark(
                clean_cases, clean_incumbents, config, args.repeats
            ),
        },
        "clean_solver_runtime_mean": statistics.fmean(solver_times),
        "clean_solver_runtime_max": max(solver_times),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for panel, summary in result["panels"].items():
        runtime = summary["runtime_seconds"]
        print(
            f"{panel}: mean={runtime['call_mean']:.9f}s "
            f"p95={runtime['call_p95']:.9f}s "
            f"sweep={runtime['sweep_mean']:.6f}s "
            f"changed={len(summary['changed_cases'])}"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
