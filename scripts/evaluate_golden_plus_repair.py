#!/usr/bin/env python3
"""Evaluate fixed-topology repair with the pinned official metric functions.

This research harness is intentionally limited to the released public set and
an explicitly named heavy-clean-v1 development fold.  It never discovers or
opens other manifests.  Golden ablations measure the repair ceiling; the v32
run measures deployable value on a real incumbent.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
SOLUTION_DIR = ROOT / "contest_solution"
sys.path.insert(0, str(SOLUTION_DIR))

from golden_plus_repair import RepairConfig, repair_fixed_topology  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit(path: Path) -> str | None:
    run = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return run.stdout.strip() if run.returncode == 0 else None


def _load_official(floorset_root: Path):
    contest_dir = floorset_root / "iccad2026contest"
    sys.path[:0] = [str(contest_dir), str(floorset_root)]
    path = contest_dir / "iccad2026_evaluate.py"
    spec = importlib.util.spec_from_file_location("_golden_plus_official", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _polygon_positions(polygons, block_count: int):
    result = []
    for index in range(block_count):
        block = polygons[index]
        valid = block[block[:, 0] != -1]
        low = valid.min(dim=0).values
        high = valid.max(dim=0).values
        result.append(
            (
                float(low[0]),
                float(low[1]),
                float(high[0] - low[0]),
                float(high[1] - low[1]),
            )
        )
    return result


def _training_positions(fp_solution, block_count: int):
    return [
        (
            float(fp_solution[index, 2]),
            float(fp_solution[index, 3]),
            float(fp_solution[index, 0]),
            float(fp_solution[index, 1]),
        )
        for index in range(block_count)
    ]


def _baseline(metrics):
    return {
        "area_baseline": float(metrics[0]),
        "hpwl_baseline": float(metrics[-2]) + float(metrics[-1]),
    }


def _optimizer_targets(constraints, golden, block_count: int):
    result = torch.full((block_count, 4), -1.0)
    columns = constraints.shape[1] if constraints is not None and constraints.dim() > 1 else 0
    for index in range(block_count):
        fixed = columns > 0 and constraints[index, 0] != 0
        preplaced = columns > 1 and constraints[index, 1] != 0
        if preplaced:
            result[index] = torch.tensor(golden[index])
        elif fixed:
            result[index, 2:] = torch.tensor(golden[index][2:])
    return result


def _case_record(case_id, sample_index, inputs, golden, metrics):
    areas, b2b, p2b, pins, constraints = inputs
    block_count = int((areas != -1).sum().item())
    golden = list(golden[:block_count])
    return {
        "case_id": case_id,
        "sample_index": sample_index,
        "block_count": block_count,
        "areas": areas[:block_count],
        "b2b": b2b,
        "p2b": p2b,
        "pins": pins,
        "constraints": constraints[:block_count],
        "golden": golden,
        "baseline": _baseline(metrics),
        "optimizer_targets": _optimizer_targets(
            constraints[:block_count], golden, block_count
        ),
    }


def _load_public_cases(floorset_root: Path, official):
    dataset = official.FloorplanDatasetLiteTest(str(floorset_root))
    result = []
    for index in range(len(dataset)):
        sample = dataset[index]
        inputs = sample["input"]
        block_count = int((inputs[0] != -1).sum().item())
        golden = _polygon_positions(sample["label"][0], block_count)
        result.append(
            _case_record(
                f"public/{index}", index, inputs, golden, sample["label"][1]
            )
        )
    return result


def _load_clean_fold0_cases(floorset_root: Path, manifest_path: Path):
    data = json.loads(manifest_path.read_text())
    if data.get("schema_version") != 3 or data.get("split_unit") != "source_file":
        raise ValueError("heavy-clean manifest must use schema 3 source-file splits")
    if data.get("generation", {}).get("mib_policy") != (
        "input_area_interval_and_hard_target_compatible"
    ):
        raise ValueError("manifest is not the clean MIB-compatible panel")
    matches = [row for row in data.get("manifests", []) if int(row["fold"]) == 0]
    if len(matches) != 1:
        raise ValueError("manifest must contain exactly one fold 0")
    result = []
    cache_path = None
    cache = None
    for row in matches[0]["cases"]:
        source = floorset_root / row["source_file"]
        if source != cache_path:
            cache = torch.load(source, weights_only=False)
            cache_path = source
        offset = int(row["file_offset"])
        combined = cache[0][offset]
        inputs = (
            combined[:, 0],
            cache[1][offset],
            cache[2][offset],
            cache[3][offset],
            combined[:, 1:],
        )
        block_count = int((inputs[0] != -1).sum().item())
        golden = _training_positions(cache[5][offset], block_count)
        case = _case_record(
            row["case_id"],
            int(row["sample_index"]),
            inputs,
            golden,
            cache[6][offset],
        )
        case["source_file"] = row["source_file"]
        case["file_offset"] = offset
        result.append(case)
    return result, data


def _load_optimizer(solver_dir: Path):
    sys.path.insert(0, str(solver_dir))
    path = solver_dir / "my_optimizer.py"
    spec = importlib.util.spec_from_file_location("golden_plus_v32", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import optimizer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MyOptimizer(verbose=False)


def _load_public_positions(path: Path):
    data = json.loads(path.read_text())
    return {
        int(row["test_id"]): [tuple(map(float, rectangle)) for rectangle in row["positions"]]
        for row in data["test_results"]
    }, data


def _official_metrics(official, case, positions):
    return official.evaluate_solution(
        {"positions": positions, "runtime": 1.0},
        case["baseline"],
        case["constraints"],
        case["b2b"],
        case["p2b"],
        case["pins"],
        case["areas"],
        case["golden"],
        median_runtime=1.0,
    )


def _official_accept(before, after, config: RepairConfig) -> bool:
    hpwl_allowance = max(
        config.hpwl_abs_tolerance,
        config.hpwl_rel_tolerance * max(1.0, abs(before.hpwl_total)),
    )
    return (
        after.is_feasible
        and after.hpwl_total <= before.hpwl_total + hpwl_allowance
        and after.bbox_area <= before.bbox_area + config.bbox_abs_tolerance
        and after.boundary_violations <= before.boundary_violations
        and after.grouping_violations <= before.grouping_violations
        and after.mib_violations <= before.mib_violations
        and after.total_soft_violations < before.total_soft_violations
    )


def _metric_row(metrics):
    return {
        "is_feasible": bool(metrics.is_feasible),
        "cost": float(metrics.cost),
        "hpwl_total": float(metrics.hpwl_total),
        "hpwl_gap": float(metrics.hpwl_gap),
        "bbox_area": float(metrics.bbox_area),
        "area_gap": float(metrics.area_gap),
        "boundary_violations": int(metrics.boundary_violations),
        "grouping_violations": int(metrics.grouping_violations),
        "mib_violations": int(metrics.mib_violations),
        "total_soft_violations": int(metrics.total_soft_violations),
        "max_possible_violations": int(metrics.max_possible_violations),
        "violations_relative": float(metrics.violations_relative),
    }


def _run_repair(official, case, incumbent, config: RepairConfig):
    before = _official_metrics(official, case, incumbent)
    repaired, report = repair_fixed_topology(
        incumbent,
        case["areas"],
        case["b2b"],
        case["p2b"],
        case["pins"],
        case["constraints"],
        case["optimizer_targets"],
        config=config,
        return_report=True,
    )
    proposed = _official_metrics(official, case, repaired)
    official_rejected = bool(report.changed and not _official_accept(before, proposed, config))
    if official_rejected:
        repaired = incumbent
        after = before
        accepted = False
    else:
        after = proposed
        accepted = bool(report.changed)
    module_soft = report.after.soft if report.after is not None else None
    metric_mismatch = bool(
        module_soft is not None
        and (
            module_soft.boundary != proposed.boundary_violations
            or module_soft.grouping != proposed.grouping_violations
            or module_soft.mib != proposed.mib_violations
        )
    )
    return {
        "case_id": case["case_id"],
        "sample_index": case["sample_index"],
        "block_count": case["block_count"],
        "accepted": accepted,
        "official_rejected": official_rejected,
        "metric_mismatch": metric_mismatch,
        "repair_runtime_seconds": report.runtime_seconds,
        "attempted": report.attempted,
        "mechanism_accepts": report.accepted,
        "fallback_reason": report.fallback_reason,
        "before": _metric_row(before),
        "after": _metric_row(after),
    }, repaired


def _weights(rows):
    maximum = max(row["block_count"] for row in rows)
    raw = [math.exp((row["block_count"] - maximum) / 12.0) for row in rows]
    total = sum(raw)
    return [value / total for value in raw]


def _summary(official, rows, golden_costs=None):
    weights = _weights(rows)

    def weighted(section, key, *, clamp=False):
        values = [float(row[section][key]) for row in rows]
        if clamp:
            values = [max(0.0, value) for value in values]
        return sum(weight * value for weight, value in zip(weights, values))

    before_costs = [row["before"]["cost"] for row in rows]
    after_costs = [row["after"]["cost"] for row in rows]
    counts = [row["block_count"] for row in rows]
    runtimes = [row["repair_runtime_seconds"] for row in rows]
    accepted = [row for row in rows if row["accepted"]]
    result = {
        "cases": len(rows),
        "accepted_cases": len(accepted),
        "official_rejected_cases": sum(row["official_rejected"] for row in rows),
        "metric_mismatch_cases": sum(row["metric_mismatch"] for row in rows),
        "feasible_before": sum(row["before"]["is_feasible"] for row in rows),
        "feasible_after": sum(row["after"]["is_feasible"] for row in rows),
        "rf1_score_before": official.compute_total_score(before_costs, counts),
        "rf1_score_after": official.compute_total_score(after_costs, counts),
        "rf1_score_delta": official.compute_total_score(after_costs, counts)
        - official.compute_total_score(before_costs, counts),
        "weighted_hpwl_gap_before": weighted("before", "hpwl_gap", clamp=True),
        "weighted_hpwl_gap_after": weighted("after", "hpwl_gap", clamp=True),
        "weighted_area_gap_before": weighted("before", "area_gap", clamp=True),
        "weighted_area_gap_after": weighted("after", "area_gap", clamp=True),
        "weighted_violations_relative_before": weighted(
            "before", "violations_relative"
        ),
        "weighted_violations_relative_after": weighted(
            "after", "violations_relative"
        ),
        "violations_removed": {
            key: sum(row["before"][key] - row["after"][key] for row in rows)
            for key in (
                "boundary_violations",
                "grouping_violations",
                "mib_violations",
                "total_soft_violations",
            )
        },
        "hpwl_delta_sum": sum(
            row["after"]["hpwl_total"] - row["before"]["hpwl_total"]
            for row in rows
        ),
        "hpwl_delta_max": max(
            row["after"]["hpwl_total"] - row["before"]["hpwl_total"]
            for row in rows
        ),
        "bbox_area_delta_sum": sum(
            row["after"]["bbox_area"] - row["before"]["bbox_area"]
            for row in rows
        ),
        "bbox_area_delta_max": max(
            row["after"]["bbox_area"] - row["before"]["bbox_area"]
            for row in rows
        ),
        "repair_runtime_mean": statistics.fmean(runtimes),
        "repair_runtime_p95": sorted(runtimes)[max(0, math.ceil(0.95 * len(runtimes)) - 1)],
        "repair_runtime_max": max(runtimes),
        "attempted": {
            key: sum(row["attempted"][key] for row in rows)
            for key in ("boundary", "mib", "grouping")
        },
        "mechanism_accepts": {
            key: sum(row["mechanism_accepts"][key] for row in rows)
            for key in ("boundary", "mib", "grouping")
        },
    }
    if golden_costs is not None:
        golden_score = official.compute_total_score(golden_costs, counts)
        result.update(
            {
                "golden_rf1_score": golden_score,
                "beats_golden_cases": sum(
                    row["after"]["cost"] < golden - 1e-12
                    for row, golden in zip(rows, golden_costs)
                ),
                "aggregate_beats_golden": result["rf1_score_after"]
                < golden_score - 1e-12,
            }
        )
    return result


def _ablation_configs():
    base = RepairConfig()
    return {
        "boundary_only": replace(
            base, enable_boundary=True, enable_mib=False, enable_grouping=False
        ),
        "mib_only": replace(
            base, enable_boundary=False, enable_mib=True, enable_grouping=False
        ),
        "grouping_only": replace(
            base, enable_boundary=False, enable_mib=False, enable_grouping=True
        ),
        "combined": base,
    }


def _evaluate_panel(
    official,
    cases,
    optimizer,
    public_positions,
    panel_name,
    *,
    include_v32,
    include_golden_ablations,
    v32_ablation,
):
    golden_metrics = [_official_metrics(official, case, case["golden"]) for case in cases]
    golden_costs = [metrics.cost for metrics in golden_metrics]
    output = {"golden_ablations": {}}
    if include_golden_ablations:
        for name, config in _ablation_configs().items():
            rows = []
            for case in cases:
                row, _positions = _run_repair(
                    official, case, case["golden"], config
                )
                rows.append(row)
            output["golden_ablations"][name] = {
                "summary": _summary(official, rows, golden_costs),
                "cases": rows,
            }

    if include_v32:
        v32_config = _ablation_configs()[v32_ablation]
        rows = []
        solver_runtime = []
        for case in cases:
            if panel_name == "public" and public_positions is not None:
                incumbent = public_positions[case["sample_index"]]
                solver_runtime.append(None)
            else:
                started = time.perf_counter()
                incumbent = optimizer.solve(
                    case["block_count"],
                    case["areas"],
                    case["b2b"],
                    case["p2b"],
                    case["pins"],
                    case["constraints"],
                    case["optimizer_targets"],
                )
                solver_runtime.append(time.perf_counter() - started)
            row, _positions = _run_repair(
                official, case, incumbent, v32_config
            )
            row["solver_runtime_seconds"] = solver_runtime[-1]
            rows.append(row)
        summary = _summary(official, rows, golden_costs)
        measured = [value for value in solver_runtime if value is not None]
        summary["solver_runtime_mean"] = (
            statistics.fmean(measured) if measured else None
        )
        summary["solver_runtime_max"] = max(measured) if measured else None
        output[f"v32_{v32_ablation}"] = {"summary": summary, "cases": rows}
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--floorset-root", type=Path, default=ROOT / "external" / "FloorSet"
    )
    parser.add_argument(
        "--v32-ablation",
        choices=tuple(_ablation_configs()),
        default="combined",
    )
    parser.add_argument(
        "--fold-manifest",
        type=Path,
        default=ROOT / "results" / "folds" / "heavy_clean_v1.json",
    )
    parser.add_argument("--solver-dir", type=Path, default=SOLUTION_DIR)
    parser.add_argument("--public-v32-result", type=Path)
    parser.add_argument("--skip-v32", action="store_true")
    parser.add_argument(
        "--v32-only",
        action="store_true",
        help="skip repeated golden ablations and evaluate only v32 + repair",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    official = _load_official(args.floorset_root.resolve())
    public_cases = _load_public_cases(args.floorset_root.resolve(), official)
    clean_cases, manifest = _load_clean_fold0_cases(
        args.floorset_root.resolve(), args.fold_manifest.resolve()
    )
    optimizer = None if args.skip_v32 else _load_optimizer(args.solver_dir.resolve())
    public_positions = None
    public_result_meta = None
    if not args.skip_v32 and args.public_v32_result is not None:
        public_positions, public_result_meta = _load_public_positions(
            args.public_v32_result.resolve()
        )

    started = time.time()
    panels = {
        "public": _evaluate_panel(
            official,
            public_cases,
            optimizer,
            public_positions,
            "public",
            include_v32=not args.skip_v32,
            include_golden_ablations=not args.v32_only,
            v32_ablation=args.v32_ablation,
        ),
        "heavy_clean_v1_fold0": _evaluate_panel(
            official,
            clean_cases,
            optimizer,
            public_positions,
            "heavy_clean_v1_fold0",
            include_v32=not args.skip_v32,
            include_golden_ablations=not args.v32_only,
            v32_ablation=args.v32_ablation,
        ),
    }
    result = {
        "schema_version": 1,
        "experiment": "generic_fixed_topology_golden_plus_repair",
        "scope": {
            "public": True,
            "heavy_clean_v1_fold": 0,
            "sealed_v2_accessed": False,
        },
        "provenance": {
            "repository_commit": _git_commit(ROOT),
            "floorset_commit": _git_commit(args.floorset_root.resolve()),
            "official_evaluator_sha256": _sha256(
                args.floorset_root.resolve()
                / "iccad2026contest"
                / "iccad2026_evaluate.py"
            ),
            "fold_manifest": str(args.fold_manifest),
            "fold_manifest_sha256": _sha256(args.fold_manifest.resolve()),
            "fold_manifest_generation": manifest["generation"],
            "repair_source_sha256": _sha256(
                SOLUTION_DIR / "golden_plus_repair.py"
            ),
            "public_v32_result": str(args.public_v32_result)
            if args.public_v32_result
            else None,
            "public_v32_result_sha256": _sha256(args.public_v32_result.resolve())
            if args.public_v32_result
            else None,
            "public_v32_recorded_score": public_result_meta.get("total_score")
            if public_result_meta
            else None,
        },
        "config": RepairConfig().__dict__,
        "panels": panels,
        "wall_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    for panel, data in panels.items():
        print(panel)
        for name, experiment in data["golden_ablations"].items():
            summary = experiment["summary"]
            print(
                f"  golden {name}: {summary['rf1_score_before']:.9f} -> "
                f"{summary['rf1_score_after']:.9f} "
                f"({summary['rf1_score_delta']:+.9f}), accepted="
                f"{summary['accepted_cases']}, removed="
                f"{summary['violations_removed']['total_soft_violations']}"
            )
        v32_key = f"v32_{args.v32_ablation}"
        if v32_key in data:
            summary = data[v32_key]["summary"]
            print(
                f"  v32 {args.v32_ablation}: {summary['rf1_score_before']:.9f} -> "
                f"{summary['rf1_score_after']:.9f} "
                f"({summary['rf1_score_delta']:+.9f}), accepted="
                f"{summary['accepted_cases']}, removed="
                f"{summary['violations_removed']['total_soft_violations']}"
            )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
