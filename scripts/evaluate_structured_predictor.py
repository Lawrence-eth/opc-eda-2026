#!/usr/bin/env python3
"""Evaluate a structured artifact on development v1 folds only.

This harness deliberately refuses sealed manifests.  It reports supervised
label accuracy, decoded full-layout exactness, hard feasibility, RF=1 scores,
paired inference time, the confidence-gated fallback result, and an offline
v32-best-of upper bound.  The offline best-of uses golden baselines and is
diagnostic only; it is never represented as a deployable selector.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SOLUTION_DIR = ROOT / "contest_solution"
DEFAULT_DATA_ROOT = ROOT / "external" / "FloorSet"
sys.path.insert(0, str(SOLUTION_DIR))

from dual_parent_decoder import (  # noqa: E402
    compare_geometry,
    extract_oracle_labels,
    hard_targets_from_golden,
    training_rectangles,
)
from structured_predictor import load_artifact, predict_candidate  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_optimizer(path: Path):
    sys.path.insert(0, str(path.resolve()))
    module = _load_module(path / "my_optimizer.py", "_structured_v32_optimizer")
    return module.MyOptimizer(verbose=False)


def _canonical_source(path: str | Path, data_root: Path) -> str:
    return Path(path).resolve().relative_to(data_root.resolve()).as_posix()


def _selected_cases(dataset: Any, data_root: Path, path: Path, folds: set[int]):
    if "sealed" in path.name.lower():
        raise ValueError("sealed manifests are prohibited in development evaluation")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("split_unit") != "source_file":
        raise ValueError("evaluation manifest must be source-file-disjoint")
    by_source = {
        _canonical_source(source, data_root): index
        for index, source in enumerate(dataset.all_files)
    }
    selected = []
    for manifest in data.get("manifests", [data]):
        fold = int(manifest.get("fold", 0))
        if fold not in folds:
            continue
        role = str(manifest.get("role", ""))
        if "sealed" in role.lower():
            raise ValueError("sealed manifest roles are prohibited")
        for case in manifest.get("cases", []):
            source = str(case["source_file"])
            if source not in by_source:
                raise ValueError(f"manifest source is absent: {source}")
            offset = int(case["file_offset"])
            if not 0 <= offset < int(dataset.layouts_per_file):
                raise ValueError(f"manifest offset is invalid: {source}#{offset}")
            selected.append(
                (
                    by_source[source] * int(dataset.layouts_per_file) + offset,
                    source,
                    offset,
                    fold,
                )
            )
    if not selected:
        raise ValueError("fold selection contains no development cases")
    return selected


def _baseline(stored_metrics: Any) -> dict[str, float]:
    return {
        "area_baseline": float(stored_metrics[0]),
        "hpwl_baseline": float(stored_metrics[-2]) + float(stored_metrics[-1]),
    }


def _targets_tensor(rows: list[tuple[float, float, float, float]]):
    return torch.tensor(rows, dtype=torch.float32)


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": p95,
        "max": ordered[-1],
    }


def _labels_equal(predicted: Any, expected: Any) -> dict[str, int | bool]:
    if predicted is None:
        return {
            "shape_correct": 0,
            "shape_total": len(expected.selected_shape_indices),
            "root_correct": False,
            "horizontal_correct": 0,
            "horizontal_total": len(expected.horizontal),
            "vertical_correct": 0,
            "vertical_total": len(expected.vertical_supports),
            "full_exact": False,
        }
    shape_correct = sum(
        left == right
        for left, right in zip(
            predicted.selected_shape_indices, expected.selected_shape_indices
        )
    )
    expected_h = {
        row.child: (row.parent, row.side) for row in expected.horizontal
    }
    predicted_h = {
        row.child: (row.parent, row.side) for row in predicted.horizontal
    }
    horizontal_correct = sum(
        predicted_h.get(child) == relation for child, relation in expected_h.items()
    )
    vertical_correct = sum(
        left == right
        for left, right in zip(predicted.vertical_supports, expected.vertical_supports)
    )
    root_correct = predicted.root == expected.root
    full = (
        shape_correct == len(expected.selected_shape_indices)
        and root_correct
        and horizontal_correct == len(expected.horizontal)
        and vertical_correct == len(expected.vertical_supports)
    )
    return {
        "shape_correct": shape_correct,
        "shape_total": len(expected.selected_shape_indices),
        "root_correct": root_correct,
        "horizontal_correct": horizontal_correct,
        "horizontal_total": len(expected.horizontal),
        "vertical_correct": vertical_correct,
        "vertical_total": len(expected.vertical_supports),
        "full_exact": full,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    data_root = args.data_root.resolve()
    sys.path.insert(0, str(data_root))
    dataset_module = _load_module(data_root / "lite_dataset.py", "_structured_dataset")
    evaluator = _load_module(
        data_root / "iccad2026contest" / "iccad2026_evaluate.py",
        "_structured_evaluator",
    )
    dataset = dataset_module.FloorplanDatasetLite(str(data_root))
    dataset.all_files = sorted(str(Path(path).resolve()) for path in dataset.all_files)
    if hasattr(dataset, "cached_file_idx"):
        dataset.cached_file_idx = -1
    folds = set(args.fold)
    selected = _selected_cases(dataset, data_root, args.manifest, folds)
    model = load_artifact(args.model)
    optimizer = _load_optimizer(args.solver_dir)
    modes = ("learned", "skyline")
    aggregate = {
        mode: {
            "counts": Counter(),
            "candidate_costs": [],
            "fallback_costs": [],
            "best_of_costs": [],
            "model_runtime": [],
            "paired_runtime_delta": [],
            "confidence": [],
            "failure_taxonomy": Counter(),
        }
        for mode in modes
    }
    baseline_costs = []
    block_counts = []
    baseline_runtime = []

    for ordinal, (sample_index, _source, _offset, _fold) in enumerate(selected, 1):
        sample = dataset[sample_index]
        area, b2b, p2b, pins, constraints = sample["input"]
        tree, fp_solution, stored_metrics = sample["label"]
        count = int((area != -1).sum().item())
        golden = training_rectangles(fp_solution, count)
        oracle = extract_oracle_labels(area, constraints, tree, golden)
        targets = hard_targets_from_golden(constraints, golden)
        baseline = _baseline(stored_metrics)

        start = time.perf_counter()
        baseline_positions = optimizer.solve(
            count,
            area.clone(),
            b2b.clone(),
            p2b.clone(),
            pins.clone(),
            constraints.clone(),
            _targets_tensor(targets),
        )
        elapsed_baseline = time.perf_counter() - start
        baseline_metrics = evaluator.evaluate_solution(
            {"positions": baseline_positions, "runtime": 1.0},
            baseline,
            constraints,
            b2b,
            p2b,
            pins,
            area,
            golden,
            median_runtime=1.0,
        )
        baseline_costs.append(float(baseline_metrics.cost))
        baseline_runtime.append(elapsed_baseline)
        block_counts.append(count)

        for mode in modes:
            start = time.perf_counter()
            prediction = predict_candidate(
                model,
                count,
                area,
                b2b,
                p2b,
                pins,
                constraints,
                targets,
                vertical_mode=mode,
            )
            elapsed_model = time.perf_counter() - start
            rows = aggregate[mode]
            rows["model_runtime"].append(elapsed_model)
            rows["paired_runtime_delta"].append(elapsed_model)
            rows["confidence"].append(prediction.confidence)
            rows["failure_taxonomy"][prediction.reason] += 1
            labels = _labels_equal(prediction.labels, oracle)
            for key, value in labels.items():
                rows["counts"][key] += int(value)

            geometry_exact = False
            if prediction.positions is not None:
                comparison = compare_geometry(prediction.positions, golden)
                geometry_exact = comparison.is_exact()
                candidate_metrics = evaluator.evaluate_solution(
                    {"positions": prediction.positions, "runtime": 1.0},
                    baseline,
                    constraints,
                    b2b,
                    p2b,
                    pins,
                    area,
                    golden,
                    median_runtime=1.0,
                )
                candidate_cost = float(candidate_metrics.cost)
                feasible = bool(candidate_metrics.is_feasible)
            else:
                candidate_cost = 10.0
                feasible = False
            rows["counts"]["candidate_feasible"] += int(feasible)
            rows["counts"]["geometry_exact"] += int(geometry_exact)
            rows["counts"]["full_layout_exact"] += int(
                geometry_exact and labels["full_exact"]
            )
            rows["candidate_costs"].append(candidate_cost)
            accepted = (
                feasible
                and prediction.confidence
                >= float(model["calibration"]["confidence_threshold"])
            )
            rows["counts"]["confidence_accepted"] += int(accepted)
            rows["fallback_costs"].append(
                candidate_cost if accepted else float(baseline_metrics.cost)
            )
            rows["counts"]["produced_feasible"] += int(
                feasible if accepted else bool(baseline_metrics.is_feasible)
            )
            best = min(candidate_cost, float(baseline_metrics.cost))
            rows["best_of_costs"].append(best)
            rows["counts"]["offline_best_of_wins"] += int(
                candidate_cost < float(baseline_metrics.cost) - 1e-12
            )

        if ordinal % 10 == 0 or ordinal == len(selected):
            print(f"evaluated {ordinal}/{len(selected)} development cases", flush=True)

    cases = len(selected)
    result_modes = {}
    for mode, rows in aggregate.items():
        counts = rows["counts"]
        result_modes[mode] = {
            "label_accuracy": {
                "shape": counts["shape_correct"] / max(1, counts["shape_total"]),
                "horizontal_root": counts["root_correct"] / cases,
                "horizontal_edge": counts["horizontal_correct"] / max(1, counts["horizontal_total"]),
                "vertical_support_or_floor": counts["vertical_correct"] / max(1, counts["vertical_total"]),
                "full_label_exact_rate": counts["full_exact"] / cases,
            },
            "decoded": {
                "candidate_feasible": counts["candidate_feasible"],
                "candidate_feasible_rate": counts["candidate_feasible"] / cases,
                "geometry_exact": counts["geometry_exact"],
                "full_layout_exact": counts["full_layout_exact"],
                "confidence_accepted": counts["confidence_accepted"],
                "produced_with_fallback_feasible": counts["produced_feasible"],
                "produced_with_fallback_feasible_rate": counts["produced_feasible"] / cases,
                "failure_taxonomy": dict(sorted(rows["failure_taxonomy"].items())),
            },
            "score_rf1": {
                "candidate_only": evaluator.compute_total_score(rows["candidate_costs"], block_counts),
                "confidence_gated_v32_fallback": evaluator.compute_total_score(rows["fallback_costs"], block_counts),
                "offline_v32_best_of": evaluator.compute_total_score(rows["best_of_costs"], block_counts),
                "offline_best_of_wins": counts["offline_best_of_wins"],
                "offline_best_of_weighted_delta": evaluator.compute_total_score(rows["best_of_costs"], block_counts)
                - evaluator.compute_total_score(baseline_costs, block_counts),
            },
            "runtime_seconds": {
                "model_only": _stats(rows["model_runtime"]),
                "increment_over_v32": _stats(rows["paired_runtime_delta"]),
            },
            "confidence": _stats(rows["confidence"]),
        }
    return {
        "schema_version": 1,
        "experiment": "structured_predictor_development_v1",
        "contract": {
            "sealed_v2_access": "prohibited",
            "manifest_policy": "existing v1 development folds only",
            "offline_best_of_is_deployable": False,
            "metric_runtime_factor": 1.0,
        },
        "provenance": {
            "manifest": str(args.manifest),
            "manifest_sha256": _sha256(args.manifest),
            "folds": sorted(folds),
            "model": str(args.model),
            "model_sha256": _sha256(args.model),
            "solver_dir": str(args.solver_dir),
            "official_evaluator_sha256": _sha256(
                data_root / "iccad2026contest" / "iccad2026_evaluate.py"
            ),
        },
        "summary": {
            "cases": cases,
            "baseline_v32_score_rf1": evaluator.compute_total_score(
                baseline_costs, block_counts
            ),
            "baseline_v32_runtime_seconds": _stats(baseline_runtime),
        },
        "modes": result_modes,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fold", type=int, action="append", default=[])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--solver-dir", type=Path, default=SOLUTION_DIR)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.fold:
        args.fold = [0, 1, 2]
    result = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
