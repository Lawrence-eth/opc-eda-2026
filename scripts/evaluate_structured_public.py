#!/usr/bin/env python3
"""Score structured candidates and fail-closed v32 fallback on public cases."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
SOLUTION_DIR = ROOT / "contest_solution"
DEFAULT_DATA_ROOT = ROOT / "external" / "FloorSet"
SCRIPTS_DIR = ROOT / "scripts"
sys.path[:0] = [str(SOLUTION_DIR), str(SCRIPTS_DIR)]

from dual_parent_decoder import compare_geometry, hard_targets_from_golden  # noqa: E402
from evaluate_structured_predictor import _stats  # noqa: E402
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
    module = _load_module(path / "my_optimizer.py", "_structured_public_v32")
    return module.MyOptimizer(verbose=False)


def evaluate_public(args: argparse.Namespace):
    data_root = args.data_root.resolve()
    sys.path[:0] = [str(data_root / "iccad2026contest"), str(data_root)]
    evaluator = _load_module(
        data_root / "iccad2026contest" / "iccad2026_evaluate.py",
        "_structured_public_evaluator",
    )
    contest = evaluator.ContestEvaluator(data_path=str(data_root), verbose=False)
    contest._load_dataset()
    model = load_artifact(args.model)
    optimizer = _load_optimizer(args.solver_dir)
    modes = ("learned", "skyline")
    aggregate = {
        mode: {
            "counts": Counter(),
            "candidate_costs": [],
            "produced_costs": [],
            "best_of_costs": [],
            "runtime": [],
            "confidence": [],
            "failure_taxonomy": Counter(),
        }
        for mode in modes
    }
    baseline_costs = []
    baseline_runtime = []
    block_counts = []

    for index in range(len(contest.dataset)):
        sample = contest.dataset[index]
        area, b2b, p2b, pins, constraints = sample["input"]
        count = int((area != -1).sum().item())
        baseline, golden = contest._extract_baseline(
            index, sample["label"], b2b, p2b, pins, count
        )
        targets = hard_targets_from_golden(constraints, golden)
        target_tensor = torch.tensor(targets, dtype=torch.float32)
        started = time.perf_counter()
        baseline_positions = optimizer.solve(
            count,
            area.clone(),
            b2b.clone(),
            p2b.clone(),
            pins.clone(),
            constraints.clone(),
            target_tensor,
        )
        elapsed_baseline = time.perf_counter() - started
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
            started = time.perf_counter()
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
            elapsed = time.perf_counter() - started
            rows = aggregate[mode]
            rows["runtime"].append(elapsed)
            rows["confidence"].append(prediction.confidence)
            rows["failure_taxonomy"][prediction.reason] += 1
            shape_correct = 0
            if prediction.labels is not None:
                shape_correct = sum(
                    abs(prediction.labels.dimensions[block][0] - golden[block][2]) <= 1e-6
                    and abs(prediction.labels.dimensions[block][1] - golden[block][3]) <= 1e-6
                    for block in range(count)
                )
            rows["counts"]["shape_correct"] += shape_correct
            rows["counts"]["shape_total"] += count

            geometry_exact = False
            if prediction.positions is not None:
                geometry_exact = compare_geometry(prediction.positions, golden).is_exact()
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
            rows["candidate_costs"].append(candidate_cost)
            accepted = (
                feasible
                and prediction.confidence
                >= float(model["calibration"]["confidence_threshold"])
            )
            rows["counts"]["confidence_accepted"] += int(accepted)
            produced_cost = candidate_cost if accepted else float(baseline_metrics.cost)
            produced_feasible = feasible if accepted else bool(baseline_metrics.is_feasible)
            rows["produced_costs"].append(produced_cost)
            rows["counts"]["produced_feasible"] += int(produced_feasible)
            rows["best_of_costs"].append(min(candidate_cost, float(baseline_metrics.cost)))
            rows["counts"]["offline_best_of_wins"] += int(
                candidate_cost < float(baseline_metrics.cost) - 1e-12
            )
        if (index + 1) % 10 == 0:
            print(f"evaluated {index + 1}/{len(contest.dataset)} public cases", flush=True)

    baseline_score = evaluator.compute_total_score(baseline_costs, block_counts)
    result_modes = {}
    cases = len(block_counts)
    for mode, rows in aggregate.items():
        counts = rows["counts"]
        best_score = evaluator.compute_total_score(rows["best_of_costs"], block_counts)
        result_modes[mode] = {
            "shape_accuracy": counts["shape_correct"] / max(1, counts["shape_total"]),
            "candidate": {
                "feasible": counts["candidate_feasible"],
                "feasible_rate": counts["candidate_feasible"] / cases,
                "geometry_exact": counts["geometry_exact"],
                "score_rf1": evaluator.compute_total_score(rows["candidate_costs"], block_counts),
                "failure_taxonomy": dict(sorted(rows["failure_taxonomy"].items())),
            },
            "confidence_gated_v32_fallback": {
                "accepted": counts["confidence_accepted"],
                "feasible": counts["produced_feasible"],
                "feasible_rate": counts["produced_feasible"] / cases,
                "score_rf1": evaluator.compute_total_score(rows["produced_costs"], block_counts),
            },
            "offline_v32_best_of": {
                "score_rf1": best_score,
                "wins": counts["offline_best_of_wins"],
                "weighted_delta": best_score - baseline_score,
                "deployable": False,
            },
            "runtime_seconds": {
                "model_increment": _stats(rows["runtime"]),
                "v32_plus_model": _stats(
                    [left + right for left, right in zip(baseline_runtime, rows["runtime"])]
                ),
            },
            "confidence": _stats(rows["confidence"]),
        }
    return {
        "schema_version": 1,
        "experiment": "structured_predictor_public_validation",
        "contract": {
            "sealed_v2_access": "none",
            "metric_runtime_factor": 1.0,
            "offline_best_of_is_deployable": False,
        },
        "provenance": {
            "model": str(args.model),
            "model_sha256": _sha256(args.model),
            "official_evaluator_sha256": _sha256(
                data_root / "iccad2026contest" / "iccad2026_evaluate.py"
            ),
        },
        "summary": {
            "cases": cases,
            "baseline_v32_score_rf1": baseline_score,
            "baseline_v32_runtime_seconds": _stats(baseline_runtime),
        },
        "modes": result_modes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--solver-dir", type=Path, default=SOLUTION_DIR)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_public(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
