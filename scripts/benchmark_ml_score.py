#!/usr/bin/env python3
"""Compare heuristic and guarded ML optimizers with official contest cost."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
CONTEST_DIR = Path("/workspace/eda/FloorSet/iccad2026contest")
DATA_DIR = Path("/workspace/eda/FloorSet")

sys.path.insert(0, str(CONTEST_DIR))
from iccad2026_evaluate import ContestEvaluator, compute_total_score, evaluate_solution

sys.path.insert(0, str(ROOT / "contest_solution"))
from my_optimizer import MyOptimizer


def build_optimizer_target_positions(block_count: int, constraints: torch.Tensor, target_pos: list | None) -> torch.Tensor:
    opt_target_pos = torch.full((block_count, 4), -1.0)
    if target_pos is None or constraints is None:
        return opt_target_pos

    ncols = constraints.shape[1] if constraints.dim() > 1 else 0
    for i in range(block_count):
        is_fixed = ncols > 0 and constraints[i, 0] != 0
        is_preplaced = ncols > 1 and constraints[i, 1] != 0
        if is_preplaced:
            tx, ty, tw, th = target_pos[i]
            opt_target_pos[i] = torch.tensor([tx, ty, tw, th])
        elif is_fixed:
            _, _, tw, th = target_pos[i]
            opt_target_pos[i, 2] = tw
            opt_target_pos[i, 3] = th
    return opt_target_pos


def summarize(costs: list[float], blocks: list[int]) -> dict[str, float]:
    if not costs:
        return {"total_score": 0.0, "avg_cost": 0.0, "min_cost": 0.0, "max_cost": 0.0}
    return {
        "total_score": compute_total_score(costs, blocks),
        "avg_cost": sum(costs) / len(costs),
        "min_cost": min(costs),
        "max_cost": max(costs),
    }


def apply_layout_variant(optimizer: MyOptimizer, variant: tuple[float, float, float] | None) -> None:
    if variant is None:
        return

    optimizer._layout_variants = lambda block_count: [variant]


def run_benchmark(
    num_cases: int,
    model: Path | None,
    output: Path,
    data_dir: Path,
    case_index: int | None,
    variant: tuple[float, float, float] | None,
) -> dict[str, Any]:
    evaluator = ContestEvaluator(data_path=str(data_dir), verbose=False)
    evaluator._load_dataset()

    heuristic = MyOptimizer()
    ml = MyOptimizer()
    apply_layout_variant(heuristic, variant)
    apply_layout_variant(ml, variant)
    ml_available = False

    case_indices = [case_index] if case_index is not None else list(range(num_cases))

    heuristic_costs: list[float] = []
    ml_costs: list[float] = []
    blocks: list[int] = []
    cases: list[dict[str, Any]] = []

    for idx in case_indices:
        sample = evaluator.dataset[idx]
        inputs, labels = sample["input"], sample["label"]
        area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
        block_count = int((area_target != -1).sum().item())
        print(f"Evaluating case={idx} blocks={block_count}", flush=True)
        baseline, target_pos = evaluator._extract_baseline(
            idx, labels, b2b_conn, p2b_conn, pins_pos, block_count
        )
        opt_target_pos = build_optimizer_target_positions(block_count, constraints, target_pos)

        heuristic_positions = heuristic.solve(
            block_count, area_target, b2b_conn, p2b_conn, pins_pos, constraints, opt_target_pos
        )
        heuristic_metrics = evaluate_solution(
            {"positions": heuristic_positions, "runtime": 1.0},
            baseline,
            constraints,
            b2b_conn,
            p2b_conn,
            pins_pos,
            area_target,
            target_pos,
            median_runtime=1.0,
        )
        heuristic_costs.append(heuristic_metrics.cost)
        blocks.append(block_count)

        case: dict[str, Any] = {
            "case": idx,
            "blocks": block_count,
            "heuristic_cost": heuristic_metrics.cost,
            "heuristic_feasible": heuristic_metrics.is_feasible,
            "heuristic_hpwl_gap": heuristic_metrics.hpwl_gap,
            "heuristic_area_gap": heuristic_metrics.area_gap,
            "heuristic_soft": heuristic_metrics.total_soft_violations,
        }

        if ml_available:
            ml_positions = ml.solve(
                block_count, area_target, b2b_conn, p2b_conn, pins_pos, constraints, opt_target_pos
            )
            ml_metrics = evaluate_solution(
                {"positions": ml_positions, "runtime": 1.0},
                baseline,
                constraints,
                b2b_conn,
                p2b_conn,
                pins_pos,
                area_target,
                target_pos,
                median_runtime=1.0,
            )
            ml_costs.append(ml_metrics.cost)
            case.update({
                "ml_cost": ml_metrics.cost,
                "ml_feasible": ml_metrics.is_feasible,
                "ml_hpwl_gap": ml_metrics.hpwl_gap,
                "ml_area_gap": ml_metrics.area_gap,
                "ml_soft": ml_metrics.total_soft_violations,
                "same_as_heuristic": ml_positions == heuristic_positions,
            })

        cases.append(case)

    result: dict[str, Any] = {
        "num_cases": len(case_indices),
        "model": str(model) if model else None,
        "ml_available": ml_available,
        "heuristic": summarize(heuristic_costs, blocks),
        "cases": cases,
    }
    if ml_available:
        result["ml"] = summarize(ml_costs, blocks)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-cases", type=int, default=10)
    parser.add_argument("--case-index", type=int, default=None)
    parser.add_argument("--variant", nargs=3, type=float, default=None, metavar=("ROW", "SMALL", "LARGE"))
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "benchmark_ml_score.json")
    args = parser.parse_args()

    variant = tuple(args.variant) if args.variant is not None else None
    result = run_benchmark(
        args.num_cases,
        args.model,
        args.output,
        args.data_dir,
        args.case_index,
        variant,
    )
    print(json.dumps(result, indent=2))
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
