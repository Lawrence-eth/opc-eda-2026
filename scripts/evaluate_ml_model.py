"""Evaluate ML-guided optimizer on validation cases.

Usage:
    python scripts/evaluate_ml_model.py --model models/gt_model_50k_v2.pt --num-cases 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "contest_solution"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "FloorSet" / "iccad2026contest"))
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from iccad2026_evaluate import get_training_dataloader
from my_optimizer import MyOptimizer
from ml_integration import integrate_ml_optimizer


def evaluate_model(model_path: str, num_cases: int = 100, data_dir: str = "/workspace/eda/FloorSet") -> dict:
    """Evaluate ML-guided optimizer on validation cases.

    Returns:
        dict with scores and comparison metrics.
    """
    # Load data
    dataloader = get_training_dataloader(
        data_dir,
        batch_size=1,
        num_samples=num_cases,
        shuffle=False,
    )

    # Create optimizers
    heuristic_opt = MyOptimizer()
    ml_opt = MyOptimizer()

    # Try to integrate ML
    ml_available = integrate_ml_optimizer(ml_opt, model_path)
    if not ml_available:
        print(f"Model not found at {model_path}, cannot evaluate ML-guided optimizer")
        return {"error": "model_not_found"}

    heuristic_scores = []
    ml_scores = []
    case_details = []

    for idx, batch in enumerate(dataloader):
        (area_target, b2b_conn, p2b_conn, pins_pos,
         constraints, tree_sol, fp_sol, metrics) = batch

        block_count = int((area_target[0] > 0).sum().item())

        # Heuristic solve
        heuristic_positions = heuristic_opt.solve(
            block_count, area_target[0], b2b_conn[0], p2b_conn[0],
            pins_pos[0], constraints[0], fp_sol[0]
        )

        # ML solve
        ml_positions = ml_opt.solve(
            block_count, area_target[0], b2b_conn[0], p2b_conn[0],
            pins_pos[0], constraints[0], fp_sol[0]
        )

        # Calculate placement validity
        heuristic_valid = len([p for p in heuristic_positions if p is not None])
        ml_valid = len([p for p in ml_positions if p is not None])

        # Count overlaps
        heuristic_overlap = count_overlaps(heuristic_positions)
        ml_overlap = count_overlaps(ml_positions)

        heuristic_scores.append(heuristic_valid)
        ml_scores.append(ml_valid)

        case_details.append({
            "case": idx,
            "n_blocks": block_count,
            "heuristic_valid": heuristic_valid,
            "ml_valid": ml_valid,
            "heuristic_overlap": heuristic_overlap,
            "ml_overlap": ml_overlap,
        })

        if (idx + 1) % 10 == 0:
            print(f"  Evaluated {idx + 1}/{num_cases} cases")

    return {
        "heuristic_avg_valid": sum(heuristic_scores) / len(heuristic_scores),
        "ml_avg_valid": sum(ml_scores) / len(ml_scores),
        "num_cases": num_cases,
        "cases": case_details,
    }


def count_overlaps(positions: list) -> int:
    """Count overlapping pairs in a layout."""
    rects = []
    for p in positions:
        if p is not None:
            rects.append((p[0], p[1], p[2], p[3]))

    overlap_count = 0
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            x1, y1, w1, h1 = rects[i]
            x2, y2, w2, h2 = rects[j]
            if not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1):
                overlap_count += 1
    return overlap_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/workspace/eda/opc-eda-2026/models/gt_model_50k_v2.pt")
    parser.add_argument("--num-cases", type=int, default=100)
    parser.add_argument("--output", default="/workspace/eda/opc-eda-2026/results/ml_evaluation.json")
    args = parser.parse_args()

    print(f"Evaluating model: {args.model}")
    results = evaluate_model(args.model, args.num_cases)

    print(json.dumps(results, indent=2))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
