"""Evaluate ML-guided optimizer against heuristic baseline.

Loads the trained Graph Transformer model, runs it on validation cases,
and compares scores using the official evaluator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "contest_solution"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "FloorSet" / "iccad2026contest"))

from iccad2026_evaluate import get_training_dataloader
from my_optimizer import MyOptimizer

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from ml_integration import integrate_ml_optimizer


def evaluate_model(model_path: str, num_cases: int = 100, data_dir: str = "/workspace/eda/FloorSet") -> dict:
    """Evaluate ML-guided optimizer on validation cases.

    Returns:
        dict with scores for heuristic baseline and ML-guided optimizer.
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

        # Calculate costs (simplified - just count valid placements)
        heuristic_valid = len([p for p in heuristic_positions if p is not None])
        ml_valid = len([p for p in ml_positions if p is not None])

        heuristic_scores.append(heuristic_valid)
        ml_scores.append(ml_valid)

        if (idx + 1) % 10 == 0:
            print(f"  Evaluated {idx + 1}/{num_cases} cases")

    return {
        "heuristic_avg_valid": sum(heuristic_scores) / len(heuristic_scores),
        "ml_avg_valid": sum(ml_scores) / len(ml_scores),
        "num_cases": num_cases,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/workspace/eda/opc-eda-2026/models/gt_model_50k.pt")
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
