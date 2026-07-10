#!/usr/bin/env python3
"""Measure solver quality on a stratified, unseen FloorSet training holdout.

The contest validation set has one public instance per block count and is easy
to overfit.  This gate samples different training instances for every size in
the requested range, scores both the solver and the supplied layout with the
official evaluator, and reports MIB-corrupted samples separately.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SOLUTION_DIR = ROOT / "contest_solution"
OFFICIAL_ROOT = ROOT / "external" / "FloorSet"
sys.path.insert(0, str(OFFICIAL_ROOT))
sys.path.insert(0, str(SOLUTION_DIR))

from iccad2026_evaluate import compute_total_score, evaluate_solution  # noqa: E402
from lite_dataset import FloorplanDatasetLite  # noqa: E402


def _load_optimizer(solver_dir: Path):
    solver_dir = solver_dir.resolve()
    sys.path.insert(0, str(solver_dir))
    optimizer_path = solver_dir / "my_optimizer.py"
    spec = importlib.util.spec_from_file_location("holdout_optimizer", optimizer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load optimizer from {optimizer_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MyOptimizer(verbose=False)


def _positions_from_training_label(fp_sol: torch.Tensor, n: int):
    """Convert training [w,h,x,y] rows to evaluator [x,y,w,h] tuples."""
    return [
        (
            float(fp_sol[i, 2]),
            float(fp_sol[i, 3]),
            float(fp_sol[i, 0]),
            float(fp_sol[i, 1]),
        )
        for i in range(n)
    ]


def _optimizer_targets(constraints, golden_positions, n):
    out = torch.full((n, 4), -1.0)
    ncols = constraints.shape[1] if constraints is not None and constraints.dim() > 1 else 0
    for i in range(n):
        is_fixed = ncols > 0 and constraints[i, 0] != 0
        is_preplaced = ncols > 1 and constraints[i, 1] != 0
        x, y, w, h = golden_positions[i]
        if is_preplaced:
            out[i] = torch.tensor((x, y, w, h))
        elif is_fixed:
            out[i, 2] = w
            out[i, 3] = h
    return out


def _baseline(metrics):
    return {
        "area_baseline": float(metrics[0]),
        "hpwl_baseline": float(metrics[-2]) + float(metrics[-1]),
    }


def _golden_mib_violation_count(sample, n):
    constraints = sample["input"][4]
    fp_sol = sample["label"][1]
    if constraints is None or constraints.dim() < 2 or constraints.shape[1] <= 2:
        return 0
    mib = constraints[:n, 2]
    max_group = int(mib.max().item()) if mib.numel() else 0
    violations = 0
    for group_id in range(1, max_group + 1):
        indices = torch.where(mib == group_id)[0].tolist()
        shapes = {
            (round(float(fp_sol[i, 0]), 4), round(float(fp_sol[i, 1]), 4))
            for i in indices
        }
        violations += max(0, len(shapes) - 1)
    return violations


def _summary(rows):
    if not rows:
        return {"cases": 0}
    costs = [r["cost"] for r in rows]
    counts = [r["block_count"] for r in rows]
    max_n = max(counts)
    weights = [math.exp((n - max_n) / 12.0) for n in counts]
    z = sum(weights)

    def weighted(key):
        return sum(w * float(r[key]) for w, r in zip(weights, rows)) / z

    runtimes = [r["runtime_seconds"] for r in rows]
    return {
        "cases": len(rows),
        "feasible": sum(bool(r["is_feasible"]) for r in rows),
        "total_score": compute_total_score(costs, counts),
        "weighted_hpwl_gap_clamped": weighted("hpwl_gap_clamped"),
        "weighted_area_gap_clamped": weighted("area_gap_clamped"),
        "weighted_violations_relative": weighted("violations_relative"),
        "runtime_mean": statistics.fmean(runtimes),
        "runtime_p95": sorted(runtimes)[max(0, math.ceil(0.95 * len(runtimes)) - 1)],
        "runtime_max": max(runtimes),
    }


def _portable_path(path: Path | None):
    """Keep result metadata reproducible when a path lives in this checkout."""
    if path is None:
        return None
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _metric_row(sample_index, n, metrics, runtime, golden_mib_violations):
    return {
        "sample_index": sample_index,
        "block_count": n,
        "is_feasible": bool(metrics.is_feasible),
        "cost": float(metrics.cost),
        "hpwl_gap": float(metrics.hpwl_gap),
        "hpwl_gap_clamped": max(0.0, float(metrics.hpwl_gap)),
        "area_gap": float(metrics.area_gap),
        "area_gap_clamped": max(0.0, float(metrics.area_gap)),
        "violations_relative": float(metrics.violations_relative),
        "boundary_violations": int(metrics.boundary_violations),
        "grouping_violations": int(metrics.grouping_violations),
        "mib_violations": int(metrics.mib_violations),
        "golden_mib_violations": int(golden_mib_violations),
        "runtime_seconds": float(runtime),
    }


def collect_stratified(
    dataset, min_blocks, max_blocks, per_size, seed, max_files,
    require_golden_mib_clean=False,
):
    # glob order is filesystem-dependent; make sampling stable across machines.
    dataset.all_files = sorted(dataset.all_files)
    file_ids = list(range(len(dataset.all_files)))
    random.Random(seed).shuffle(file_ids)
    buckets = {n: [] for n in range(min_blocks, max_blocks + 1)}
    scanned_files = 0
    for file_idx in file_ids[:max_files]:
        scanned_files += 1
        base = file_idx * dataset.layouts_per_file
        for offset in range(dataset.layouts_per_file):
            sample_index = base + offset
            sample = dataset[sample_index]
            area_target = sample["input"][0]
            n = int((area_target != -1).sum().item())
            if n not in buckets or len(buckets[n]) >= per_size:
                continue
            if require_golden_mib_clean and _golden_mib_violation_count(sample, n) != 0:
                continue
            buckets[n].append((sample_index, sample))
        if all(len(rows) >= per_size for rows in buckets.values()):
            break
    missing = {n: per_size - len(rows) for n, rows in buckets.items() if len(rows) < per_size}
    if missing:
        raise RuntimeError(f"stratified sample incomplete after {scanned_files} files: {missing}")
    selected = [item for n in sorted(buckets) for item in buckets[n]]
    return selected, scanned_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "external" / "FloorSet")
    parser.add_argument("--solver-dir", type=Path, default=SOLUTION_DIR)
    parser.add_argument("--min-blocks", type=int, default=100)
    parser.add_argument("--max-blocks", type=int, default=120)
    parser.add_argument("--per-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--max-files", type=int, default=1000)
    parser.add_argument("--require-golden-mib-clean", action="store_true")
    parser.add_argument(
        "--indices-from", type=Path,
        help="reuse sample_index values from a previous holdout result",
    )
    parser.add_argument(
        "--oracle-baseline-selector", action="store_true",
        help="diagnostic only: expose the golden HPWL/area baseline to candidate selection",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "training_holdout.json")
    args = parser.parse_args()

    dataset = FloorplanDatasetLite(str(args.data_root))
    dataset.all_files = sorted(dataset.all_files)
    if args.indices_from:
        prior = json.loads(args.indices_from.read_text())
        indices = [int(row["sample_index"]) for row in prior["cases"]]
        selected = [(index, dataset[index]) for index in indices]
        scanned_files = 0
    else:
        selected, scanned_files = collect_stratified(
            dataset, args.min_blocks, args.max_blocks, args.per_size, args.seed,
            args.max_files, args.require_golden_mib_clean,
        )
    optimizer = _load_optimizer(args.solver_dir)
    solver_rows = []
    golden_rows = []

    for ordinal, (sample_index, sample) in enumerate(selected, 1):
        area_target, b2b, p2b, pins, constraints = sample["input"]
        _, fp_sol, stored_metrics = sample["label"]
        n = int((area_target != -1).sum().item())
        golden_positions = _positions_from_training_label(fp_sol, n)
        opt_targets = _optimizer_targets(constraints, golden_positions, n)
        baseline = _baseline(stored_metrics)

        golden_metrics = evaluate_solution(
            {"positions": golden_positions, "runtime": 1.0},
            baseline,
            constraints,
            b2b,
            p2b,
            pins,
            area_target,
            golden_positions,
            median_runtime=1.0,
        )
        if args.oracle_baseline_selector:
            optimizer._baselines_by_n = {
                n: (baseline["hpwl_baseline"], baseline["area_baseline"])
            }
        t0 = time.perf_counter()
        positions = optimizer.solve(
            n, area_target, b2b, p2b, pins, constraints, opt_targets
        )
        runtime = time.perf_counter() - t0
        solver_metrics = evaluate_solution(
            {"positions": positions, "runtime": 1.0},
            baseline,
            constraints,
            b2b,
            p2b,
            pins,
            area_target,
            golden_positions,
            median_runtime=1.0,
        )
        solver_rows.append(
            _metric_row(sample_index, n, solver_metrics, runtime, golden_metrics.mib_violations)
        )
        golden_rows.append(
            _metric_row(sample_index, n, golden_metrics, 0.0, golden_metrics.mib_violations)
        )
        if ordinal % 10 == 0 or ordinal == len(selected):
            print(f"scored {ordinal}/{len(selected)}", flush=True)

    clean_solver = [r for r in solver_rows if r["golden_mib_violations"] == 0]
    clean_golden = [r for r in golden_rows if r["golden_mib_violations"] == 0]
    by_size = defaultdict(list)
    for row in solver_rows:
        by_size[row["block_count"]].append(row)
    result = {
        "config": {
            "min_blocks": args.min_blocks,
            "max_blocks": args.max_blocks,
            "per_size": args.per_size,
            "seed": args.seed,
            "scanned_files": scanned_files,
            "require_golden_mib_clean": args.require_golden_mib_clean,
            "solver_dir": _portable_path(args.solver_dir),
            "indices_from": _portable_path(args.indices_from),
            "oracle_baseline_selector": args.oracle_baseline_selector,
        },
        "solver_all": _summary(solver_rows),
        "golden_all": _summary(golden_rows),
        "solver_golden_mib_clean": _summary(clean_solver),
        "golden_mib_clean": _summary(clean_golden),
        "golden_mib_corrupt_cases": len(solver_rows) - len(clean_solver),
        "solver_by_size": {str(n): _summary(rows) for n, rows in sorted(by_size.items())},
        "cases": solver_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("cases", "solver_by_size")}, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
