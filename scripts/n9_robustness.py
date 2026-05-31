"""N9 Robustness Hardening Test Harness.

Runs v9's solve() on FloorSet training instances and asserts feasibility.
Logs every failure for fixing.

Usage:
    python3 scripts/n9_robustness.py [--num_samples N] [--seed S]
"""

import sys
import os
import time
import math
import json
import random
import argparse
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'external' / 'FloorSet'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'external' / 'FloorSet' / 'iccad2026contest'))
os.chdir(str(Path(__file__).parent.parent))

import torch
import numpy as np
from lite_dataset import FloorplanDatasetLite, floorplan_collate
from torch.utils.data import DataLoader, Subset


def check_feasibility(positions, area_targets, constraints, target_positions, block_count):
    """Check if a layout is feasible.

    Returns:
        feasible: bool
        violations: list of violation descriptions
    """
    violations = []
    n = block_count

    if positions is None or len(positions) == 0:
        return False, ["Empty positions"]

    if len(positions) < n:
        return False, [f"Positions too short: {len(positions)} < {n}"]

    # Check for None positions
    for i in range(n):
        if positions[i] is None:
            violations.append(f"Block {i}: None position")
            continue

    if violations:
        return False, violations

    # Check for overlaps
    for i in range(n):
        xi, yi, wi, hi = positions[i]
        for j in range(i + 1, n):
            xj, yj, wj, hj = positions[j]
            ox = min(xi + wi, xj + wj) - max(xi, xj)
            oy = min(yi + hi, yj + hj) - max(yi, yj)
            if ox > 1e-6 and oy > 1e-6:
                violations.append(f"Overlap: blocks {i} and {j}")
                if len(violations) > 10:
                    return False, violations

    # Check soft-block area tolerance (±1%)
    ncols = constraints.shape[1] if constraints is not None and constraints.dim() > 1 else 0
    for i in range(n):
        # Skip fixed and preplaced blocks
        if ncols > 0 and constraints[i, 0] != 0:
            continue  # fixed
        if ncols > 1 and constraints[i, 1] != 0:
            continue  # preplaced

        target_area = float(area_targets[i]) if i < len(area_targets) and area_targets[i] > 0 else 0
        if target_area <= 0:
            continue

        xi, yi, wi, hi = positions[i]
        actual_area = wi * hi
        if abs(actual_area - target_area) / target_area > 0.01:
            violations.append(f"Area violation: block {i} actual={actual_area:.2f} target={target_area:.2f} ratio={actual_area/target_area:.4f}")
            if len(violations) > 10:
                return False, violations

    # Check fixed/preplaced dimensions
    if target_positions is not None and ncols > 0:
        for i in range(n):
            is_fixed = ncols > 0 and constraints[i, 0] != 0
            is_preplaced = ncols > 1 and constraints[i, 1] != 0

            if is_preplaced:
                # Check exact (x, y, w, h)
                tx, ty, tw, th = float(target_positions[i][0]), float(target_positions[i][1]), float(target_positions[i][2]), float(target_positions[i][3])
                xi, yi, wi, hi = positions[i]
                if abs(xi - tx) > 1e-4 or abs(yi - ty) > 1e-4 or abs(wi - tw) > 1e-4 or abs(hi - th) > 1e-4:
                    violations.append(f"Preplaced violation: block {i} pos=({xi:.2f},{yi:.2f},{wi:.2f},{hi:.2f}) expected=({tx:.2f},{ty:.2f},{tw:.2f},{th:.2f})")
                    if len(violations) > 10:
                        return False, violations
            elif is_fixed:
                # Check exact (w, h)
                tw, th = float(target_positions[i][2]), float(target_positions[i][3])
                if tw > 0 and th > 0:
                    xi, yi, wi, hi = positions[i]
                    if abs(wi - tw) > 1e-4 or abs(hi - th) > 1e-4:
                        violations.append(f"Fixed dim violation: block {i} dims=({wi:.2f},{hi:.2f}) expected=({tw:.2f},{th:.2f})")
                        if len(violations) > 10:
                            return False, violations

    return len(violations) == 0, violations


def run_robustness_test(num_samples=1000, seed=42, timeout_per_case=5.0):
    """Run robustness test on training instances.

    Args:
        num_samples: number of training instances to test
        seed: random seed for sampling
        timeout_per_case: max runtime per case in seconds

    Returns:
        results: dict with test results
    """
    print(f"Loading training dataset...")
    dataset = FloorplanDatasetLite('external/FloorSet')
    print(f"Total instances: {len(dataset)}")

    # Sample instances
    random.seed(seed)
    if num_samples < len(dataset):
        indices = random.sample(range(len(dataset)), num_samples)
    else:
        indices = list(range(len(dataset)))

    # Load optimizer
    print("Loading optimizer...")
    from iccad2026_evaluate import FloorplanOptimizer

    # Import the optimizer class
    sys.path.insert(0, 'contest_solution')
    from my_optimizer import MyOptimizer
    optimizer = MyOptimizer(verbose=False)

    # Run tests
    print(f"\nRunning {num_samples} instances...")
    results = {
        'total': len(indices),
        'passed': 0,
        'failed': 0,
        'errors': 0,
        'failures': [],
        'runtimes': [],
    }

    for idx, sample_idx in enumerate(indices):
        if idx % 100 == 0:
            print(f"  Progress: {idx}/{len(indices)} ({results['passed']} passed, {results['failed']} failed, {results['errors']} errors)")

        try:
            sample = dataset[sample_idx]
            inputs, labels = sample['input'], sample['label']
            area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
            tree_sol, fp_sol, metrics = labels

            n = int((area_target != -1).sum().item())

            # Build target_positions (x, y, w, h format)
            # fp_sol is (w, h, x, y) — need to swap to (x, y, w, h)
            target_positions = torch.full((n, 4), -1.0)
            if constraints is not None and constraints.dim() > 1:
                ncols = constraints.shape[1]
                for i in range(n):
                    is_fixed = ncols > 0 and constraints[i, 0] != 0
                    is_preplaced = ncols > 1 and constraints[i, 1] != 0
                    if is_preplaced:
                        # fp_sol[i] = (w, h, x, y) -> target_positions = (x, y, w, h)
                        target_positions[i, 0] = fp_sol[i, 2]  # x
                        target_positions[i, 1] = fp_sol[i, 3]  # y
                        target_positions[i, 2] = fp_sol[i, 0]  # w
                        target_positions[i, 3] = fp_sol[i, 1]  # h
                    elif is_fixed:
                        # Fixed blocks: only w, h matter
                        target_positions[i, 2] = fp_sol[i, 0]  # w
                        target_positions[i, 3] = fp_sol[i, 1]  # h

            # Run solve()
            t0 = time.time()
            positions = optimizer.solve(
                n,
                area_target[:n],
                b2b_conn,
                p2b_conn,
                pins_pos,
                constraints[:n] if constraints is not None else None,
                target_positions
            )
            runtime = time.time() - t0
            results['runtimes'].append(runtime)

            # Check feasibility
            feasible, violations = check_feasibility(
                positions, area_target[:n], constraints[:n] if constraints is not None else None,
                target_positions, n
            )

            if feasible:
                results['passed'] += 1
            else:
                results['failed'] += 1
                results['failures'].append({
                    'sample_idx': sample_idx,
                    'n_blocks': n,
                    'violations': violations[:5],  # first 5 violations
                    'runtime': runtime,
                })
                if len(results['failures']) <= 20:  # print first 20 failures
                    print(f"  FAIL: sample={sample_idx}, n={n}, violations={violations[:3]}")

        except Exception as e:
            results['errors'] += 1
            results['failures'].append({
                'sample_idx': sample_idx,
                'n_blocks': n if 'n' in dir() else -1,
                'violations': [f"Exception: {str(e)[:200]}"],
                'runtime': 0,
            })
            if results['errors'] <= 10:
                print(f"  ERROR: sample={sample_idx}, n={n if 'n' in dir() else '?'}, error={str(e)[:100]}")

    # Summary
    print(f"\n{'='*60}")
    print(f"N9 ROBUSTNESS TEST RESULTS")
    print(f"{'='*60}")
    print(f"Total tested: {results['total']}")
    print(f"Passed: {results['passed']} ({100*results['passed']/results['total']:.1f}%)")
    print(f"Failed: {results['failed']} ({100*results['failed']/results['total']:.1f}%)")
    print(f"Errors: {results['errors']} ({100*results['errors']/results['total']:.1f}%)")

    if results['runtimes']:
        avg_rt = sum(results['runtimes']) / len(results['runtimes'])
        max_rt = max(results['runtimes'])
        print(f"Runtime: avg={avg_rt:.3f}s, max={max_rt:.3f}s")

    if results['failures']:
        print(f"\nFirst 20 failures:")
        for f in results['failures'][:20]:
            print(f"  sample={f['sample_idx']}, n={f['n_blocks']}, violations={f['violations'][:3]}")

    # Save results
    output_path = 'results/n9_robustness.json'
    with open(output_path, 'w') as fp:
        json.dump(results, fp, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--timeout', type=float, default=5.0)
    args = parser.parse_args()

    run_robustness_test(args.num_samples, args.seed, args.timeout)
