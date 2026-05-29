#!/usr/bin/env python3
"""P3.1: Mine golden solutions for shape/utilization/structure priors.

Loads golden layouts from the validation set and measures:
- Aspect ratio distributions of soft blocks
- Area utilization (golden bbox vs Σarea)
- Perimeter-ring structure
- How clusters are shaped/placed
- Typical wirelength per net

Usage:
    python3 scripts/mine_golden.py
"""
import sys
sys.path.insert(0, '/home/ubuntu/EDA/external/FloorSet/iccad2026contest')

import math
import json
from collections import defaultdict
from iccad2026_evaluate import ContestEvaluator


def main():
    ev = ContestEvaluator(data_path="../", verbose=False)
    ev._load_dataset()

    aspect_ratios = []
    utilizations = []
    cluster_shapes = defaultdict(list)
    boundary_counts = []
    preplaced_counts = []
    cluster_counts = []
    mib_counts = []

    for idx in range(100):
        sample = ev.dataset[idx]
        inputs, labels = sample['input'], sample['label']
        area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
        block_count = int((area_target != -1).sum().item())

        baseline, target_pos_list = ev._extract_baseline(
            idx, labels, b2b_conn, p2b_conn, pins_pos, block_count)

        # Extract golden positions
        positions = []
        for i in range(block_count):
            block = labels[0][i]
            valid = block[block[:, 0] != -1]
            if len(valid) > 0:
                x_min, y_min = valid.min(dim=0).values
                x_max, y_max = valid.max(dim=0).values
                positions.append((float(x_min), float(y_min),
                                float(x_max - x_min), float(y_max - y_min)))
            else:
                positions.append((0, 0, 1, 1))

        # Utilization
        total_block_area = sum(float(area_target[i]) for i in range(block_count) if area_target[i] > 0)
        x_min = min(p[0] for p in positions)
        y_min = min(p[1] for p in positions)
        x_max = max(p[0] + p[2] for p in positions)
        y_max = max(p[1] + p[3] for p in positions)
        bbox_area = (x_max - x_min) * (y_max - y_min)
        util = total_block_area / max(bbox_area, 1e-6)
        utilizations.append(util)

        # Aspect ratios of soft blocks
        nc = constraints.shape[1] if constraints.dim() > 1 else 0
        n_boundary = 0
        n_preplaced = 0
        n_cluster = 0
        n_mib = 0
        for i in range(block_count):
            is_fixed = nc > 0 and constraints[i, 0] != 0
            is_preplaced = nc > 1 and constraints[i, 1] != 0
            is_mib = nc > 2 and constraints[i, 2] != 0
            is_cluster = nc > 3 and constraints[i, 3] != 0
            boundary_code = int(constraints[i, 4].item()) if nc > 4 else 0

            if is_preplaced:
                n_preplaced += 1
            if is_cluster:
                n_cluster += 1
            if is_mib:
                n_mib += 1
            if boundary_code != 0:
                n_boundary += 1

            if not is_fixed and not is_preplaced:
                w, h = positions[i][2], positions[i][3]
                ar = max(w, h) / max(min(w, h), 1e-9)
                aspect_ratios.append(ar)

                if is_cluster:
                    gid = int(constraints[i, 3].item())
                    cluster_shapes[gid].append((w, h, float(area_target[i])))

        boundary_counts.append(n_boundary)
        preplaced_counts.append(n_preplaced)
        cluster_counts.append(n_cluster // max(1, n_cluster // 3))  # approximate group count
        mib_counts.append(n_mib)

    # Report
    print("=== Golden Mining Results ===\n")

    print(f"Utilization (Σblock_area / bbox_area):")
    print(f"  mean: {sum(utilizations)/len(utilizations):.3f}")
    print(f"  min:  {min(utilizations):.3f}")
    print(f"  max:  {max(utilizations):.3f}")
    print(f"  median: {sorted(utilizations)[len(utilizations)//2]:.3f}")

    print(f"\nAspect ratios of soft blocks:")
    aspect_ratios.sort()
    print(f"  count: {len(aspect_ratios)}")
    print(f"  mean: {sum(aspect_ratios)/len(aspect_ratios):.2f}")
    print(f"  median: {aspect_ratios[len(aspect_ratios)//2]:.2f}")
    print(f"  p90: {aspect_ratios[int(len(aspect_ratios)*0.9)]:.2f}")
    print(f"  p95: {aspect_ratios[int(len(aspect_ratios)*0.95)]:.2f}")
    print(f"  max: {aspect_ratios[-1]:.2f}")

    print(f"\nConstraint counts per case:")
    print(f"  boundary: mean={sum(boundary_counts)/len(boundary_counts):.1f} min={min(boundary_counts)} max={max(boundary_counts)}")
    print(f"  preplaced: mean={sum(preplaced_counts)/len(preplaced_counts):.1f} min={min(preplaced_counts)} max={max(preplaced_counts)}")
    print(f"  cluster groups: mean={sum(cluster_counts)/len(cluster_counts):.1f}")
    print(f"  MIB blocks: mean={sum(mib_counts)/len(mib_counts):.1f}")

    # Per-cluster shape analysis
    print(f"\nCluster shape analysis ({len(cluster_shapes)} groups):")
    for gid in sorted(cluster_shapes.keys())[:10]:
        shapes = cluster_shapes[gid]
        ws = [s[0] for s in shapes]
        hs = [s[1] for s in shapes]
        areas = [s[2] for s in shapes]
        print(f"  group {gid}: {len(shapes)} members, w=[{min(ws):.1f},{max(ws):.1f}], h=[{min(hs):.1f},{max(hs):.1f}], area=[{min(areas):.1f},{max(areas):.1f}]")


if __name__ == '__main__':
    main()
