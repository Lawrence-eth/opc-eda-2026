"""Torch-free stand-in for the pieces of iccad2026_evaluate.py that
my_optimizer.py imports.

Shipped inside the executable as `iccad2026_evaluate.py` so the unmodified
my_optimizer.py imports resolve without pulling in the full official
evaluator (torch / shapely / matplotlib). The three metric helpers are copied
verbatim from the official evaluator (they are already pure Python over
sequences); FloorplanOptimizer is the trivial base class.
"""

from typing import Dict, List, Optional, Tuple


def calculate_hpwl_b2b(positions, b2b_connectivity) -> float:
    """Calculate block-to-block HPWL. (Verbatim from the official evaluator.)"""
    if b2b_connectivity is None or len(b2b_connectivity) == 0:
        return 0.0

    total_wl = 0.0
    for edge in b2b_connectivity:
        if edge[0] == -1:
            continue
        i, j, weight = int(edge[0]), int(edge[1]), float(edge[2])
        if i < len(positions) and j < len(positions):
            x1 = positions[i][0] + positions[i][2] / 2
            y1 = positions[i][1] + positions[i][3] / 2
            x2 = positions[j][0] + positions[j][2] / 2
            y2 = positions[j][1] + positions[j][3] / 2
            total_wl += weight * (abs(x2 - x1) + abs(y2 - y1))
    return total_wl


def calculate_hpwl_p2b(positions, p2b_connectivity, pins_pos) -> float:
    """Calculate pin-to-block HPWL. (Verbatim from the official evaluator.)"""
    if p2b_connectivity is None or len(p2b_connectivity) == 0:
        return 0.0

    total_wl = 0.0
    for edge in p2b_connectivity:
        if edge[0] == -1:
            continue
        pin_idx, block_idx, weight = int(edge[0]), int(edge[1]), float(edge[2])
        if block_idx < len(positions) and pin_idx < len(pins_pos):
            px, py = float(pins_pos[pin_idx][0]), float(pins_pos[pin_idx][1])
            bx = positions[block_idx][0] + positions[block_idx][2] / 2
            by = positions[block_idx][1] + positions[block_idx][3] / 2
            total_wl += weight * (abs(px - bx) + abs(py - by))
    return total_wl


def calculate_bbox_area(positions) -> float:
    """Calculate bounding box area of all blocks. (Verbatim from the official evaluator.)"""
    if not positions:
        return 0.0

    x_min = min(p[0] for p in positions)
    y_min = min(p[1] for p in positions)
    x_max = max(p[0] + p[2] for p in positions)
    y_max = max(p[1] + p[3] for p in positions)

    return (x_max - x_min) * (y_max - y_min)


class FloorplanOptimizer:
    """Base class for floorplanning optimizers (interface-compatible subset)."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def solve(self, block_count, area_targets, b2b_connectivity,
              p2b_connectivity, pins_pos, constraints,
              target_positions=None):
        raise NotImplementedError("Subclasses must implement solve()")
