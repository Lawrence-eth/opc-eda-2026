"""
Sequence-pair SA floorplanner — M1/M2 implementation.

Sequence-pair (SP): two permutations Γ+ and Γ- of block indices.
Packing rule: for blocks i, j:
  - i before j in BOTH Γ+ and Γ- → i is LEFT of j
  - i before j in Γ+ but after j in Γ- → i is BELOW j
  - i after j in Γ+ but before j in Γ- → i is ABOVE j
  - i after j in BOTH → i is RIGHT of j

Positions derived via longest-path on horizontal/vertical constraint graphs.
Every SP state is a valid non-overlapping packing.

SA perturbations: swap two random elements in Γ+ or Γ-.
"""

import math
import random
import time
from typing import Dict, List, Tuple


def sp_pack(gamma_plus, gamma_minus, widths, heights):
    """Pack blocks from a sequence-pair via longest-path.

    Args:
        gamma_plus: permutation (list of block indices)
        gamma_minus: permutation (list of block indices)
        widths: dict {block_idx: width}
        heights: dict {block_idx: height}

    Returns:
        positions: dict {block_idx: (x, y, w, h)}
        bbox: (x_max, y_max) — bounding box extent
    """
    n = len(gamma_plus)

    # Position maps: pos_plus[i] = position of block i in Γ+
    pos_plus = [0] * n
    pos_minus = [0] * n
    for idx, val in enumerate(gamma_plus):
        pos_plus[val] = idx
    for idx, val in enumerate(gamma_minus):
        pos_minus[val] = idx

    # Longest-path for x coordinates (horizontal constraint graph)
    # x[j] = max over all i LEFT of j of (x[i] + w[i])
    # i is LEFT of j iff pos_plus[i] < pos_plus[j] AND pos_minus[i] < pos_minus[j]
    x = [0.0] * n
    # Process in order of pos_plus (left to right in Γ+)
    order_plus = sorted(range(n), key=lambda k: pos_plus[k])
    for j_idx in range(n):
        j = order_plus[j_idx]
        x_max = 0.0
        for i_idx in range(j_idx):
            i = order_plus[i_idx]
            if pos_minus[i] < pos_minus[j]:  # i is LEFT of j
                if x[i] + widths[i] > x_max:
                    x_max = x[i] + widths[i]
        x[j] = x_max

    # Longest-path for y coordinates (vertical constraint graph)
    # y[j] = max over all i BELOW j of (y[i] + h[i])
    # i is BELOW j iff pos_plus[i] < pos_plus[j] AND pos_minus[i] > pos_minus[j]
    y = [0.0] * n
    order_minus = sorted(range(n), key=lambda k: pos_minus[k])
    for j_idx in range(n):
        j = order_minus[j_idx]
        y_max = 0.0
        for i_idx in range(j_idx):
            i = order_minus[i_idx]
            if pos_plus[i] > pos_plus[j]:  # i is BELOW j (in Γ+, i is right of j; in Γ-, i is before j)
                if y[i] + heights[i] > y_max:
                    y_max = y[i] + heights[i]
        y[j] = y_max

    positions = {}
    for i in range(n):
        positions[i] = (x[i], y[i], widths[i], heights[i])

    x_max = max(x[i] + widths[i] for i in range(n)) if n > 0 else 0
    y_max = max(y[i] + heights[i] for i in range(n)) if n > 0 else 0
    return positions, (x_max, y_max)


def sp_sa_movable_only(block_count, area_targets, b2b_edges, p2b_edges, pins_pos,
                        dims, max_time=30.0, seed=42):
    """M2: SA over sequence-pair minimizing bbox area + HPWL on movable-only blocks.

    Args:
        block_count: number of blocks (only movable ones)
        area_targets: list/tensor of area targets per block
        b2b_edges: list of (i, j, weight) b2b edges
        p2b_edges: list of (pin_idx, block_idx, weight) p2b edges
        pins_pos: list of (x, y) pin positions
        dims: list of (w, h) per block (pre-computed, near-square)
        max_time: time budget in seconds
        seed: random seed

    Returns:
        positions: dict {block_idx: (x, y, w, h)}
        bbox: (x_max, y_max)
        cost: float (bbox_area + lambda * HPWL)
        utilization: float
    """
    random.seed(seed)
    n = block_count

    widths = {i: dims[i][0] for i in range(n)}
    heights = {i: dims[i][1] for i in range(n)}
    total_area = sum(widths[i] * heights[i] for i in range(n))

    # Build adjacency lists for HPWL
    b_adj = {i: [] for i in range(n)}
    for a, b, w in b2b_edges:
        if 0 <= a < n and 0 <= b < n:
            b_adj[a].append((b, w))
            b_adj[b].append((a, w))
    p_adj = {i: [] for i in range(n)}
    for pin_idx, b_idx, w in p2b_edges:
        if 0 <= b_idx < n and 0 <= pin_idx < len(pins_pos):
            px, py = pins_pos[pin_idx]
            if px != -1.0 and py != -1.0:
                p_adj[b_idx].append((px, py, w))

    def compute_hpwl(positions):
        """Compute total HPWL (b2b + p2b)."""
        total = 0.0
        seen = set()
        for i in range(n):
            cx_i = positions[i][0] + positions[i][2] * 0.5
            cy_i = positions[i][1] + positions[i][3] * 0.5
            for j, w in b_adj[i]:
                if j > i:
                    cx_j = positions[j][0] + positions[j][2] * 0.5
                    cy_j = positions[j][1] + positions[j][3] * 0.5
                    total += w * (abs(cx_i - cx_j) + abs(cy_i - cy_j))
            for px, py, w in p_adj[i]:
                total += w * (abs(cx_i - px) + abs(cy_i - py))
        return total

    # Initialize random SP
    gamma_plus = list(range(n))
    gamma_minus = list(range(n))
    random.shuffle(gamma_plus)
    random.shuffle(gamma_minus)

    # Pack and compute initial cost
    positions, (xmax, ymax) = sp_pack(gamma_plus, gamma_minus, widths, heights)
    hpwl = compute_hpwl(positions)
    bbox_area = xmax * ymax
    LAMBDA = 0.01  # HPWL weight relative to area
    current_cost = bbox_area + LAMBDA * hpwl
    best_cost = current_cost
    best_positions = dict(positions)
    best_gamma_plus = list(gamma_plus)
    best_gamma_minus = list(gamma_minus)

    # SA loop
    T0 = 100.0
    T_min = 0.01
    cooling = 0.9995
    T = T0
    moves = 0
    accepts = 0
    start = time.time()

    while T > T_min and time.time() - start < max_time:
        # Perturbation: swap two random elements in Γ+ or Γ-
        if random.random() < 0.5:
            arr = gamma_plus
        else:
            arr = gamma_minus
        i_idx = random.randint(0, n - 1)
        j_idx = random.randint(0, n - 1)
        while j_idx == i_idx:
            j_idx = random.randint(0, n - 1)
        arr[i_idx], arr[j_idx] = arr[j_idx], arr[i_idx]

        # Repack
        positions, (xmax, ymax) = sp_pack(gamma_plus, gamma_minus, widths, heights)
        hpwl = compute_hpwl(positions)
        bbox_area = xmax * ymax
        new_cost = bbox_area + LAMBDA * hpwl

        # Metropolis acceptance
        delta = new_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
            current_cost = new_cost
            accepts += 1
            if current_cost < best_cost:
                best_cost = current_cost
                best_positions = dict(positions)
                best_gamma_plus = list(gamma_plus)
                best_gamma_minus = list(gamma_minus)
        else:
            # Revert
            arr[i_idx], arr[j_idx] = arr[j_idx], arr[i_idx]

        T *= cooling
        moves += 1

    elapsed = time.time() - start
    best_bbox = (max(best_positions[i][0] + best_positions[i][2] for i in range(n)),
                 max(best_positions[i][1] + best_positions[i][3] for i in range(n)))
    best_hpwl = compute_hpwl(best_positions)
    best_area = best_bbox[0] * best_bbox[1]
    util = total_area / max(best_area, 1e-6)

    print(f"  SP-SA: {moves} moves, {accepts} accepts in {elapsed:.1f}s")
    print(f"  bbox={best_bbox[0]:.1f}x{best_bbox[1]:.1f} area={best_area:.0f}")
    print(f"  HPWL={best_hpwl:.1f} cost={best_cost:.0f}")
    print(f"  utilization={util:.3f}")

    return best_positions, best_bbox, best_cost, util


def sp_sa_with_obstacles(movable_count, preplaced_rects, all_areas, all_dims,
                          b2b_edges, p2b_edges, pins_pos,
                          max_time=30.0, seed=42, penalty_weight=1e6):
    """M3-ready: SA over SP treating preplaced as fixed obstacles with penalty.

    Args:
        movable_count: number of movable blocks (indices 0..movable_count-1)
        preplaced_rects: list of (x, y, w, h) for preplaced blocks
        all_areas: area targets for ALL blocks (movable + preplaced)
        all_dims: dims for ALL blocks
        b2b_edges, p2b_edges, pins_pos: full connectivity
        max_time: time budget
        seed: random seed
        penalty_weight: large penalty for obstacle overlaps

    Returns:
        Same as sp_sa_movable_only
    """
    random.seed(seed)
    n = movable_count  # only movable blocks are in the SP

    widths = {i: all_dims[i][0] for i in range(n)}
    heights = {i: all_dims[i][1] for i in range(n)}
    total_area = sum(widths[i] * heights[i] for i in range(n))

    # Preplaced as obstacle rects (absolute coordinates)
    obstacles = [(r[0], r[1], r[0]+r[2], r[1]+r[3]) for r in preplaced_rects]

    # Build adjacency (only movable-movable and movable-preplaced edges matter)
    b_adj = {i: [] for i in range(n)}
    for a, b, w in b2b_edges:
        if 0 <= a < n and 0 <= b < n:
            b_adj[a].append((b, w))
            b_adj[b].append((a, w))
    p_adj = {i: [] for i in range(n)}
    for pin_idx, b_idx, w in p2b_edges:
        if 0 <= b_idx < n and 0 <= pin_idx < len(pins_pos):
            px, py = pins_pos[pin_idx]
            if px != -1.0 and py != -1.0:
                p_adj[b_idx].append((px, py, w))

    def compute_cost(positions):
        """Cost = bbox_area + λ·HPWL + P·obstacle_overlaps"""
        hpwl = 0.0
        for i in range(n):
            cx_i = positions[i][0] + positions[i][2] * 0.5
            cy_i = positions[i][1] + positions[i][3] * 0.5
            for j, w in b_adj[i]:
                if j > i:
                    cx_j = positions[j][0] + positions[j][2] * 0.5
                    cy_j = positions[j][1] + positions[j][3] * 0.5
                    hpwl += w * (abs(cx_i - cx_j) + abs(cy_i - cy_j))
            for px, py, w in p_adj[i]:
                hpwl += w * (abs(cx_i - px) + abs(cy_i - py))

        xmax = max(positions[i][0] + widths[i] for i in range(n))
        ymax = max(positions[i][1] + heights[i] for i in range(n))
        bbox = xmax * ymax

        # Obstacle overlap penalty
        pen = 0.0
        for i in range(n):
            ix, iy, iw, ih = positions[i]
            for (ox1, oy1, ox2, oy2) in obstacles:
                ox = min(ix+iw, ox2) - max(ix, ox1)
                oy = min(iy+ih, oy2) - max(iy, oy1)
                if ox > 1e-6 and oy > 1e-6:
                    pen += ox * oy  # overlap area

        LAMBDA = 0.01
        return bbox + LAMBDA * hpwl + penalty_weight * pen

    # Init random SP
    gamma_plus = list(range(n))
    gamma_minus = list(range(n))
    random.shuffle(gamma_plus)
    random.shuffle(gamma_minus)

    positions, (xmax, ymax) = sp_pack(gamma_plus, gamma_minus, widths, heights)
    current_cost = compute_cost(positions)
    best_cost = current_cost
    best_positions = dict(positions)

    # SA
    T0 = 100.0; T_min = 0.01; cooling = 0.9995; T = T0
    moves = 0; start = time.time()

    while T > T_min and time.time() - start < max_time:
        arr = gamma_plus if random.random() < 0.5 else gamma_minus
        i_idx = random.randint(0, n-1); j_idx = random.randint(0, n-1)
        while j_idx == i_idx: j_idx = random.randint(0, n-1)
        arr[i_idx], arr[j_idx] = arr[j_idx], arr[i_idx]

        positions, (xmax, ymax) = sp_pack(gamma_plus, gamma_minus, widths, heights)
        new_cost = compute_cost(positions)

        delta = new_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_positions = dict(positions)
        else:
            arr[i_idx], arr[j_idx] = arr[j_idx], arr[i_idx]

        T *= cooling
        moves += 1

    elapsed = time.time() - start
    total_area_best = sum(best_positions[i][2]*best_positions[i][3] for i in range(n))
    bx = max(best_positions[i][0]+best_positions[i][2] for i in range(n))
    by = max(best_positions[i][1]+best_positions[i][3] for i in range(n))
    util = total_area_best / max(bx*by, 1e-6)

    print(f"  SP-SA (obstacles): {moves} moves in {elapsed:.1f}s")
    print(f"  bbox={bx:.1f}x{by:.1f} area={bx*by:.0f}")
    print(f"  utilization={util:.3f} cost={best_cost:.0f}")

    return best_positions, (bx, by), best_cost, util
