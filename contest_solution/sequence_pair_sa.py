"""
Sequence-pair SA floorplanner — M1/M2/M3/M4 implementation.

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
from typing import Dict, List, Tuple, Optional


def compute_soft_violations(full_positions, preplaced_rects, constraints_np=None):
    """Compute soft violation cost for the full layout (movable + preplaced).

    Returns a cost term reflecting:
    - Boundary violations: blocks not touching bbox edges (per constraint bitmask)
    - Cluster violations: connected components - 1 per cluster group
    - MIB violations: distinct shapes - 1 per MIB group

    Args:
        full_positions: list of (x, y, w, h) for ALL blocks (movable then preplaced)
        preplaced_rects: list of (x, y, w, h) for preplaced blocks
        constraints_np: numpy array [n_total, 5] (fixed, preplaced, mib_id, cluster_id, boundary_code)
                        If None, returns 0 (no constraint info available).

    Returns:
        float: total soft violation cost (0 = no violations)
    """
    if constraints_np is None:
        return 0.0

    n = len(full_positions)
    ncols = len(constraints_np[0]) if len(constraints_np) > 0 else 0

    violations = 0.0

    # Boundary violations
    if ncols > 4:
        x_min = min(p[0] for p in full_positions)
        y_min = min(p[1] for p in full_positions)
        x_max = max(p[0] + p[2] for p in full_positions)
        y_max = max(p[1] + p[3] for p in full_positions)
        for i in range(n):
            code = int(constraints_np[i][4]) if i < len(constraints_np) else 0
            if code == 0:
                continue
            bx, by, bw, bh = full_positions[i]
            touches = {
                1: abs(bx - x_min) < 1e-6,
                2: abs(bx + bw - x_max) < 1e-6,
                4: abs(by + bh - y_max) < 1e-6,
                8: abs(by - y_min) < 1e-6,
            }
            if not all(touches[bit] for bit in (1, 2, 4, 8) if code & bit):
                violations += 1.0

    # Cluster (grouping) violations: connected components - 1 per group
    if ncols > 3:
        cluster_ids = {}
        for i in range(n):
            gid = int(constraints_np[i][3]) if i < len(constraints_np) else 0
            if gid > 0:
                cluster_ids.setdefault(gid, []).append(i)
        for gid, members in cluster_ids.items():
            if len(members) < 2:
                continue
            # Union-find for connected components (edge-sharing)
            parent = {i: i for i in members}
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
            for pi, i in enumerate(members):
                x1, y1, w1, h1 = full_positions[i]
                for j in members[pi+1:]:
                    x2, y2, w2, h2 = full_positions[j]
                    touch_x = abs(x1 + w1 - x2) < 1e-6 or abs(x2 + w2 - x1) < 1e-6
                    touch_y = abs(y1 + h1 - y2) < 1e-6 or abs(y2 + h2 - y1) < 1e-6
                    overlap_x = min(y1+h1, y2+h2) - max(y1, y2) > 1e-6
                    overlap_y = min(x1+w1, x2+w2) - max(x1, x2) > 1e-6
                    if (touch_x and overlap_x) or (touch_y and overlap_y):
                        union(i, j)
            n_components = len({find(i) for i in members})
            violations += max(0, n_components - 1)

    # MIB violations: distinct shapes - 1 per group
    if ncols > 2:
        mib_groups = {}
        for i in range(n):
            gid = int(constraints_np[i][2]) if i < len(constraints_np) else 0
            if gid > 0:
                mib_groups.setdefault(gid, []).append(i)
        for gid, members in mib_groups.items():
            if len(members) < 2:
                continue
            shapes = set()
            for i in members:
                w, h = full_positions[i][2], full_positions[i][3]
                shapes.add((round(w, 4), round(h, 4)))
            violations += max(0, len(shapes) - 1)

    return violations


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
    x = [0.0] * n
    order_plus = sorted(range(n), key=lambda k: pos_plus[k])
    for j_idx in range(n):
        j = order_plus[j_idx]
        x_max = 0.0
        for i_idx in range(j_idx):
            i = order_plus[i_idx]
            if pos_minus[i] < pos_minus[j]:
                if x[i] + widths[i] > x_max:
                    x_max = x[i] + widths[i]
        x[j] = x_max

    # Longest-path for y coordinates (vertical constraint graph)
    y = [0.0] * n
    order_minus = sorted(range(n), key=lambda k: pos_minus[k])
    for j_idx in range(n):
        j = order_minus[j_idx]
        y_max = 0.0
        for i_idx in range(j_idx):
            i = order_minus[i_idx]
            if pos_plus[i] > pos_plus[j]:
                if y[i] + heights[i] > y_max:
                    y_max = y[i] + heights[i]
        y[j] = y_max

    positions = {}
    for i in range(n):
        positions[i] = (x[i], y[i], widths[i], heights[i])

    x_max = max(x[i] + widths[i] for i in range(n)) if n > 0 else 0
    y_max = max(y[i] + heights[i] for i in range(n)) if n > 0 else 0
    return positions, (x_max, y_max)


def snap_boundary_to_edge(positions, boundary_codes, eps=1e-6):
    """Snap boundary blocks to their required bbox edge.

    For each boundary block, compute the required edge from its bitmask code
    (1=left, 2=right, 4=top, 8=bottom), then translate the block so it
    touches that edge. Returns the snapped positions and the new bbox.

    Args:
        positions: dict {block_idx: (x, y, w, h)}
        boundary_codes: dict {block_idx: int_bitmask}
        eps: tolerance for edge touch

    Returns:
        snapped_positions: dict {block_idx: (x, y, w, h)}
        bbox: (x_max, y_max)
    """
    if not boundary_codes:
        # No boundary blocks — just compute bbox
        x_max = max(p[0] + p[2] for p in positions.values()) if positions else 0
        y_max = max(p[1] + p[3] for p in positions.values()) if positions else 0
        return dict(positions), (x_max, y_max)

    # First pass: compute bbox from non-boundary blocks + preplaced
    non_boundary = {i: positions[i] for i in positions if i not in boundary_codes}
    if non_boundary:
        x_min_nb = min(p[0] for p in non_boundary.values())
        y_min_nb = min(p[1] for p in non_boundary.values())
        x_max_nb = max(p[0] + p[2] for p in non_boundary.values())
        y_max_nb = max(p[1] + p[3] for p in non_boundary.values())
    else:
        x_min_nb, y_min_nb, x_max_nb, y_max_nb = 0, 0, 100, 100

    # Second pass: snap boundary blocks to their required edge
    snapped = dict(positions)
    for i, code in boundary_codes.items():
        if i not in snapped:
            continue
        x, y, w, h = snapped[i]
        nx, ny = x, y

        # Snap to required edges
        if code & 1:  # left edge
            nx = x_min_nb
        if code & 2:  # right edge
            nx = x_max_nb - w
        if code & 4:  # top edge
            ny = y_max_nb - h
        if code & 8:  # bottom edge
            ny = y_min_nb

        snapped[i] = (nx, ny, w, h)

    # Update bbox
    x_max = max(p[0] + p[2] for p in snapped.values()) if snapped else 0
    y_max = max(p[1] + p[3] for p in snapped.values()) if snapped else 0
    return snapped, (x_max, y_max)


def repair_overlaps(positions, eps=1e-6, max_iters=50):
    """Repair overlaps by pushing blocks apart along minimum displacement axis.

    Iterates until no overlaps remain or max_iters is reached.
    Returns repaired positions and whether all overlaps were resolved.

    Args:
        positions: dict {block_idx: (x, y, w, h)}
        eps: overlap tolerance
        max_iters: maximum repair iterations

    Returns:
        repaired_positions: dict {block_idx: (x, y, w, h)}
        feasible: bool — True if all overlaps resolved
    """
    n = len(positions)
    indices = list(positions.keys())
    repaired = dict(positions)

    for _iter in range(max_iters):
        overlaps_found = False
        for i_idx in range(len(indices)):
            for j_idx in range(i_idx + 1, len(indices)):
                i, j = indices[i_idx], indices[j_idx]
                xi, yi, wi, hi = repaired[i]
                xj, yj, wj, hj = repaired[j]

                ox = min(xi + wi, xj + wj) - max(xi, xj)
                oy = min(yi + hi, yj + hj) - max(yi, yj)

                if ox > eps and oy > eps:
                    overlaps_found = True
                    # Push apart along minimum displacement axis
                    if ox < oy:
                        # Push along x
                        push = ox * 0.5 + eps
                        if xi < xj:
                            repaired[i] = (xi - push, yi, wi, hi)
                            repaired[j] = (xj + push, yj, wj, hj)
                        else:
                            repaired[i] = (xi + push, yi, wi, hi)
                            repaired[j] = (xj - push, yj, wj, hj)
                    else:
                        # Push along y
                        push = oy * 0.5 + eps
                        if yi < yj:
                            repaired[i] = (xi, yi - push, wi, hi)
                            repaired[j] = (xj, yj + push, wj, hj)
                        else:
                            repaired[i] = (xi, yi + push, wi, hi)
                            repaired[j] = (xj, yj - push, wj, hj)

        if not overlaps_found:
            break

    # Final overlap check
    feasible = True
    for i_idx in range(len(indices)):
        for j_idx in range(i_idx + 1, len(indices)):
            i, j = indices[i_idx], indices[j_idx]
            xi, yi, wi, hi = repaired[i]
            xj, yj, wj, hj = repaired[j]
            ox = min(xi + wi, xj + wj) - max(xi, xj)
            oy = min(yi + hi, yj + hj) - max(yi, yj)
            if ox > eps and oy > eps:
                feasible = False
                break
        if not feasible:
            break

    return repaired, feasible


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
                          max_time=30.0, seed=42, penalty_weight=1e6,
                          constraints_np=None, soft_weight=0.01):
    """M3/M4: SA over SP treating preplaced as fixed obstacles, with soft constraints.

    Args:
        movable_count: number of movable blocks (indices 0..movable_count-1)
        preplaced_rects: list of (x, y, w, h) for preplaced blocks
        all_areas: area targets for ALL blocks (movable + preplaced)
        all_dims: dims for ALL blocks
        b2b_edges, p2b_edges, pins_pos: full connectivity
        max_time: time budget
        seed: random seed
        penalty_weight: large penalty for obstacle overlaps
        constraints_np: numpy array [n_total, 5] for soft constraint evaluation
        soft_weight: weight for soft violations in SA cost

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

    # Pre-compute N_soft normalization constant
    n_soft_val = 0
    if constraints_np is not None and len(constraints_np) > 0:
        n_total = len(constraints_np)
        ncols = len(constraints_np[0]) if n_total > 0 else 0
        if ncols > 4:
            n_soft_val += sum(1 for i in range(n_total) if constraints_np[i][4] != 0)
        if ncols > 2:
            mib_groups = {}
            for i in range(n_total):
                gid = int(constraints_np[i][2])
                if gid > 0:
                    mib_groups.setdefault(gid, 0)
                    mib_groups[gid] += 1
            for gid, cnt in mib_groups.items():
                n_soft_val += max(0, cnt - 1)
        if ncols > 3:
            cl_groups = {}
            for i in range(n_total):
                gid = int(constraints_np[i][3])
                if gid > 0:
                    cl_groups.setdefault(gid, 0)
                    cl_groups[gid] += 1
            for gid, cnt in cl_groups.items():
                n_soft_val += max(0, cnt - 1)

    def compute_cost(positions):
        """Cost = bbox_area + λ·HPWL + P·obstacle_overlaps + S·soft_violations"""
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
                    pen += ox * oy

        # Soft violations (boundary, cluster, MIB) — linear penalty for SA guidance
        # Use exp(2*V_rel) only for final selection, not SA acceptance
        soft_pen = 0.0
        if constraints_np is not None and n_soft_val > 0:
            full_positions = []
            for i in range(n):
                full_positions.append(positions[i])
            for r in preplaced_rects:
                full_positions.append(r)
            soft_pen = compute_soft_violations(full_positions, preplaced_rects, constraints_np)

        LAMBDA = 0.01
        base_cost = bbox + LAMBDA * hpwl + penalty_weight * pen
        # Linear soft penalty: guides SA without dominating
        soft_penalty = soft_weight * soft_pen * (bbox / max(n_soft_val, 1))
        return base_cost + soft_penalty

    # Init random SP
    gamma_plus = list(range(n))
    gamma_minus = list(range(n))
    random.shuffle(gamma_plus)
    random.shuffle(gamma_minus)

    positions, (xmax, ymax) = sp_pack(gamma_plus, gamma_minus, widths, heights)
    current_cost = compute_cost(positions)
    best_cost = current_cost
    best_positions = dict(positions)
    # Track best non-overlapping state separately (for final-state rejection)
    best_feasible_cost = float('inf')
    best_feasible_positions = None

    def _check_obstacle_overlap(pos):
        for i in range(n):
            ix, iy, iw, ih = pos[i]
            for (ox1, oy1, ox2, oy2) in obstacles:
                if min(ix+iw, ox2) - max(ix, ox1) > 1e-6 and min(iy+ih, oy2) - max(iy, oy1) > 1e-6:
                    return True
        return False

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
            # Track best feasible (non-overlapping) state
            if not _check_obstacle_overlap(positions) and new_cost < best_feasible_cost:
                best_feasible_cost = new_cost
                best_feasible_positions = dict(positions)
        else:
            arr[i_idx], arr[j_idx] = arr[j_idx], arr[i_idx]

        T *= cooling
        moves += 1

    # Final-state rejection: if best overlaps, fall back to best feasible
    if _check_obstacle_overlap(best_positions):
        if best_feasible_positions is not None:
            print(f"  WARNING: best state has obstacle overlaps, falling back to best feasible (cost={best_feasible_cost:.0f})")
            best_positions = best_feasible_positions
            best_cost = best_feasible_cost
        else:
            print(f"  WARNING: no feasible state found during SA")

    elapsed = time.time() - start
    total_area_best = sum(best_positions[i][2]*best_positions[i][3] for i in range(n))
    bx = max(best_positions[i][0]+best_positions[i][2] for i in range(n))
    by = max(best_positions[i][1]+best_positions[i][3] for i in range(n))
    util = total_area_best / max(bx*by, 1e-6)

    print(f"  SP-SA (obstacles): {moves} moves in {elapsed:.1f}s")
    print(f"  bbox={bx:.1f}x{by:.1f} area={bx*by:.0f}")
    print(f"  utilization={util:.3f} cost={best_cost:.0f}")

    return best_positions, (bx, by), best_cost, util


def _cluster_local_pack_standalone(group, dims, area_targets):
    """Standalone shelf-packing for a cluster group (no self dependency).

    Packs blocks in a cluster into a contiguous shelf layout, returning
    local positions and bounding-box dimensions. Reuses the same logic
    as MyOptimizer._cluster_local_pack.

    Args:
        group: list of block indices in the cluster
        dims: dict {block_idx: (w, h)}
        area_targets: list of area targets per block

    Returns:
        local: dict {block_idx: (x, y, w, h)} in local coordinates
        bw: bounding box width
        bh: bounding box height
    """
    if not group:
        return {}, 0.0, 0.0

    ordered = sorted(group, key=lambda i: (-dims[i][1], -dims[i][0], i))
    total_area = sum(dims[i][0] * dims[i][1] for i in ordered)
    cluster_factor = 1.34 if len(dims) >= 120 else 1.50
    row_width = max(
        math.sqrt(max(total_area, 1.0)) * cluster_factor,
        max(dims[i][0] for i in ordered)
    )

    local = {}
    x = 0.0
    y = 0.0
    row_h = 0.0
    max_w = 0.0
    for i in ordered:
        w, h = dims[i]
        if x > 0.0 and x + w > row_width:
            max_w = max(max_w, x)
            x = 0.0
            y += row_h
            row_h = 0.0
        local[i] = (x, y, w, h)
        x += w
        row_h = max(row_h, h)
    max_w = max(max_w, x)
    return local, max_w, y + row_h


# Aspect ratios for the shape lever (golden uses median 1.45, up to 3:1)
_ASPECT_RATIOS = [1.0, 1.3, 1.6, 2.0, 2.5, 3.0]


def _reshape_block(area, current_w, current_h, exclude_ratio=None):
    """Pick a new (w, h) for a block at constant area (±1%).

    Args:
        area: target area
        current_w, current_h: current dimensions
        exclude_ratio: if set, avoid this aspect ratio (for diversity)

    Returns:
        (new_w, new_h) tuple
    """
    current_r = current_w / max(current_h, 1e-9)
    candidates = []
    for r in _ASPECT_RATIOS:
        for ratio in [r, 1.0 / r]:
            if exclude_ratio is not None and abs(ratio - exclude_ratio) < 0.05:
                continue
            if abs(ratio - current_r) < 0.05:
                continue  # skip current ratio
            w = math.sqrt(area * ratio)
            h = area / w
            candidates.append((w, h))
    if not candidates:
        # Fallback: swap w and h
        return current_h, current_w
    return random.choice(candidates)


def sp_sa_full_layout(block_count, area_targets, dims, b2b_edges, p2b_edges, pins_pos,
                       constraints_np, max_time=30.0, seed=42):
    """N2: Full constraint-aware SP-SA with cluster super-blocks + shape lever.

    Architecture:
    - Interior clusters → super-blocks (guaranteed abutment by construction)
    - Boundary clusters → individual blocks (boundary penalty handles them)
    - MIB groups → shared aspect ratio (one shape per group, mutable in SA)
    - Per-block aspect ratio in move set (shape lever: golden uses ~1.45, up to 3:1)

    SA move set (weighted random):
    - 70% SP swap (existing — reorders blocks in the sequence-pair)
    - 15% aspect reshape (per-block or per-MIB group — changes dims)
    - 15% compound (SP swap + aspect reshape together)

    Args:
        block_count: total number of blocks
        area_targets: list of area targets per block
        dims: list of (w, h) per block
        b2b_edges, p2b_edges, pins_pos: connectivity
        constraints_np: numpy array [n, 5] (fixed, preplaced, mib_id, cluster_id, boundary_code)
        max_time: SA time budget
        seed: random seed

    Returns:
        positions: dict {block_idx: (x, y, w, h)} or None if infeasible
        bbox: (x_max, y_max)
        cost: float
        util: float
    """
    random.seed(seed)
    n = block_count

    # --- Step 1: Identify block types and groups ---
    boundary_codes = {}
    preplaced_set = set()
    cluster_groups = {}   # cluster_id -> list of block_ids
    mib_groups = {}       # mib_id -> list of block_ids
    if constraints_np is not None:
        for i in range(n):
            ncols = len(constraints_np[i])
            if ncols > 4:
                code = int(constraints_np[i][4])
                if code != 0:
                    boundary_codes[i] = code
            if ncols > 1 and constraints_np[i][1] != 0:
                preplaced_set.add(i)
            if ncols > 3:
                gid = int(constraints_np[i][3])
                if gid > 0:
                    cluster_groups.setdefault(gid, []).append(i)
            if ncols > 2:
                gid = int(constraints_np[i][2])
                if gid > 0:
                    mib_groups.setdefault(gid, []).append(i)

    # --- Step 2: All blocks are individual in the SP (no super-blocks) ---
    # Cluster abutment is encouraged via a strong pairwise gap penalty in the
    # SA cost function, plus post-SA snap to edges for boundary blocks.
    interior_clusters = {}  # unused in this approach, kept for interface
    super_blocks = {}       # no super-blocks
    # Block set for SP-SA: non-cluster blocks + one entry per super-block
    sp_blocks = []  # list of indices in the SP (some map to real blocks, some to super-blocks)
    sp_to_real = {}   # sp_idx -> block_idx (for non-cluster blocks)
    sp_to_super = {}  # sp_idx -> cluster_id (for super-blocks)
    real_to_sp = {}   # block_idx -> sp_idx

    # Mutable dims for the SA (will be modified by aspect moves)
    current_dims = {i: (dims[i][0], dims[i][1]) for i in range(n)}

    # All blocks are individual in the SP (no super-blocks)
    sp_idx = 0
    for i in range(n):
        sp_to_real[sp_idx] = i
        real_to_sp[i] = sp_idx
        sp_idx += 1

    n_sp = sp_idx  # total number of entities in the SP

    # Build SP widths/heights (mutable)
    sp_widths = {}
    sp_heights = {}
    for s in range(n_sp):
        bi = sp_to_real[s]
        sp_widths[s] = current_dims[bi][0]
        sp_heights[s] = current_dims[bi][1]

    # --- Step 4: Build adjacency for HPWL (uses real block indices) ---
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

    # Precompute n_soft for normalization
    n_soft = 0
    if constraints_np is not None:
        ncols = constraints_np.shape[1] if hasattr(constraints_np, 'shape') else 0
        if ncols > 4:
            n_soft += sum(1 for i in range(n) if constraints_np[i][4] != 0)
        if ncols > 3:
            for gid, cnt in cluster_groups.items():
                n_soft += max(0, len(cnt) - 1)
        if ncols > 2:
            for gid, cnt in mib_groups.items():
                n_soft += max(0, len(cnt) - 1)
    n_soft = max(1, n_soft)

    # MIB aspect tracking: one shared ratio per MIB group
    mib_aspect_ratio = {}  # mib_gid -> ratio (w/h)
    for gid, members in mib_groups.items():
        # Initial ratio from current dims
        if members:
            w0, h0 = current_dims[members[0]]
            mib_aspect_ratio[gid] = w0 / max(h0, 1e-9)

    # --- Step 5: Helper to get real block positions from SP positions ---
    def _expand_positions(sp_positions):
        """Map SP positions to real block positions (identity when no super-blocks)."""
        positions = {}
        for s in range(n_sp):
            bi = sp_to_real[s]
            positions[bi] = sp_positions[s]
        return positions

    # --- Step 6: Cost function (uses real block positions) ---
    # Precompute boundary cluster groups for cohesion penalty
    # Include ALL members of clusters that have boundary members (both
    # super-blocked non-boundary members and individual boundary members).
    # This encourages boundary members to stay near their cluster's super-block.
    boundary_cluster_groups = {}
    for gid, members in cluster_groups.items():
        if any(m in boundary_codes for m in members):
            boundary_cluster_groups[gid] = members

    def compute_cost(sp_positions):
        """Cost = bbox + λ·HPWL + boundary_penalty + cluster_cohesion_penalty.

        HPWL uses actual member centers (not super-block centers).
        Boundary penalty from member positions.
        Cluster cohesion penalty for boundary clusters (not super-blocked).
        """
        positions = _expand_positions(sp_positions)

        # HPWL from real block centers
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

        # Bbox from SP packing (includes super-block bounding boxes)
        xmax = max(sp_positions[s][0] + sp_widths[s] for s in range(n_sp))
        ymax = max(sp_positions[s][1] + sp_heights[s] for s in range(n_sp))
        xmin = min(sp_positions[s][0] for s in range(n_sp))
        ymin = min(sp_positions[s][1] for s in range(n_sp))
        bbox = xmax * ymax

        # Boundary penalty from real block positions
        boundary_pen = 0.0
        for i, code in boundary_codes.items():
            if i not in positions:
                continue
            bx, by, bw, bh = positions[i]
            if code & 1: boundary_pen += abs(bx - xmin)
            if code & 2: boundary_pen += abs(bx + bw - xmax)
            if code & 4: boundary_pen += abs(by + bh - ymax)
            if code & 8: boundary_pen += abs(by - ymin)

        # Cluster cohesion penalty: pairwise gap penalty for cluster members.
        # For each pair that are NOT edge-sharing, penalize the gap + 1.
        cluster_pen = 0.0
        for gid, members in cluster_groups.items():
            if len(members) < 2:
                continue
            for pi, i in enumerate(members):
                x1, y1, w1, h1 = positions[i]
                for j in members[pi + 1:]:
                    x2, y2, w2, h2 = positions[j]
                    touch_x = abs(x1 + w1 - x2) < 1e-3 or abs(x2 + w2 - x1) < 1e-3
                    touch_y = abs(y1 + h1 - y2) < 1e-3 or abs(y2 + h2 - y1) < 1e-3
                    overlap_x = min(y1 + h1, y2 + h2) - max(y1, y2) > 1e-3
                    overlap_y = min(x1 + w1, x2 + w2) - max(x1, x2) > 1e-3
                    sharing = (touch_x and overlap_x) or (touch_y and overlap_y)
                    if not sharing:
                        gap_x = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
                        gap_y = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
                        cluster_pen += gap_x + gap_y + 1.0

        LAMBDA = 0.01
        BOUNDARY_WEIGHT = 500.0
        CLUSTER_WEIGHT = 50.0  # penalty for cluster non-abutment
        return bbox + LAMBDA * hpwl + BOUNDARY_WEIGHT * boundary_pen + CLUSTER_WEIGHT * cluster_pen

    # --- Step 7: SA initialization ---
    gamma_plus = list(range(n_sp))
    gamma_minus = list(range(n_sp))
    random.shuffle(gamma_plus)
    random.shuffle(gamma_minus)

    raw_positions, _ = sp_pack(gamma_plus, gamma_minus, sp_widths, sp_heights)
    current_cost = compute_cost(raw_positions)
    best_cost = current_cost
    best_positions = dict(raw_positions)
    best_dims = {s: (sp_widths[s], sp_heights[s]) for s in range(n_sp)}

    # --- Step 8: SA loop ---
    T0 = 100.0; T_min = 0.01; cooling = 0.9995; T = T0
    moves = 0; accepts = 0; start = time.time()

    while T > T_min and time.time() - start < max_time:
        # Choose move type
        r = random.random()
        if r < 0.70:
            move_type = 'swap'
        elif r < 0.85:
            move_type = 'reshape'
        else:
            move_type = 'compound'

        saved_dims = {}  # for reverting dimension changes

        # --- SWAP move ---
        if move_type in ('swap', 'compound'):
            arr = gamma_plus if random.random() < 0.5 else gamma_minus
            i_idx = random.randint(0, n_sp - 1)
            j_idx = random.randint(0, n_sp - 1)
            while j_idx == i_idx:
                j_idx = random.randint(0, n_sp - 1)
            arr[i_idx], arr[j_idx] = arr[j_idx], arr[i_idx]

        # --- RESHAPE move ---
        if move_type in ('reshape', 'compound'):
            # Pick a random block to reshape
            target_s = random.randint(0, n_sp - 1)
            bi = sp_to_real[target_s]

            # Skip fixed/preplaced blocks
            if bi in preplaced_set:
                pass  # no reshape
            elif (constraints_np is not None and len(constraints_np[bi]) > 2
                  and int(constraints_np[bi][2]) > 0
                  and int(constraints_np[bi][2]) in mib_groups
                  and len(mib_groups[int(constraints_np[bi][2])]) > 1):
                # Reshape entire MIB group (shared aspect ratio)
                mib_gid = int(constraints_np[bi][2])
                mib_members = mib_groups[mib_gid]
                area0 = current_dims[mib_members[0]][0] * current_dims[mib_members[0]][1]
                new_w, new_h = _reshape_block(area0,
                                              current_dims[mib_members[0]][0],
                                              current_dims[mib_members[0]][1])
                new_ratio = new_w / max(new_h, 1e-9)

                for m in mib_members:
                    s = real_to_sp[m]
                    saved_dims[s] = (sp_widths[s], sp_heights[s])
                    area_m = area_targets[m]
                    new_wm = math.sqrt(area_m * new_ratio)
                    new_hm = area_m / new_wm
                    sp_widths[s] = new_wm
                    sp_heights[s] = new_hm
                    current_dims[m] = (new_wm, new_hm)
                mib_aspect_ratio[mib_gid] = new_ratio

            else:
                    # Reshape a single non-MIB block
                    saved_dims[target_s] = (sp_widths[target_s], sp_heights[target_s])
                    area = area_targets[bi]
                    new_w, new_h = _reshape_block(area, sp_widths[target_s], sp_heights[target_s])
                    sp_widths[target_s] = new_w
                    sp_heights[target_s] = new_h
                    current_dims[bi] = (new_w, new_h)

        # --- Evaluate move ---
        raw_new, _ = sp_pack(gamma_plus, gamma_minus, sp_widths, sp_heights)
        new_cost = compute_cost(raw_new)

        # Metropolis acceptance (using current_cost, not best_cost)
        delta = new_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
            current_cost = new_cost
            accepts += 1
            if current_cost < best_cost:
                best_cost = current_cost
                best_positions = dict(raw_new)
                best_dims = {s: (sp_widths[s], sp_heights[s]) for s in range(n_sp)}
        else:
            # Revert all changes
            if move_type in ('swap', 'compound'):
                arr[i_idx], arr[j_idx] = arr[j_idx], arr[i_idx]
            for s, (old_w, old_h) in saved_dims.items():
                sp_widths[s] = old_w
                sp_heights[s] = old_h
                bi = sp_to_real[s]
                current_dims[bi] = (old_w, old_h)

        T *= cooling
        moves += 1

    elapsed = time.time() - start

    if best_positions is None:
        return None, (0, 0), float('inf'), 0.0

    # --- Step 9: Expand super-blocks and post-process ---
    # Restore best dims
    for s, (w, h) in best_dims.items():
        sp_widths[s] = w
        sp_heights[s] = h

    # Re-pack at best SP to get best SP positions
    raw_best, _ = sp_pack(gamma_plus, gamma_minus, sp_widths, sp_heights)
    result = _expand_positions(raw_best)

    # Snap boundary blocks to bbox edges (all boundary blocks are individual,
    # not inside super-blocks, so snap_boundary_to_edge works directly)
    result = snap_boundary_to_edge(result, boundary_codes)[0]

    # Compute final metrics
    total_area_placed = sum(result[i][2] * result[i][3] for i in range(n))
    bx = max(result[i][0] + result[i][2] for i in range(n))
    by = max(result[i][1] + result[i][3] for i in range(n))
    util = total_area_placed / max(bx * by, 1e-6)

    # Full cost for reporting (includes soft violations)
    full_positions = [result[i] for i in range(n)]
    soft_pen = compute_soft_violations(full_positions, [], constraints_np) if constraints_np is not None else 0.0
    LAMBDA = 0.01
    cost = bx * by + LAMBDA * 0.0 + soft_pen * (bx * by / n_soft)  # HPWL=0 for reporting (already in SA cost)

    # Check boundary satisfaction
    xmin = min(result[i][0] for i in range(n))
    ymin = min(result[i][1] for i in range(n))
    boundary_ok = 0; boundary_fail = 0
    for i, code in boundary_codes.items():
        bx_i, by_i, bw, bh = result[i]
        touches = {
            1: abs(bx_i - xmin) < 1e-6,
            2: abs(bx_i + bw - bx) < 1e-6,
            4: abs(by_i + bh - by) < 1e-6,
            8: abs(by_i - ymin) < 1e-6,
        }
        if all(touches[bit] for bit in (1, 2, 4, 8) if code & bit):
            boundary_ok += 1
        else:
            boundary_fail += 1

    # Count cluster violations (should be 0 for super-blocked clusters)
    cluster_viols = 0
    if constraints_np is not None:
        for gid, members in cluster_groups.items():
            if len(members) < 2:
                continue
            parent = {i: i for i in members}
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
            for pi, i in enumerate(members):
                x1, y1, w1, h1 = result[i]
                for j in members[pi + 1:]:
                    x2, y2, w2, h2 = result[j]
                    touch_x = abs(x1 + w1 - x2) < 1e-6 or abs(x2 + w2 - x1) < 1e-6
                    touch_y = abs(y1 + h1 - y2) < 1e-6 or abs(y2 + h2 - y1) < 1e-6
                    overlap_x = min(y1 + h1, y2 + h2) - max(y1, y2) > 1e-6
                    overlap_y = min(x1 + w1, x2 + w2) - max(x1, x2) > 1e-6
                    if (touch_x and overlap_x) or (touch_y and overlap_y):
                        union(i, j)
            n_comp = len({find(i) for i in members})
            cluster_viols += max(0, n_comp - 1)

    # Count MIB violations
    mib_viols = 0
    if constraints_np is not None:
        for gid, members in mib_groups.items():
            if len(members) < 2:
                continue
            shapes = set()
            for i in members:
                w, h = result[i][2], result[i][3]
                shapes.add((round(w, 4), round(h, 4)))
            mib_viols += max(0, len(shapes) - 1)

    print(f"  SP-SA (full): {moves} moves, {accepts} accepts in {elapsed:.1f}s")
    print(f"  n_sp={n_sp} (super-blocks={len(interior_clusters)}, individual={n_sp - len(interior_clusters)})")
    print(f"  bbox={bx:.1f}x{by:.1f} area={bx * by:.0f}")
    print(f"  utilization={util:.3f} cost={cost:.0f}")
    print(f"  boundary: {boundary_ok} ok, {boundary_fail} fail")
    print(f"  cluster violations: {cluster_viols} (super-blocked: {len(interior_clusters)} groups)")
    print(f"  MIB violations: {mib_viols} ({len(mib_groups)} groups)")

    return result, (bx, by), cost, util


def sp_sa_n7_contiguous_clusters(block_count, area_targets, dims, b2b_edges, p2b_edges, pins_pos,
                                  constraints_np, max_time=30.0, seed=42):
    """N7: Flexible contiguous-cluster SP-SA.

    Key innovation: cluster members are consecutive in BOTH Γ+ and Γ−,
    which guarantees they pack into one contiguous region (abut → V_rel≈0).
    But their INTERNAL order + per-block aspect ratios are flexible (recovers util).

    This is NOT rigid super-blocks (which fixed shape → util 0.421).
    This is NOT another cluster penalty (which couldn't guarantee abutment).
    This is structural enforcement via SP ordering constraints.

    SA move set (weighted random):
    - 40% intra-cluster swap (reorder within a cluster's consecutive slot)
    - 30% inter-block swap (swap two non-cluster blocks or move cluster slots)
    - 15% cluster slot move (move a cluster's consecutive block to a new position)
    - 15% aspect reshape (per-block or per-MIB group)

    Args:
        block_count: total number of blocks
        area_targets: list of area targets per block
        dims: list of (w, h) per block
        b2b_edges, p2b_edges, pins_pos: connectivity
        constraints_np: numpy array [n, 5] (fixed, preplaced, mib_id, cluster_id, boundary_code)
        max_time: SA time budget
        seed: random seed

    Returns:
        positions: dict {block_idx: (x, y, w, h)} or None if infeasible
        bbox: (x_max, y_max)
        cost: float
        util: float
    """
    random.seed(seed)
    n = block_count

    # --- Step 1: Identify block types and groups ---
    boundary_codes = {}
    preplaced_set = set()
    cluster_groups = {}   # cluster_id -> list of block_ids
    mib_groups = {}       # mib_id -> list of block_ids
    if constraints_np is not None:
        for i in range(n):
            ncols = len(constraints_np[i])
            if ncols > 4:
                code = int(constraints_np[i][4])
                if code != 0:
                    boundary_codes[i] = code
            if ncols > 1 and constraints_np[i][1] != 0:
                preplaced_set.add(i)
            if ncols > 3:
                gid = int(constraints_np[i][3])
                if gid > 0:
                    cluster_groups.setdefault(gid, []).append(i)
            if ncols > 2:
                gid = int(constraints_np[i][2])
                if gid > 0:
                    mib_groups.setdefault(gid, []).append(i)

    # --- Step 2: Build SP with cluster members consecutive in both orders ---
    # Strategy: treat each cluster as a "slot" in the SP. The slot's members
    # are consecutive in both Γ+ and Γ−. Non-cluster blocks are individual.

    # Build list of "SP entities" — each is either a single block or a cluster slot
    sp_entities = []  # list of (type, id) where type is 'block' or 'cluster'
    cluster_slot_ids = {}  # cluster_id -> index in sp_entities
    block_to_entity = {}   # block_idx -> index in sp_entities

    for gid, members in cluster_groups.items():
        if len(members) >= 2:
            slot_idx = len(sp_entities)
            sp_entities.append(('cluster', gid))
            cluster_slot_ids[gid] = slot_idx
            for m in members:
                block_to_entity[m] = slot_idx

    for i in range(n):
        if i not in block_to_entity:
            slot_idx = len(sp_entities)
            sp_entities.append(('block', i))
            block_to_entity[i] = slot_idx

    n_entities = len(sp_entities)

    # Build mapping: entity_idx -> list of block indices
    entity_to_blocks = {}
    for idx, (etype, eid) in enumerate(sp_entities):
        if etype == 'cluster':
            entity_to_blocks[idx] = cluster_groups[eid]
        else:
            entity_to_blocks[idx] = [eid]

    # Mutable dims for the SA
    current_dims = {i: (dims[i][0], dims[i][1]) for i in range(n)}

    # --- Step 3: Initialize SP with consecutive cluster members ---
    # Γ+ and Γ− are lists of block indices, with cluster members consecutive
    # BUT the internal order can differ between Γ+ and Γ− (flexible 2D shapes)
    gamma_plus = []
    gamma_minus = []

    # Create entity orderings (random permutation of entities)
    entity_order_plus = list(range(n_entities))
    entity_order_minus = list(range(n_entities))
    random.shuffle(entity_order_plus)
    random.shuffle(entity_order_minus)

    # For each cluster, maintain separate internal orderings for Γ+ and Γ−
    cluster_internal_plus = {}   # cluster_gid -> list of block indices (order in Γ+)
    cluster_internal_minus = {}  # cluster_gid -> list of block indices (order in Γ−)
    for gid, members in cluster_groups.items():
        if len(members) >= 2:
            order_p = list(members)
            order_m = list(members)
            random.shuffle(order_p)
            random.shuffle(order_m)
            cluster_internal_plus[gid] = order_p
            cluster_internal_minus[gid] = order_m

    # Expand entities to block lists
    for eidx in entity_order_plus:
        etype, eid = sp_entities[eidx]
        if etype == 'cluster':
            gamma_plus.extend(cluster_internal_plus[eid])
        else:
            gamma_plus.extend(entity_to_blocks[eidx])

    for eidx in entity_order_minus:
        etype, eid = sp_entities[eidx]
        if etype == 'cluster':
            gamma_minus.extend(cluster_internal_minus[eid])
        else:
            gamma_minus.extend(entity_to_blocks[eidx])

    # Build SP widths/heights
    sp_widths = {i: current_dims[i][0] for i in range(n)}
    sp_heights = {i: current_dims[i][1] for i in range(n)}

    # --- Step 4: Build adjacency for HPWL ---
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

    # Precompute n_soft
    n_soft = 0
    if constraints_np is not None:
        ncols = constraints_np.shape[1] if hasattr(constraints_np, 'shape') else 0
        if ncols > 4:
            n_soft += sum(1 for i in range(n) if constraints_np[i][4] != 0)
        if ncols > 3:
            for gid, cnt in cluster_groups.items():
                n_soft += max(0, len(cnt) - 1)
        if ncols > 2:
            for gid, cnt in mib_groups.items():
                n_soft += max(0, len(cnt) - 1)
    n_soft = max(1, n_soft)

    # MIB aspect tracking
    mib_aspect_ratio = {}
    for gid, members in mib_groups.items():
        if members:
            w0, h0 = current_dims[members[0]]
            mib_aspect_ratio[gid] = w0 / max(h0, 1e-9)

    # --- Step 5: Cost function ---
    def compute_cost(positions_dict):
        """Cost = bbox + λ·HPWL + boundary_penalty.
        No cluster penalty needed — consecutive ordering guarantees abutment.
        """
        # HPWL
        hpwl = 0.0
        for i in range(n):
            cx_i = positions_dict[i][0] + positions_dict[i][2] * 0.5
            cy_i = positions_dict[i][1] + positions_dict[i][3] * 0.5
            for j, w in b_adj[i]:
                if j > i:
                    cx_j = positions_dict[j][0] + positions_dict[j][2] * 0.5
                    cy_j = positions_dict[j][1] + positions_dict[j][3] * 0.5
                    hpwl += w * (abs(cx_i - cx_j) + abs(cy_i - cy_j))
            for px, py, w in p_adj[i]:
                hpwl += w * (abs(cx_i - px) + abs(cy_i - py))

        # Bbox
        xmax = max(positions_dict[i][0] + positions_dict[i][2] for i in range(n))
        ymax = max(positions_dict[i][1] + positions_dict[i][3] for i in range(n))
        xmin = min(positions_dict[i][0] for i in range(n))
        ymin = min(positions_dict[i][1] for i in range(n))
        bbox = xmax * ymax

        # Boundary penalty
        boundary_pen = 0.0
        for i, code in boundary_codes.items():
            if i not in positions_dict:
                continue
            bx, by, bw, bh = positions_dict[i]
            if code & 1: boundary_pen += abs(bx - xmin)
            if code & 2: boundary_pen += abs(bx + bw - xmax)
            if code & 4: boundary_pen += abs(by + bh - ymax)
            if code & 8: boundary_pen += abs(by - ymin)

        LAMBDA = 0.01
        BOUNDARY_WEIGHT = 500.0
        return bbox + LAMBDA * hpwl + BOUNDARY_WEIGHT * boundary_pen

    # Helper: get positions from SP
    def get_positions(gp, gm):
        raw, _ = sp_pack(gp, gm, sp_widths, sp_heights)
        return raw

    # --- Step 6: SA initialization ---
    raw_positions = get_positions(gamma_plus, gamma_minus)
    current_cost = compute_cost(raw_positions)
    best_cost = current_cost
    best_positions = dict(raw_positions)
    best_dims = {i: (sp_widths[i], sp_heights[i]) for i in range(n)}

    # --- Step 7: Helper functions for constrained moves ---
    def get_entity_at_pos(order, pos):
        """Get the entity index at position `pos` in the block order."""
        block_idx = order[pos]
        return block_to_entity[block_idx]

    def get_entity_range(order, entity_idx):
        """Get the start and end positions of an entity's blocks in the order."""
        blocks = set(entity_to_blocks[entity_idx])
        start = None
        end = None
        for pos, block_idx in enumerate(order):
            if block_idx in blocks:
                if start is None:
                    start = pos
                end = pos
        if start is None:
            return None, None
        return start, end + 1  # [start, end)

    def swap_within_entity(order, entity_idx):
        """Swap two random blocks within a cluster entity's consecutive slot."""
        blocks = entity_to_blocks[entity_idx]
        if len(blocks) < 2:
            return False  # can't swap within a single block
        start, end = get_entity_range(order, entity_idx)
        if start is None:
            return False
        i_pos = random.randint(start, end - 1)
        j_pos = random.randint(start, end - 1)
        while j_pos == i_pos:
            j_pos = random.randint(start, end - 1)
        order[i_pos], order[j_pos] = order[j_pos], order[i_pos]
        return True

    def swap_entity_slots(order):
        """Swap two entity slots (preserving each entity's internal order)."""
        e1 = random.randint(0, n_entities - 1)
        e2 = random.randint(0, n_entities - 1)
        while e2 == e1:
            e2 = random.randint(0, n_entities - 1)

        # Find positions (ensure start1 < start2)
        start1, end1 = get_entity_range(order, e1)
        start2, end2 = get_entity_range(order, e2)
        if start1 is None or start2 is None:
            return False
        if start1 > start2:
            start1, end1, start2, end2 = start2, end2, start1, end1

        # Extract the three segments: [before1] [seq1] [between] [seq2] [after2]
        seq1 = order[start1:end1]
        seq2 = order[start2:end2]
        between = order[end1:start2]
        before = order[:start1]
        after = order[end2:]

        # Rebuild: [before] [seq2] [between] [seq1] [after]
        new_order = before + seq2 + between + seq1 + after

        # Replace order contents
        for i in range(len(order)):
            order[i] = new_order[i]
        return True

    def move_entity_slot(order):
        """Move a cluster entity's slot to a new random position."""
        # Pick a random cluster entity
        cluster_entities = [idx for idx, (etype, _) in enumerate(sp_entities) if etype == 'cluster']
        if not cluster_entities:
            return False
        eidx = random.choice(cluster_entities)

        # Get the entity's blocks
        blocks = entity_to_blocks[eidx]
        start, end = get_entity_range(order, eidx)
        if start is None:
            return False
        seq = order[start:end]

        # Remove from current position
        new_order = order[:start] + order[end:]

        # Insert at random new position
        insert_pos = random.randint(0, len(new_order))
        new_order = new_order[:insert_pos] + seq + new_order[insert_pos:]

        # Replace order contents
        for i in range(len(order)):
            order[i] = new_order[i]
        return True

    # --- Step 8: SA move helpers ---
    def _rebuild_sp():
        """Rebuild gamma_plus and gamma_minus from entity orderings + internal orderings."""
        gamma_plus.clear()
        gamma_minus.clear()
        for eidx in entity_order_plus:
            etype, eid = sp_entities[eidx]
            if etype == 'cluster':
                gamma_plus.extend(cluster_internal_plus[eid])
            else:
                gamma_plus.extend(entity_to_blocks[eidx])
        for eidx in entity_order_minus:
            etype, eid = sp_entities[eidx]
            if etype == 'cluster':
                gamma_minus.extend(cluster_internal_minus[eid])
            else:
                gamma_minus.extend(entity_to_blocks[eidx])

    def _swap_entity_slots(entity_order):
        """Swap two random entity slots in the entity ordering."""
        i = random.randint(0, n_entities - 1)
        j = random.randint(0, n_entities - 1)
        while j == i:
            j = random.randint(0, n_entities - 1)
        entity_order[i], entity_order[j] = entity_order[j], entity_order[i]

    def _move_entity_slot(entity_order):
        """Move a random cluster entity to a new position in the entity ordering."""
        cluster_idxs = [idx for idx, (etype, _) in enumerate(sp_entities) if etype == 'cluster']
        if not cluster_idxs:
            return
        src = random.choice(cluster_idxs)
        el = entity_order[src]
        entity_order.pop(src)
        dst = random.randint(0, len(entity_order))
        entity_order.insert(dst, el)

    # --- Step 9: SA loop ---
    T0 = 100.0; T_min = 0.01; cooling = 0.9995; T = T0
    moves = 0; accepts = 0; sa_start = time.time()

    while T > T_min and time.time() - sa_start < max_time:
        # Choose move type
        r = random.random()
        if r < 0.40:
            move_type = 'intra_cluster_swap'
        elif r < 0.70:
            move_type = 'inter_entity_swap'
        elif r < 0.85:
            move_type = 'cluster_slot_move'
        else:
            move_type = 'reshape'

        saved_dims = {}

        # --- Apply move ---
        if move_type == 'intra_cluster_swap':
            # Pick a random cluster and swap two blocks within its internal ordering
            cluster_gids = list(cluster_groups.keys())
            if cluster_gids:
                gid = random.choice(cluster_gids)
                if random.random() < 0.5:
                    internal = cluster_internal_plus[gid]
                else:
                    internal = cluster_internal_minus[gid]
                if len(internal) >= 2:
                    i = random.randint(0, len(internal) - 1)
                    j = random.randint(0, len(internal) - 1)
                    while j == i:
                        j = random.randint(0, len(internal) - 1)
                    internal[i], internal[j] = internal[j], internal[i]
                    _rebuild_sp()

        elif move_type == 'inter_entity_swap':
            if random.random() < 0.5:
                _swap_entity_slots(entity_order_plus)
            else:
                _swap_entity_slots(entity_order_minus)
            _rebuild_sp()

        elif move_type == 'cluster_slot_move':
            if random.random() < 0.5:
                _move_entity_slot(entity_order_plus)
            else:
                _move_entity_slot(entity_order_minus)
            _rebuild_sp()

        elif move_type == 'reshape':
            # Pick a random block to reshape
            target = random.randint(0, n - 1)

            # Skip fixed/preplaced blocks
            if target in preplaced_set:
                pass
            elif (constraints_np is not None and len(constraints_np[target]) > 2
                  and int(constraints_np[target][2]) > 0
                  and int(constraints_np[target][2]) in mib_groups
                  and len(mib_groups[int(constraints_np[target][2])]) > 1):
                # Reshape entire MIB group
                mib_gid = int(constraints_np[target][2])
                mib_members = mib_groups[mib_gid]
                area0 = current_dims[mib_members[0]][0] * current_dims[mib_members[0]][1]
                new_w, new_h = _reshape_block(area0,
                                              current_dims[mib_members[0]][0],
                                              current_dims[mib_members[0]][1])
                new_ratio = new_w / max(new_h, 1e-9)

                for m in mib_members:
                    saved_dims[m] = (sp_widths[m], sp_heights[m])
                    area_m = area_targets[m]
                    new_wm = math.sqrt(area_m * new_ratio)
                    new_hm = area_m / new_wm
                    sp_widths[m] = new_wm
                    sp_heights[m] = new_hm
                    current_dims[m] = (new_wm, new_hm)
                mib_aspect_ratio[mib_gid] = new_ratio

            else:
                # Reshape a single block
                saved_dims[target] = (sp_widths[target], sp_heights[target])
                area = area_targets[target]
                new_w, new_h = _reshape_block(area, sp_widths[target], sp_heights[target])
                sp_widths[target] = new_w
                sp_heights[target] = new_h
                current_dims[target] = (new_w, new_h)

        # --- Evaluate move ---
        raw_new = get_positions(gamma_plus, gamma_minus)
        new_cost = compute_cost(raw_new)

        # Metropolis acceptance
        delta = new_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
            current_cost = new_cost
            accepts += 1
            if current_cost < best_cost:
                best_cost = current_cost
                best_positions = dict(raw_new)
                best_dims = {i: (sp_widths[i], sp_heights[i]) for i in range(n)}
        else:
            # Revert
            for m, (old_w, old_h) in saved_dims.items():
                sp_widths[m] = old_w
                sp_heights[m] = old_h
                current_dims[m] = (old_w, old_h)

        T *= cooling
        moves += 1

    elapsed = time.time() - sa_start

    if best_positions is None:
        return None, (0, 0), float('inf'), 0.0

    # --- Step 10: Post-process ---
    # Restore best dims
    for i, (w, h) in best_dims.items():
        sp_widths[i] = w
        sp_heights[i] = h

    # Re-pack at best SP
    result = get_positions(gamma_plus, gamma_minus)

    # Snap boundary blocks to bbox edges
    result = snap_boundary_to_edge(result, boundary_codes)[0]

    # Compute final metrics
    total_area_placed = sum(result[i][2] * result[i][3] for i in range(n))
    bx = max(result[i][0] + result[i][2] for i in range(n))
    by = max(result[i][1] + result[i][3] for i in range(n))
    util = total_area_placed / max(bx * by, 1e-6)

    # Full cost for reporting
    full_positions = [result[i] for i in range(n)]
    soft_pen = compute_soft_violations(full_positions, [], constraints_np) if constraints_np is not None else 0.0
    LAMBDA = 0.01
    cost = bx * by + LAMBDA * 0.0 + soft_pen * (bx * by / n_soft)

    # Check boundary satisfaction
    xmin = min(result[i][0] for i in range(n))
    ymin = min(result[i][1] for i in range(n))
    boundary_ok = 0; boundary_fail = 0
    for i, code in boundary_codes.items():
        bx_i, by_i, bw, bh = result[i]
        touches = {
            1: abs(bx_i - xmin) < 1e-6,
            2: abs(bx_i + bw - bx) < 1e-6,
            4: abs(by_i + bh - by) < 1e-6,
            8: abs(by_i - ymin) < 1e-6,
        }
        if all(touches[bit] for bit in (1, 2, 4, 8) if code & bit):
            boundary_ok += 1
        else:
            boundary_fail += 1

    # Count cluster violations
    cluster_viols = 0
    if constraints_np is not None:
        for gid, members in cluster_groups.items():
            if len(members) < 2:
                continue
            parent = {i: i for i in members}
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
            for pi, i in enumerate(members):
                x1, y1, w1, h1 = result[i]
                for j in members[pi + 1:]:
                    x2, y2, w2, h2 = result[j]
                    touch_x = abs(x1 + w1 - x2) < 1e-6 or abs(x2 + w2 - x1) < 1e-6
                    touch_y = abs(y1 + h1 - y2) < 1e-6 or abs(y2 + h2 - y1) < 1e-6
                    overlap_x = min(y1 + h1, y2 + h2) - max(y1, y2) > 1e-6
                    overlap_y = min(x1 + w1, x2 + w2) - max(x1, x2) > 1e-6
                    if (touch_x and overlap_x) or (touch_y and overlap_y):
                        union(i, j)
            n_comp = len({find(i) for i in members})
            cluster_viols += max(0, n_comp - 1)

    # Count MIB violations
    mib_viols = 0
    if constraints_np is not None:
        for gid, members in mib_groups.items():
            if len(members) < 2:
                continue
            shapes = set()
            for i in members:
                w, h = result[i][2], result[i][3]
                shapes.add((round(w, 4), round(h, 4)))
            mib_viols += max(0, len(shapes) - 1)

    print(f"  N7 SP-SA: {moves} moves, {accepts} accepts in {elapsed:.1f}s")
    print(f"  n_entities={n_entities} (clusters={len(cluster_groups)}, blocks={n - sum(len(m) for m in cluster_groups.values())})")
    print(f"  bbox={bx:.1f}x{by:.1f} area={bx * by:.0f}")
    print(f"  utilization={util:.3f} cost={cost:.0f}")
    print(f"  boundary: {boundary_ok} ok, {boundary_fail} fail")
    print(f"  cluster violations: {cluster_viols}")
    print(f"  MIB violations: {mib_viols}")

    return result, (bx, by), cost, util
