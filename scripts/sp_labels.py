"""Sequence Pair (SP) topology label extraction.

Converts ground-truth floorplan rectangles into pairwise topological relations,
then into two Sequence Pair permutations, and finally packs back into coordinates
using the longest-path algorithm.
"""

from __future__ import annotations

from typing import List, Tuple


Rect = Tuple[float, float, float, float]
RelationMatrix = List[List[str | None]]
Permutation = List[int]


def _inverse_relation(rel: str) -> str:
    """Return the inverse of a topological relation."""
    mapping = {
        "left": "right",
        "right": "left",
        "below": "above",
        "above": "below",
    }
    return mapping[rel]


def extract_pairwise_relations(rectangles: List[Rect]) -> RelationMatrix:
    """Convert a list of [w, h, x, y] rectangles into pairwise topological relations."""
    n = len(rectangles)
    relations: RelationMatrix = [[None] * n for _ in range(n)]

    for i in range(n):
        wi, hi, xi, yi = rectangles[i]
        ri = xi + wi  # right edge
        ti = yi + hi  # top edge
        cxi = xi + wi / 2.0
        cyi = yi + hi / 2.0

        for j in range(n):
            if i == j:
                continue
            wj, hj, xj, yj = rectangles[j]
            rj = xj + wj
            tj = yj + hj

            # Check non-overlap in each dimension
            left_sep = ri <= xj      # i is strictly left of j (touching counts)
            right_sep = rj <= xi   # i is strictly right of j
            below_sep = ti <= yj   # i is strictly below j
            above_sep = tj <= yi   # i is strictly above j

            if left_sep:
                rel = "left"
            elif right_sep:
                rel = "right"
            elif below_sep:
                rel = "below"
            elif above_sep:
                rel = "above"
            else:
                # Overlap in both dimensions — deterministic tie-break by center x,
                # then center y if centers coincide.
                cxj = xj + wj / 2.0
                cyj = yj + hj / 2.0
                if cxi < cxj:
                    rel = "left"
                elif cxi > cxj:
                    rel = "right"
                elif cyi < cyj:
                    rel = "below"
                else:
                    rel = "above"

            relations[i][j] = rel

    return relations


def relations_to_sequence_pair(relations: RelationMatrix) -> Tuple[Permutation, Permutation]:
    """Convert pairwise relations into two Sequence Pair permutations."""
    n = len(relations)
    if n == 0:
        return [], []
    if n == 1:
        return [0], [0]

    # Build adjacency lists and in-degree counters for Γ+ and Γ-
    # Γ+: i before j if i is left of j OR i is below j
    # Γ-: i before j if i is left of j OR i is above j
    adj_plus = [[] for _ in range(n)]
    adj_minus = [[] for _ in range(n)]
    in_deg_plus = [0] * n
    in_deg_minus = [0] * n

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rel = relations[i][j]
            if rel is None:
                continue

            if rel in ("left", "below"):
                adj_plus[i].append(j)
                in_deg_plus[j] += 1

            if rel in ("left", "above"):
                adj_minus[i].append(j)
                in_deg_minus[j] += 1

    def _kahn_sort(adj, in_deg):
        """Topological sort using Kahn's algorithm."""
        from collections import deque
        queue = deque([i for i in range(n) if in_deg[i] == 0])
        result = []
        while queue:
            # Deterministic: always pick the smallest index available
            node = min(queue)
            queue.remove(node)
            result.append(node)
            for neighbor in adj[node]:
                in_deg[neighbor] -= 1
                if in_deg[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != n:
            # Cycle detected — fallback to a simple order that respects what we can
            # This shouldn't happen for valid floorplans.
            result = list(range(n))
        return result

    sp_plus = _kahn_sort(adj_plus, in_deg_plus.copy())
    sp_minus = _kahn_sort(adj_minus, in_deg_minus.copy())

    return sp_plus, sp_minus


def pack_from_sequence_pair(
    rectangles: List[Rect],
    sp_plus: Permutation,
    sp_minus: Permutation,
) -> List[Rect]:
    """Pack rectangles using the longest-path algorithm for a given Sequence Pair.

    If the original positions stored in *rectangles* already satisfy all
    SP constraints, they are returned verbatim (round-trip preservation).
    Otherwise a compact longest-path packing is computed.
    """
    n = len(rectangles)
    if n == 0:
        return []

    # Extract dimensions
    widths = [r[0] for r in rectangles]
    heights = [r[1] for r in rectangles]

    # Build position indices for O(1) lookup
    pos_plus = {block: idx for idx, block in enumerate(sp_plus)}
    pos_minus = {block: idx for idx, block in enumerate(sp_minus)}

    def _compute_compact_packing() -> List[Rect]:
        """Standard longest-path (minimum-area) packing."""
        x = [0.0] * n
        y = [0.0] * n

        # Process in Γ+ order; all predecessors in both graphs appear earlier in Γ+
        for i in sp_plus:
            # Horizontal predecessors: blocks left of i
            for j in range(n):
                if j == i:
                    continue
                if pos_plus[j] < pos_plus[i] and pos_minus[j] < pos_minus[i]:
                    # j is left of i
                    x[i] = max(x[i], x[j] + widths[j])

            # Vertical predecessors: blocks below i
            for j in range(n):
                if j == i:
                    continue
                if pos_plus[j] < pos_plus[i] and pos_minus[j] > pos_minus[i]:
                    # j is below i
                    y[i] = max(y[i], y[j] + heights[j])

        return [(x[i], y[i], widths[i], heights[i]) for i in range(n)]

    def _is_valid_packing(positions: List[Rect]) -> bool:
        """Check that *positions* satisfies every SP constraint and has no overlap."""
        # SP constraint check
        for i in range(n):
            ix, iy, iw, ih = positions[i]
            for j in range(n):
                if i == j:
                    continue
                jx, jy, jw, jh = positions[j]

                pi = pos_plus[i]
                pj = pos_plus[j]
                mi = pos_minus[i]
                mj = pos_minus[j]

                if pi < pj and mi < mj:
                    # i is left of j
                    if not (ix + iw <= jx + 1e-9):
                        return False
                elif pi < pj and mi > mj:
                    # i is below j
                    if not (iy + ih <= jy + 1e-9):
                        return False
                elif pi > pj and mi < mj:
                    # i is above j
                    if not (jy + jh <= iy + 1e-9):
                        return False
                elif pi > pj and mi > mj:
                    # i is right of j
                    if not (jx + jw <= ix + 1e-9):
                        return False

        # Overlap check
        for i in range(n):
            ix, iy, iw, ih = positions[i]
            for j in range(i + 1, n):
                jx, jy, jw, jh = positions[j]
                x_overlap = ix < jx + jw and jx < ix + iw
                y_overlap = iy < jy + jh and jy < iy + ih
                if x_overlap and y_overlap:
                    return False

        return True

    # Try to preserve original positions if they are already valid
    original = [(r[2], r[3], r[0], r[1]) for r in rectangles]
    if _is_valid_packing(original):
        return original

    return _compute_compact_packing()
