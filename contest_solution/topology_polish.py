"""Fast fixed-topology HPWL polish for the low/medium-size solve path.

The incumbent dimensions, bounding box, preplaced coordinates, satisfied
boundary equalities, non-overlap topology, and a spanning forest of every
existing grouping component are frozen.  Each contact-connected component is
then translated by weighted-median coordinate descent.  A move cannot worsen
HPWL for the frozen topology; the caller must still run the normal official-
fidelity feasibility/cost gate before accepting the result.

This module is stdlib-only and accepts real torch tensors, the packaging tensor
stub, or ordinary nested lists.  It is enabled for n <= 90, where paired
runtime/quality gates improve the score at the assumed 1s median; broader use
remains deferred until beta reveals the actual field runtime median.
"""

from __future__ import annotations

import math


def _number(value):
    return float(value.item()) if hasattr(value, "item") else float(value)


def _ncols(constraints):
    if constraints is None:
        return 0
    shape = getattr(constraints, "shape", None)
    if shape is not None and len(shape) > 1:
        return int(shape[1])
    return len(constraints[0]) if len(constraints) else 0


def _constraint(constraints, row, column):
    try:
        return _number(constraints[row, column])
    except (TypeError, IndexError):
        return _number(constraints[row][column])


def _valid_edges(rows):
    result = []
    if rows is None:
        return result
    for row in rows:
        a = int(_number(row[0]))
        b = int(_number(row[1]))
        weight = _number(row[2])
        if a >= 0 and b >= 0 and weight > 0.0:
            result.append((a, b, weight))
    return result


def _contact_forest(positions, constraints):
    """Return an exact-contact spanning forest for every grouping component."""
    n = len(positions)
    if _ncols(constraints) <= 3:
        return []
    max_group = max((int(_constraint(constraints, i, 3)) for i in range(n)), default=0)
    contacts = []
    for group_id in range(1, max_group + 1):
        group = [i for i in range(n) if int(_constraint(constraints, i, 3)) == group_id]
        parent = {i: i for i in group}

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for offset, i in enumerate(group):
            xi, yi, wi, hi = positions[i]
            for j in group[offset + 1:]:
                xj, yj, wj, hj = positions[j]
                x_overlap = min(xi + wi, xj + wj) - max(xi, xj)
                y_overlap = min(yi + hi, yj + hj) - max(yi, yj)
                kind = None
                if xi + wi == xj and y_overlap > 0.0:
                    kind = "i_left_j"
                elif xj + wj == xi and y_overlap > 0.0:
                    kind = "j_left_i"
                elif yi + hi == yj and x_overlap > 0.0:
                    kind = "i_below_j"
                elif yj + hj == yi and x_overlap > 0.0:
                    kind = "j_below_i"
                if kind is None:
                    continue
                root_i, root_j = find(i), find(j)
                if root_i != root_j:
                    parent[root_j] = root_i
                    contacts.append((i, j, kind))
    return contacts


def _weighted_median(items):
    items.sort(key=lambda item: item[0])
    total = sum(weight for _, weight in items)
    cumulative = 0.0
    for value, weight in items:
        cumulative += weight
        if cumulative * 2.0 >= total:
            return value
    return items[-1][0]


def polish_fixed_topology(
    positions,
    b2b_edges,
    p2b_edges,
    pins,
    constraints,
    *,
    sweeps=1,
    reverse=False,
):
    """Return an HPWL-polished candidate with the incumbent topology frozen."""
    rectangles = [tuple(map(float, rectangle)) for rectangle in positions]
    n = len(rectangles)
    if n < 2:
        return list(rectangles)
    b2b = _valid_edges(b2b_edges)
    p2b = _valid_edges(p2b_edges)
    pin_values = pins.tolist() if hasattr(pins, "tolist") else pins
    contacts = _contact_forest(rectangles, constraints)
    contact_pairs = {(min(i, j), max(i, j)) for i, j, _ in contacts}
    x_min = min(r[0] for r in rectangles)
    y_min = min(r[1] for r in rectangles)
    x_max = max(r[0] + r[2] for r in rectangles)
    y_max = max(r[1] + r[3] for r in rectangles)

    # Choose the incumbent's tightest legal separating direction once; both
    # axis solves reuse it.
    separations = []
    for i in range(n):
        xi, yi, wi, hi = rectangles[i]
        for j in range(i + 1, n):
            if (i, j) in contact_pairs:
                continue
            xj, yj, wj, hj = rectangles[j]
            candidates = (
                (xj - (xi + wi), 0, i, j),
                (xi - (xj + wj), 0, j, i),
                (yj - (yi + hi), 1, i, j),
                (yi - (yj + hj), 1, j, i),
            )
            legal = [candidate for candidate in candidates if candidate[0] >= -1e-9]
            chosen = (
                min(legal, key=lambda candidate: max(candidate[0], 0.0))
                if legal else max(candidates, key=lambda candidate: candidate[0])
            )
            separations.append(chosen)

    def solve_axis(axis):
        coordinate = [rectangle[axis] for rectangle in rectangles]
        size = [rectangle[axis + 2] for rectangle in rectangles]
        bbox_low, bbox_high = (x_min, x_max) if axis == 0 else (y_min, y_max)

        # Equality graph u_child = u_parent + delta for exact group contacts.
        graph = [[] for _ in range(n)]
        for i, j, kind in contacts:
            if axis == 0 and kind == "i_left_j":
                graph[i].append((j, size[i])); graph[j].append((i, -size[i]))
            elif axis == 0 and kind == "j_left_i":
                graph[j].append((i, size[j])); graph[i].append((j, -size[j]))
            elif axis == 1 and kind == "i_below_j":
                graph[i].append((j, size[i])); graph[j].append((i, -size[i]))
            elif axis == 1 and kind == "j_below_i":
                graph[j].append((i, size[j])); graph[i].append((j, -size[j]))

        component = [-1] * n
        relative = [0.0] * n
        members = []
        roots = []
        for start in range(n):
            if component[start] != -1:
                continue
            component_id = len(members)
            roots.append(start)
            members.append([])
            component[start] = component_id
            stack = [start]
            while stack:
                node = stack.pop()
                members[component_id].append(node)
                for neighbor, delta in graph[node]:
                    if component[neighbor] == -1:
                        component[neighbor] = component_id
                        relative[neighbor] = relative[node] + delta
                        stack.append(neighbor)

        count = len(members)
        base = [coordinate[root] for root in roots]
        lower = [-math.inf] * count
        upper = [math.inf] * count
        low_extreme = min(range(n), key=lambda i: coordinate[i])
        high_extreme = max(range(n), key=lambda i: coordinate[i] + size[i])
        low_bit = 1 if axis == 0 else 8
        high_bit = 2 if axis == 0 else 4
        columns = _ncols(constraints)
        for i in range(n):
            low = bbox_low
            high = bbox_high - size[i]
            boundary = int(_constraint(constraints, i, 4)) if columns > 4 else 0
            if boundary & low_bit and abs(coordinate[i] - bbox_low) < 1e-6:
                low = high = bbox_low
            if boundary & high_bit and abs(coordinate[i] + size[i] - bbox_high) < 1e-6:
                low = high = bbox_high - size[i]
            if i == low_extreme or i == high_extreme:
                low = high = coordinate[i]
            if columns > 1 and _constraint(constraints, i, 1) != 0.0:
                low = high = coordinate[i]
            cid = component[i]
            lower[cid] = max(lower[cid], low - relative[i])
            upper[cid] = min(upper[cid], high - relative[i])
        for cid in range(count):
            if lower[cid] > upper[cid] + 1e-7:
                return coordinate
            base[cid] = min(max(base[cid], lower[cid]), upper[cid])

        # Difference constraints base[a] - base[b] <= rhs.
        incident = [[] for _ in range(count)]
        for gap, relation_axis, left, right in separations:
            if relation_axis != axis:
                continue
            a, b = component[left], component[right]
            if a == b:
                continue
            allowance = max(0.0, -gap) + (1e-12 if gap < 0.0 else 0.0)
            rhs = -size[left] - relative[left] + relative[right] + allowance
            incident[a].append((1, b, rhs))
            incident[b].append((-1, a, rhs))

        net_terms = [[] for _ in range(count)]
        for first, second, weight in b2b:
            if first >= n or second >= n:
                continue
            a, b = component[first], component[second]
            if a == b:
                continue
            first_offset = relative[first] + size[first] * 0.5
            second_offset = relative[second] + size[second] * 0.5
            net_terms[a].append((b, second_offset - first_offset, weight))
            net_terms[b].append((a, first_offset - second_offset, weight))
        fixed_terms = [[] for _ in range(count)]
        for pin, block, weight in p2b:
            if block >= n or pin >= len(pin_values):
                continue
            cid = component[block]
            target = _number(pin_values[pin][axis]) - relative[block] - size[block] * 0.5
            fixed_terms[cid].append((target, weight))

        order = list(range(count))
        if reverse:
            order.reverse()
        for _ in range(max(1, int(sweeps))):
            largest_move = 0.0
            for cid in order:
                low, high = lower[cid], upper[cid]
                for role, other, rhs in incident[cid]:
                    if role == 1:
                        high = min(high, rhs + base[other])
                    else:
                        low = max(low, base[other] - rhs)
                if low > high + 1e-7:
                    continue
                terms = list(fixed_terms[cid])
                terms.extend(
                    (base[other] + delta, weight)
                    for other, delta, weight in net_terms[cid]
                )
                if not terms:
                    continue
                target = min(max(_weighted_median(terms), low), high)
                largest_move = max(largest_move, abs(target - base[cid]))
                base[cid] = target
            if largest_move < 1e-9:
                break

        values = [base[component[i]] + relative[i] for i in range(n)]
        # Re-propagate equality trees so shared coordinates are bit-exact for
        # Shapely rather than merely numerically equal.
        for cid, root in enumerate(roots):
            seen = {root}
            values[root] = base[cid]
            queue = [root]
            while queue:
                node = queue.pop(0)
                for neighbor, delta in graph[node]:
                    if neighbor in seen:
                        continue
                    values[neighbor] = values[node] + delta
                    seen.add(neighbor)
                    queue.append(neighbor)
        return values

    x_values = solve_axis(0)
    y_values = solve_axis(1)
    return [
        (x_values[i], y_values[i], rectangles[i][2], rectangles[i][3])
        for i in range(n)
    ]
