"""Conservative fixed-topology repair for an already-feasible floorplan.

The repair is deliberately generator agnostic.  It uses only the submitted
instance, its incumbent placement, and the published contest constraints.
Every accepted move must:

* preserve hard feasibility and the incumbent's fixed/preplaced geometry;
* preserve one frozen non-overlap separation per pair and a spanning forest
  of every already-connected grouping component;
* not increase any soft-violation category;
* strictly reduce the total number of soft violations; and
* not increase HPWL or bounding-box area beyond the declared tiny tolerance.

If the incumbent is infeasible, an exception is raised internally, or no
candidate passes every gate, the original placement is returned unchanged.
The implementation is stdlib-only and accepts torch tensors, the submission
tensor shim, or ordinary Python sequences.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import time


Rect = tuple[float, float, float, float]


@dataclass(frozen=True)
class SoftCounts:
    boundary: int = 0
    grouping: int = 0
    mib: int = 0

    @property
    def total(self) -> int:
        return self.boundary + self.grouping + self.mib


@dataclass(frozen=True)
class RepairMetrics:
    feasible: bool
    hpwl: float
    bbox_area: float
    soft: SoftCounts


@dataclass(frozen=True)
class RepairConfig:
    """Search and acceptance controls.

    The HPWL allowance is ``max(hpwl_abs_tolerance,
    hpwl_rel_tolerance * max(1, incumbent_hpwl))``.  The default is small
    enough to cover floating summation noise, not a quality trade-off.
    """

    enable_boundary: bool = True
    enable_mib: bool = True
    enable_grouping: bool = True
    max_passes: int = 2
    max_factor_shapes: int = 12
    max_boundary_axis_values: int = 12
    max_cluster_candidates: int = 512
    max_aspect_ratio: float = 3.0
    require_safe_mib_pattern: bool = True
    hpwl_abs_tolerance: float = 1e-7
    hpwl_rel_tolerance: float = 1e-12
    bbox_abs_tolerance: float = 1e-7
    geometry_tolerance: float = 1e-7
    soft_contact_tolerance: float = 0.0
    overlap_tolerance: float = 1e-6
    area_tolerance: float = 0.01


@dataclass
class RepairReport:
    input_feasible: bool = False
    output_feasible: bool = False
    changed: bool = False
    fallback_reason: str | None = None
    attempted: dict[str, int] = field(
        default_factory=lambda: {"boundary": 0, "mib": 0, "grouping": 0}
    )
    accepted: dict[str, int] = field(
        default_factory=lambda: {"boundary": 0, "mib": 0, "grouping": 0}
    )
    before: RepairMetrics | None = None
    after: RepairMetrics | None = None
    runtime_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _number(value) -> float:
    return float(value.item()) if hasattr(value, "item") else float(value)


def _rows(value, limit: int | None = None) -> list[list[float]]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    result = []
    for row in value if limit is None else value[:limit]:
        if not isinstance(row, (list, tuple)):
            row = [row]
        result.append([_number(item) for item in row])
    return result


def _values(value, limit: int | None = None) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    data = value if limit is None else value[:limit]
    return [_number(item) for item in data]


def _rectangles(positions) -> list[Rect]:
    result = []
    for row in positions:
        if len(row) < 4:
            raise ValueError("each placement row must contain x, y, width, height")
        rectangle = tuple(_number(item) for item in row[:4])
        if not all(math.isfinite(item) for item in rectangle):
            raise ValueError("placement contains a non-finite coordinate")
        result.append(rectangle)
    return result


def _raw_matrix_number(value, row: int, column: int) -> float:
    try:
        return _number(value[row, column])
    except (TypeError, IndexError):
        return _number(value[row][column])


def _raw_safe_mib_patterns(
    positions, area_targets, constraints, config: RepairConfig
) -> list[tuple[tuple[int, ...], tuple[float, float]]]:
    """Allocation-light version of the deployable MIB opportunity gate."""

    if constraints is None:
        return []
    result = []
    groups: dict[int, list[int]] = {}
    for index in range(len(positions)):
        group = int(_raw_matrix_number(constraints, index, 2))
        if group > 0:
            groups.setdefault(group, []).append(index)
    for members in groups.values():
        if len(members) != 3:
            continue
        target_area = _number(area_targets[members[0]])
        if target_area <= 0.0 or any(
            abs(_number(area_targets[index]) - target_area) > 1e-7
            for index in members[1:]
        ):
            continue
        factors = enumerate_factor_shapes(
            target_area, max_aspect_ratio=config.max_aspect_ratio
        )
        if len(factors) < 4:
            continue
        factor_shapes = {
            (round(width, 4), round(height, 4)) for width, height in factors
        }
        shapes = {
            index: (
                round(_number(positions[index][2]), 4),
                round(_number(positions[index][3]), 4),
            )
            for index in members
        }
        present = {shape for shape in shapes.values() if shape in factor_shapes}
        if len(present) != 1 or len(set(shapes.values())) <= 1:
            continue
        target_shape = next(iter(present))
        outliers = [index for index in members if shapes[index] != target_shape]
        if not outliers:
            continue
        valid = True
        for index in outliers:
            fixed = _raw_matrix_number(constraints, index, 0) != 0.0
            preplaced = _raw_matrix_number(constraints, index, 1) != 0.0
            width = _number(positions[index][2])
            height = _number(positions[index][3])
            if fixed or preplaced or abs(width - height) > 1e-7:
                valid = False
                break
        if not valid:
            continue
        for index in members:
            fixed = _raw_matrix_number(constraints, index, 0) != 0.0
            preplaced = _raw_matrix_number(constraints, index, 1) != 0.0
            if (fixed or preplaced) and shapes[index] != target_shape:
                valid = False
                break
        if valid:
            target = next(
                shape for shape in factors if (
                    round(shape[0], 4), round(shape[1], 4)
                ) == target_shape
            )
            result.append((tuple(members), target))
    return result


def _raw_has_safe_mib_pattern(
    positions, area_targets, constraints, config: RepairConfig
) -> bool:
    return bool(_raw_safe_mib_patterns(
        positions, area_targets, constraints, config
    ))


def _groups(constraints: list[list[float]], column: int) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    for index, row in enumerate(constraints):
        group = int(row[column]) if len(row) > column else 0
        if group > 0:
            result.setdefault(group, []).append(index)
    return result


def enumerate_factor_shapes(
    area,
    *,
    max_aspect_ratio: float = 3.0,
    tolerance: float = 1e-7,
) -> tuple[tuple[float, float], ...]:
    """Return deterministic oriented integer factor pairs of ``area``."""

    numeric = _number(area)
    integer = round(numeric)
    if numeric <= 0 or abs(numeric - integer) > tolerance:
        return ()
    shapes: set[tuple[float, float]] = set()
    for width in range(1, math.isqrt(int(integer)) + 1):
        if int(integer) % width:
            continue
        height = int(integer) // width
        if max(width / height, height / width) <= max_aspect_ratio + tolerance:
            shapes.add((float(width), float(height)))
            shapes.add((float(height), float(width)))
    return tuple(
        sorted(
            shapes,
            key=lambda shape: (
                abs(math.log(shape[0] / shape[1])),
                shape[0],
                shape[1],
            ),
        )
    )


def _valid_edges(rows) -> list[tuple[int, int, float]]:
    result = []
    for row in _rows(rows):
        if len(row) < 3:
            continue
        first, second, weight = int(row[0]), int(row[1]), row[2]
        if first >= 0 and second >= 0 and math.isfinite(weight) and weight != 0.0:
            result.append((first, second, weight))
    return result


def _pin_rows(pins) -> list[tuple[float, float]]:
    return [tuple(row[:2]) for row in _rows(pins) if len(row) >= 2]


def _hpwl_from_valid(positions, b2b, p2b, pin_values) -> float:
    rectangles = positions
    value = 0.0
    for first, second, weight in b2b:
        if first >= len(rectangles) or second >= len(rectangles):
            continue
        a, b = rectangles[first], rectangles[second]
        value += weight * (
            abs((a[0] + a[2] * 0.5) - (b[0] + b[2] * 0.5))
            + abs((a[1] + a[3] * 0.5) - (b[1] + b[3] * 0.5))
        )
    for pin, block, weight in p2b:
        if block >= len(rectangles) or pin >= len(pin_values):
            continue
        px, py = pin_values[pin]
        if px == -1.0 or py == -1.0:
            continue
        rectangle = rectangles[block]
        value += weight * (
            abs(px - (rectangle[0] + rectangle[2] * 0.5))
            + abs(py - (rectangle[1] + rectangle[3] * 0.5))
        )
    return value


def calculate_hpwl(positions, b2b_edges, p2b_edges, pins) -> float:
    return _hpwl_from_valid(
        positions,
        _valid_edges(b2b_edges),
        _valid_edges(p2b_edges),
        _pin_rows(pins),
    )


def calculate_bbox_area(positions) -> float:
    if not positions:
        return 0.0
    x_min = min(rectangle[0] for rectangle in positions)
    y_min = min(rectangle[1] for rectangle in positions)
    x_max = max(rectangle[0] + rectangle[2] for rectangle in positions)
    y_max = max(rectangle[1] + rectangle[3] for rectangle in positions)
    return (x_max - x_min) * (y_max - y_min)


def _bbox(positions) -> tuple[float, float, float, float]:
    return (
        min(rectangle[0] for rectangle in positions),
        min(rectangle[1] for rectangle in positions),
        max(rectangle[0] + rectangle[2] for rectangle in positions),
        max(rectangle[1] + rectangle[3] for rectangle in positions),
    )


def _positive_overlap(a0: float, a1: float, b0: float, b1: float, eps: float) -> bool:
    return min(a1, b1) - max(a0, b0) > eps


def _contact_kind(a: Rect, b: Rect, tolerance: float) -> str | None:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    y_overlap = _positive_overlap(ay, ay + ah, by, by + bh, tolerance)
    x_overlap = _positive_overlap(ax, ax + aw, bx, bx + bw, tolerance)
    if abs(ax + aw - bx) <= tolerance and y_overlap:
        return "a_left_b"
    if abs(bx + bw - ax) <= tolerance and y_overlap:
        return "b_left_a"
    if abs(ay + ah - by) <= tolerance and x_overlap:
        return "a_below_b"
    if abs(by + bh - ay) <= tolerance and x_overlap:
        return "b_below_a"
    return None


def _components(
    positions: list[Rect], members: list[int], tolerance: float
) -> list[list[int]]:
    parent = {member: member for member in members}

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for offset, first in enumerate(members):
        for second in members[offset + 1 :]:
            if _contact_kind(positions[first], positions[second], tolerance) is not None:
                root_first, root_second = find(first), find(second)
                if root_first != root_second:
                    parent[root_second] = root_first
    result: dict[int, list[int]] = {}
    for member in members:
        result.setdefault(find(member), []).append(member)
    return sorted((sorted(group) for group in result.values()), key=lambda row: row[0])


def soft_counts(
    positions,
    constraints,
    *,
    geometry_tolerance: float = 0.0,
) -> SoftCounts:
    rectangles = _rectangles(positions)
    rows = _rows(constraints, len(rectangles))
    if len(rows) < len(rectangles):
        rows.extend([[] for _ in range(len(rectangles) - len(rows))])
    x_min, y_min, x_max, y_max = _bbox(rectangles)
    boundary = 0
    for index, rectangle in enumerate(rectangles):
        code = int(rows[index][4]) if len(rows[index]) > 4 else 0
        if not code:
            continue
        x, y, width, height = rectangle
        touches = {
            1: abs(x - x_min) < 1e-6,
            2: abs(x + width - x_max) < 1e-6,
            4: abs(y + height - y_max) < 1e-6,
            8: abs(y - y_min) < 1e-6,
        }
        if not all(touches[bit] for bit in (1, 2, 4, 8) if code & bit):
            boundary += 1

    grouping = 0
    for members in _groups(rows, 3).values():
        grouping += max(0, len(_components(rectangles, members, geometry_tolerance)) - 1)

    mib = 0
    for members in _groups(rows, 2).values():
        shapes = {
            (round(rectangles[index][2], 4), round(rectangles[index][3], 4))
            for index in members
        }
        mib += max(0, len(shapes) - 1)
    return SoftCounts(boundary=boundary, grouping=grouping, mib=mib)


def _overlap_free(positions: list[Rect], tolerance: float) -> bool:
    for first, a in enumerate(positions):
        if a[2] <= 0 or a[3] <= 0:
            return False
        for b in positions[first + 1 :]:
            x_overlap = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
            y_overlap = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
            if x_overlap > tolerance and y_overlap > tolerance:
                return False
    return True


def _hard_feasible(
    positions: list[Rect],
    original: list[Rect],
    area_targets: list[float],
    constraints: list[list[float]],
    target_positions: list[list[float]],
    config: RepairConfig,
) -> bool:
    if len(positions) != len(original) or not _overlap_free(
        positions, config.overlap_tolerance
    ):
        return False
    for index, rectangle in enumerate(positions):
        if not all(math.isfinite(value) for value in rectangle):
            return False
        row = constraints[index] if index < len(constraints) else []
        fixed = len(row) > 0 and row[0] != 0.0
        preplaced = len(row) > 1 and row[1] != 0.0
        if fixed or preplaced:
            if (
                abs(rectangle[2] - original[index][2]) > config.geometry_tolerance
                or abs(rectangle[3] - original[index][3]) > config.geometry_tolerance
            ):
                return False
        if preplaced and (
            abs(rectangle[0] - original[index][0]) > config.geometry_tolerance
            or abs(rectangle[1] - original[index][1]) > config.geometry_tolerance
        ):
            return False
        if index < len(target_positions) and len(target_positions[index]) >= 4:
            target = target_positions[index]
            if fixed or preplaced:
                if target[2] != -1.0 and abs(rectangle[2] - target[2]) > 1e-4:
                    return False
                if target[3] != -1.0 and abs(rectangle[3] - target[3]) > 1e-4:
                    return False
            if preplaced and (
                (target[0] != -1.0 and abs(rectangle[0] - target[0]) > 1e-4)
                or (target[1] != -1.0 and abs(rectangle[1] - target[1]) > 1e-4)
            ):
                return False
        if not (fixed or preplaced) and index < len(area_targets):
            target_area = area_targets[index]
            if target_area > 0.0:
                error = abs(rectangle[2] * rectangle[3] - target_area) / target_area
                if error > config.area_tolerance + 1e-12:
                    return False
    return True


def _contact_forest(
    positions: list[Rect], constraints: list[list[float]], tolerance: float
) -> list[tuple[int, int, str]]:
    result = []
    for members in _groups(constraints, 3).values():
        parent = {member: member for member in members}

        def find(item: int) -> int:
            while parent[item] != item:
                parent[item] = parent[parent[item]]
                item = parent[item]
            return item

        for offset, first in enumerate(members):
            for second in members[offset + 1 :]:
                kind = _contact_kind(positions[first], positions[second], tolerance)
                if kind is None:
                    continue
                root_first, root_second = find(first), find(second)
                if root_first != root_second:
                    parent[root_second] = root_first
                    result.append((first, second, kind))
    return result


def _separation_topology(
    positions: list[Rect], frozen_contacts: list[tuple[int, int, str]]
) -> list[tuple[int, int, int]]:
    contacts = {(min(a, b), max(a, b)) for a, b, _kind in frozen_contacts}
    result = []
    for first, a in enumerate(positions):
        for second in range(first + 1, len(positions)):
            if (first, second) in contacts:
                continue
            b = positions[second]
            candidates = (
                (b[0] - (a[0] + a[2]), 0, first, second),
                (a[0] - (b[0] + b[2]), 0, second, first),
                (b[1] - (a[1] + a[3]), 1, first, second),
                (a[1] - (b[1] + b[3]), 1, second, first),
            )
            legal = [candidate for candidate in candidates if candidate[0] >= -1e-9]
            if not legal:
                raise ValueError("incumbent contains an overlapping pair")
            _gap, axis, lower, upper = min(
                legal, key=lambda item: (max(item[0], 0.0), item[1], item[2], item[3])
            )
            result.append((axis, lower, upper))
    return result


def _topology_preserved(
    positions: list[Rect],
    separations: list[tuple[int, int, int]],
    contacts: list[tuple[int, int, str]],
    tolerance: float,
) -> bool:
    for axis, lower, upper in separations:
        if positions[lower][axis] + positions[lower][axis + 2] > (
            positions[upper][axis] + tolerance
        ):
            return False
    for first, second, kind in contacts:
        if _contact_kind(positions[first], positions[second], tolerance) != kind:
            return False
    return True


def _measure(
    positions: list[Rect],
    original: list[Rect],
    areas: list[float],
    constraints: list[list[float]],
    targets: list[list[float]],
    b2b_edges,
    p2b_edges,
    pins,
    config: RepairConfig,
) -> RepairMetrics:
    return RepairMetrics(
        feasible=_hard_feasible(
            positions, original, areas, constraints, targets, config
        ),
        hpwl=calculate_hpwl(positions, b2b_edges, p2b_edges, pins),
        bbox_area=calculate_bbox_area(positions),
        soft=soft_counts(
            positions,
            constraints,
            geometry_tolerance=config.soft_contact_tolerance,
        ),
    )


def _acceptable(
    candidate: list[Rect],
    current_metrics: RepairMetrics,
    original: list[Rect],
    areas: list[float],
    constraints: list[list[float]],
    targets: list[list[float]],
    b2b_edges,
    p2b_edges,
    pins,
    separations,
    contacts,
    config: RepairConfig,
) -> tuple[bool, RepairMetrics]:
    if not _topology_preserved(
        candidate, separations, contacts, config.geometry_tolerance
    ):
        # Most local proposals violate a frozen separation.  Reject them
        # before the more expensive HPWL and soft-component calculations.
        return False, current_metrics
    metrics = _measure(
        candidate,
        original,
        areas,
        constraints,
        targets,
        b2b_edges,
        p2b_edges,
        pins,
        config,
    )
    hpwl_allowance = max(
        config.hpwl_abs_tolerance,
        config.hpwl_rel_tolerance * max(1.0, abs(current_metrics.hpwl)),
    )
    old = current_metrics.soft
    new = metrics.soft
    accepted = (
        metrics.feasible
        and metrics.hpwl <= current_metrics.hpwl + hpwl_allowance
        and metrics.bbox_area <= current_metrics.bbox_area + config.bbox_abs_tolerance
        and new.boundary <= old.boundary
        and new.grouping <= old.grouping
        and new.mib <= old.mib
        and new.total < old.total
    )
    return accepted, metrics


def _candidate_key(item: tuple[list[Rect], RepairMetrics]):
    _positions, metrics = item
    return (metrics.soft.total, metrics.hpwl, metrics.bbox_area)


def _unique(values, tolerance: float, limit: int) -> list[float]:
    result = []
    for value in values:
        if not math.isfinite(value):
            continue
        if not any(abs(value - prior) <= tolerance for prior in result):
            result.append(float(value))
        if len(result) >= limit:
            break
    return result


def _boundary_axis_values(
    index: int,
    axis: int,
    size: float,
    current: list[Rect],
    config: RepairConfig,
) -> list[float]:
    old = current[index]
    old_size = old[axis + 2]
    values = [old[axis], old[axis] + old_size - size]
    for other_index, other in enumerate(current):
        if other_index == index:
            continue
        values.extend(
            (
                other[axis],
                other[axis] + other[axis + 2] - size,
                other[axis] - size,
                other[axis] + other[axis + 2],
            )
        )
    return _unique(
        values, config.geometry_tolerance, config.max_boundary_axis_values
    )


def _boundary_candidates(
    current: list[Rect],
    areas: list[float],
    constraints: list[list[float]],
    config: RepairConfig,
):
    x_min, y_min, x_max, y_max = _bbox(current)
    for index, old in enumerate(current):
        row = constraints[index] if index < len(constraints) else []
        fixed = len(row) > 0 and row[0] != 0.0
        preplaced = len(row) > 1 and row[1] != 0.0
        mib = int(row[2]) if len(row) > 2 else 0
        code = int(row[4]) if len(row) > 4 else 0
        if not code or fixed or preplaced:
            continue
        # Avoid enumerating an already-satisfied boundary block.
        x, y, width, height = old
        touches = {
            1: abs(x - x_min) < 1e-6,
            2: abs(x + width - x_max) < 1e-6,
            4: abs(y + height - y_max) < 1e-6,
            8: abs(y - y_min) < 1e-6,
        }
        if all(touches[bit] for bit in (1, 2, 4, 8) if code & bit):
            continue

        shapes = [old[2:4]]
        if mib == 0 and index < len(areas):
            factors = enumerate_factor_shapes(
                areas[index], max_aspect_ratio=config.max_aspect_ratio
            )
            factors = sorted(
                factors,
                key=lambda shape: (
                    abs(shape[0] - old[2]) + abs(shape[1] - old[3]),
                    shape,
                ),
            )[: config.max_factor_shapes]
            shapes.extend(factors)
        seen_shapes = set()
        for width, height in shapes:
            shape_key = (round(width, 12), round(height, 12))
            if shape_key in seen_shapes:
                continue
            seen_shapes.add(shape_key)
            if (code & 1) and (code & 2) and abs(width - (x_max - x_min)) > 1e-6:
                continue
            if (code & 4) and (code & 8) and abs(height - (y_max - y_min)) > 1e-6:
                continue
            if code & 1:
                x_values = [x_min]
            elif code & 2:
                x_values = [x_max - width]
            else:
                x_values = _boundary_axis_values(
                    index, 0, width, current, config
                )
            if code & 8:
                y_values = [y_min]
            elif code & 4:
                y_values = [y_max - height]
            else:
                y_values = _boundary_axis_values(
                    index, 1, height, current, config
                )
            for x in x_values:
                for y in y_values:
                    if (
                        x < x_min - config.geometry_tolerance
                        or y < y_min - config.geometry_tolerance
                        or x + width > x_max + config.geometry_tolerance
                        or y + height > y_max + config.geometry_tolerance
                    ):
                        continue
                    rectangle = (x, y, width, height)
                    if all(abs(rectangle[k] - old[k]) <= 1e-12 for k in range(4)):
                        continue
                    candidate = list(current)
                    candidate[index] = rectangle
                    yield candidate


def _mib_candidates(
    current: list[Rect],
    areas: list[float],
    constraints: list[list[float]],
    config: RepairConfig,
):
    generic_anchors = (
        "lower_left",
        "lower_right",
        "upper_left",
        "upper_right",
        "center",
    )
    eligible = (
        _eligible_mib_repairs(current, areas, constraints, config)
        if config.require_safe_mib_pattern
        else None
    )
    for members in _groups(constraints, 2).values():
        distinct = {
            (round(current[index][2], 4), round(current[index][3], 4))
            for index in members
        }
        if len(distinct) <= 1:
            continue
        safe_shape = None
        if eligible is not None:
            safe_shape = eligible.get(tuple(members))
            if safe_shape is None:
                continue
        locked_shapes = {
            current[index][2:4]
            for index in members
            if (len(constraints[index]) > 0 and constraints[index][0] != 0.0)
            or (len(constraints[index]) > 1 and constraints[index][1] != 0.0)
        }
        if len(locked_shapes) > 1:
            continue
        shapes: set[tuple[float, float]] = (
            {safe_shape} if safe_shape is not None else set(locked_shapes)
        )
        if not shapes:
            for index in members:
                if index < len(areas):
                    shapes.update(
                        enumerate_factor_shapes(
                            areas[index], max_aspect_ratio=config.max_aspect_ratio
                        )
                    )
        legal_shapes = []
        for shape in shapes:
            shape_area = shape[0] * shape[1]
            if all(
                index >= len(areas)
                or areas[index] <= 0.0
                or abs(shape_area - areas[index]) / areas[index]
                <= config.area_tolerance + 1e-12
                for index in members
                if not (
                    (len(constraints[index]) > 0 and constraints[index][0] != 0.0)
                    or (len(constraints[index]) > 1 and constraints[index][1] != 0.0)
                )
            ):
                legal_shapes.append(shape)
        legal_shapes.sort(
            key=lambda shape: sum(
                abs(shape[0] - current[index][2])
                + abs(shape[1] - current[index][3])
                for index in members
            )
        )
        anchors = ("lower_right",) if safe_shape is not None else generic_anchors
        for width, height in legal_shapes[: config.max_factor_shapes]:
            for anchor in anchors:
                candidate = list(current)
                changed = False
                for index in members:
                    row = constraints[index]
                    locked = (len(row) > 0 and row[0] != 0.0) or (
                        len(row) > 1 and row[1] != 0.0
                    )
                    if locked:
                        continue
                    x, y, old_width, old_height = current[index]
                    if anchor in ("lower_right", "upper_right"):
                        x += old_width - width
                    elif anchor == "center":
                        x += (old_width - width) * 0.5
                    if anchor in ("upper_left", "upper_right"):
                        y += old_height - height
                    elif anchor == "center":
                        y += (old_height - height) * 0.5
                    candidate[index] = (x, y, width, height)
                    changed = changed or candidate[index] != current[index]
                if changed:
                    yield candidate


def _eligible_mib_repairs(
    current: list[Rect],
    areas: list[float],
    constraints: list[list[float]],
    config: RepairConfig,
) -> dict[tuple[int, ...], tuple[float, float]]:
    """Return narrow, input/output-visible MIB repair opportunities.

    v32 emits square fallback shapes when its normalizer cannot place the
    group's chosen exact shape.  The safe pattern has three equal-area
    members, exactly one oriented factor shape already present, and only
    movable square fallbacks as outliers.  No case identity or dataset
    metadata participates in this gate.
    """

    result = {}
    for members in _groups(constraints, 2).values():
        if len(members) != 3 or any(index >= len(areas) for index in members):
            continue
        target_area = areas[members[0]]
        if target_area <= 0.0 or any(
            abs(areas[index] - target_area) > 1e-7 for index in members[1:]
        ):
            continue
        factors = enumerate_factor_shapes(
            target_area, max_aspect_ratio=config.max_aspect_ratio
        )
        # A single unoriented factor pair is effectively fixed-shape and is
        # especially likely to conflict with the incumbent topology.  Require
        # genuine shape flexibility before paying for the expensive ledger.
        if len(factors) < 4:
            continue
        factor_by_rounded = {
            (round(width, 4), round(height, 4)): (width, height)
            for width, height in factors
        }
        current_shapes = {
            (round(current[index][2], 4), round(current[index][3], 4))
            for index in members
        }
        present = [shape for shape in current_shapes if shape in factor_by_rounded]
        if len(present) != 1 or len(current_shapes) <= 1:
            continue
        target_rounded = present[0]
        outliers = [
            index
            for index in members
            if (round(current[index][2], 4), round(current[index][3], 4))
            != target_rounded
        ]
        if not outliers:
            continue
        if any(
            (len(constraints[index]) > 0 and constraints[index][0] != 0.0)
            or (len(constraints[index]) > 1 and constraints[index][1] != 0.0)
            or abs(current[index][2] - current[index][3]) > 1e-7
            for index in outliers
        ):
            continue
        if any(
            (
                (len(constraints[index]) > 0 and constraints[index][0] != 0.0)
                or (len(constraints[index]) > 1 and constraints[index][1] != 0.0)
            )
            and (round(current[index][2], 4), round(current[index][3], 4))
            != target_rounded
            for index in members
        ):
            continue
        result[tuple(members)] = factor_by_rounded[target_rounded]
    return result


def _cluster_translation_candidates(
    current: list[Rect],
    mover: list[int],
    target: list[int],
    constraints: list[list[float]],
    config: RepairConfig,
):
    if any(len(constraints[index]) > 1 and constraints[index][1] != 0.0 for index in mover):
        return
    shifts = []
    for first in mover:
        a = current[first]
        for second in target:
            b = current[second]
            y_shifts = (0.0, b[1] - a[1], b[1] + b[3] - (a[1] + a[3]))
            for dy in y_shifts:
                shifts.append((b[0] - (a[0] + a[2]), dy))
                shifts.append((b[0] + b[2] - a[0], dy))
            x_shifts = (0.0, b[0] - a[0], b[0] + b[2] - (a[0] + a[2]))
            for dx in x_shifts:
                shifts.append((dx, b[1] - (a[1] + a[3])))
                shifts.append((dx, b[1] + b[3] - a[1]))
    seen = set()
    emitted = 0
    for dx, dy in shifts:
        key = (round(dx, 12), round(dy, 12))
        if key in seen or (abs(dx) <= 1e-12 and abs(dy) <= 1e-12):
            continue
        seen.add(key)
        candidate = list(current)
        for index in mover:
            x, y, width, height = current[index]
            candidate[index] = (x + dx, y + dy, width, height)
        # Require that this translation really adds a mover-target edge contact.
        if not any(
            _contact_kind(candidate[first], candidate[second], config.geometry_tolerance)
            is not None
            for first in mover
            for second in target
        ):
            continue
        yield candidate
        emitted += 1
        if emitted >= config.max_cluster_candidates:
            return


def _grouping_candidates(
    current: list[Rect],
    constraints: list[list[float]],
    config: RepairConfig,
):
    for members in _groups(constraints, 3).values():
        components = _components(current, members, config.geometry_tolerance)
        if len(components) <= 1:
            continue
        for mover_index, mover in enumerate(components):
            for target_index, target in enumerate(components):
                if mover_index == target_index:
                    continue
                yield from _cluster_translation_candidates(
                    current, mover, target, constraints, config
                )


def _changed_topology_preserved(
    original: list[Rect],
    candidate: list[Rect],
    changed: set[int],
    constraints: list[list[float]],
    config: RepairConfig,
) -> bool:
    contacts = _contact_forest(
        original, constraints, config.geometry_tolerance
    )
    contact_map = {
        (min(first, second), max(first, second)): (first, second, kind)
        for first, second, kind in contacts
    }
    checked = set()
    for first in changed:
        for second in range(len(original)):
            if first == second:
                continue
            pair = (min(first, second), max(first, second))
            if pair in checked:
                continue
            checked.add(pair)
            contact = contact_map.get(pair)
            if contact is not None:
                a, b, kind = contact
                if _contact_kind(
                    candidate[a], candidate[b], config.geometry_tolerance
                ) != kind:
                    return False
                continue
            a, b = original[pair[0]], original[pair[1]]
            candidates = (
                (b[0] - (a[0] + a[2]), 0, pair[0], pair[1]),
                (a[0] - (b[0] + b[2]), 0, pair[1], pair[0]),
                (b[1] - (a[1] + a[3]), 1, pair[0], pair[1]),
                (a[1] - (b[1] + b[3]), 1, pair[1], pair[0]),
            )
            legal = [item for item in candidates if item[0] >= -1e-9]
            if not legal:
                return False
            _gap, axis, lower, upper = min(
                legal,
                key=lambda item: (
                    max(item[0], 0.0), item[1], item[2], item[3]
                ),
            )
            if candidate[lower][axis] + candidate[lower][axis + 2] > (
                candidate[upper][axis] + config.geometry_tolerance
            ):
                return False
    return True


def _changed_overlap_free(
    candidate: list[Rect], changed: set[int], tolerance: float
) -> bool:
    checked = set()
    for first in changed:
        a = candidate[first]
        if a[2] <= 0.0 or a[3] <= 0.0:
            return False
        for second, b in enumerate(candidate):
            if first == second:
                continue
            pair = (min(first, second), max(first, second))
            if pair in checked:
                continue
            checked.add(pair)
            x_overlap = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
            y_overlap = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
            if x_overlap > tolerance and y_overlap > tolerance:
                return False
    return True


def _repair_one_safe_mib_pattern(
    positions,
    area_targets,
    b2b_connectivity,
    p2b_connectivity,
    pins_pos,
    constraints,
    pattern: tuple[tuple[int, ...], tuple[float, float]],
    config: RepairConfig,
    started: float,
):
    """Specialized O(k*n + edges) acceptance path for one safe MIB group."""

    report = RepairReport(input_feasible=True, output_feasible=True)
    original = _rectangles(positions)
    constraint_rows = _rows(constraints, len(original))
    if len(constraint_rows) < len(original):
        constraint_rows.extend(
            [[] for _ in range(len(original) - len(constraint_rows))]
        )
    members, (width, height) = pattern
    candidate = list(original)
    changed = set()
    target_rounded = (round(width, 4), round(height, 4))
    for index in members:
        old = original[index]
        if (round(old[2], 4), round(old[3], 4)) == target_rounded:
            continue
        candidate[index] = (
            old[0] + old[2] - width,
            old[1],
            width,
            height,
        )
        changed.add(index)
    report.attempted["mib"] = 1
    b2b = _valid_edges(b2b_connectivity)
    p2b = _valid_edges(p2b_connectivity)
    pins = _pin_rows(pins_pos)
    before = RepairMetrics(
        feasible=True,
        hpwl=_hpwl_from_valid(original, b2b, p2b, pins),
        bbox_area=calculate_bbox_area(original),
        soft=soft_counts(
            original,
            constraint_rows,
            geometry_tolerance=config.soft_contact_tolerance,
        ),
    )
    report.before = before
    hard_ok = bool(changed) and _changed_overlap_free(
        candidate, changed, config.overlap_tolerance
    )
    if hard_ok:
        for index in changed:
            target_area = _number(area_targets[index])
            if target_area <= 0.0 or (
                abs(candidate[index][2] * candidate[index][3] - target_area)
                / target_area
                > config.area_tolerance + 1e-12
            ):
                hard_ok = False
                break
    topology_ok = hard_ok and _changed_topology_preserved(
        original, candidate, changed, constraint_rows, config
    )
    after = RepairMetrics(
        feasible=bool(hard_ok and topology_ok),
        hpwl=_hpwl_from_valid(candidate, b2b, p2b, pins),
        bbox_area=calculate_bbox_area(candidate),
        soft=soft_counts(
            candidate,
            constraint_rows,
            geometry_tolerance=config.soft_contact_tolerance,
        ),
    )
    allowance = max(
        config.hpwl_abs_tolerance,
        config.hpwl_rel_tolerance * max(1.0, abs(before.hpwl)),
    )
    accepted = (
        after.feasible
        and after.hpwl <= before.hpwl + allowance
        and after.bbox_area <= before.bbox_area + config.bbox_abs_tolerance
        and after.soft.boundary <= before.soft.boundary
        and after.soft.grouping <= before.soft.grouping
        and after.soft.mib <= before.soft.mib
        and after.soft.total < before.soft.total
    )
    if accepted:
        report.changed = True
        report.accepted["mib"] = 1
        report.after = after
        output = candidate
    else:
        report.fallback_reason = "no_candidate_passed_all_gates"
        report.after = before
        output = original
    report.runtime_seconds = time.perf_counter() - started
    return output, report


def repair_fixed_topology(
    positions,
    area_targets,
    b2b_connectivity,
    p2b_connectivity,
    pins_pos,
    constraints,
    target_positions=None,
    *,
    config: RepairConfig | None = None,
    return_report: bool = False,
):
    """Return a conservatively repaired placement, or the exact incumbent.

    The function never intentionally raises to its caller: malformed or
    infeasible incumbents are reported and returned byte-for-byte as numeric
    tuples.  ``return_report=True`` returns ``(positions, RepairReport)``.
    """

    started = time.perf_counter()
    config = config or RepairConfig()
    report = RepairReport()
    mib_only_safe = (
        config.enable_mib
        and not config.enable_boundary
        and not config.enable_grouping
        and config.require_safe_mib_pattern
    )
    if mib_only_safe:
        try:
            safe_patterns = _raw_safe_mib_patterns(
                positions, area_targets, constraints, config
            )
            if not safe_patterns:
                report.input_feasible = True
                report.output_feasible = True
                report.fallback_reason = "no_safe_mib_pattern_fast_path"
                report.runtime_seconds = time.perf_counter() - started
                return (positions, report) if return_report else positions
            if len(safe_patterns) == 1:
                repaired, specialized_report = _repair_one_safe_mib_pattern(
                    positions,
                    area_targets,
                    b2b_connectivity,
                    p2b_connectivity,
                    pins_pos,
                    constraints,
                    safe_patterns[0],
                    config,
                    started,
                )
                return (
                    (repaired, specialized_report)
                    if return_report
                    else repaired
                )
        except (IndexError, TypeError, ValueError):
            # Malformed values fall through to the normal fail-closed parser.
            pass
    try:
        original = _rectangles(positions)
    except Exception as error:
        # We cannot safely normalize malformed input; retain the caller's object.
        report.fallback_reason = f"malformed_incumbent:{type(error).__name__}"
        report.runtime_seconds = time.perf_counter() - started
        return (positions, report) if return_report else positions
    if not original:
        report.fallback_reason = "empty_incumbent"
        report.runtime_seconds = time.perf_counter() - started
        return (original, report) if return_report else original

    try:
        areas = _values(area_targets, len(original))
        constraint_rows = _rows(constraints, len(original))
        if len(constraint_rows) < len(original):
            constraint_rows.extend(
                [[] for _ in range(len(original) - len(constraint_rows))]
            )
        mib_only = (
            config.enable_mib
            and not config.enable_boundary
            and not config.enable_grouping
        )
        if mib_only:
            # Deployable fast path: most solver outputs either have no MIB
            # constraint or are already uniform.  Shape inconsistency is
            # entirely output-visible, so detect it in O(n) before building
            # the O(n^2) feasibility/topology ledger.  The API contract is an
            # already-feasible incumbent; returning it cannot invalidate it.
            if config.require_safe_mib_pattern:
                has_candidate = bool(
                    _eligible_mib_repairs(
                        original, areas, constraint_rows, config
                    )
                )
            else:
                has_candidate = any(
                    len(
                        {
                            (
                                round(original[index][2], 4),
                                round(original[index][3], 4),
                            )
                            for index in members
                        }
                    )
                    > 1
                    for members in _groups(constraint_rows, 2).values()
                )
            if not has_candidate:
                report.input_feasible = True
                report.output_feasible = True
                report.fallback_reason = "no_safe_mib_pattern_fast_path"
                report.runtime_seconds = time.perf_counter() - started
                return (original, report) if return_report else original
        target_rows = _rows(target_positions, len(original))
        before = _measure(
            original,
            original,
            areas,
            constraint_rows,
            target_rows,
            b2b_connectivity,
            p2b_connectivity,
            pins_pos,
            config,
        )
        report.before = before
        report.input_feasible = before.feasible
        if not before.feasible:
            report.fallback_reason = "incumbent_not_hard_feasible"
            report.after = before
            report.output_feasible = False
            report.runtime_seconds = time.perf_counter() - started
            return (original, report) if return_report else original

        contacts = _contact_forest(
            original, constraint_rows, config.geometry_tolerance
        )
        separations = _separation_topology(original, contacts)
        current = list(original)
        current_metrics = before
        mechanisms = []
        if config.enable_boundary:
            mechanisms.append(
                (
                    "boundary",
                    lambda state: _boundary_candidates(
                        state, areas, constraint_rows, config
                    ),
                )
            )
        if config.enable_mib:
            mechanisms.append(
                (
                    "mib",
                    lambda state: _mib_candidates(
                        state, areas, constraint_rows, config
                    ),
                )
            )
        if config.enable_grouping:
            mechanisms.append(
                (
                    "grouping",
                    lambda state: _grouping_candidates(
                        state, constraint_rows, config
                    ),
                )
            )

        for _pass in range(max(1, config.max_passes)):
            pass_changed = False
            for name, generate in mechanisms:
                best = None
                for candidate in generate(current):
                    report.attempted[name] += 1
                    accepted, metrics = _acceptable(
                        candidate,
                        current_metrics,
                        original,
                        areas,
                        constraint_rows,
                        target_rows,
                        b2b_connectivity,
                        p2b_connectivity,
                        pins_pos,
                        separations,
                        contacts,
                        config,
                    )
                    if accepted and (best is None or _candidate_key((candidate, metrics)) < _candidate_key(best)):
                        best = (candidate, metrics)
                if best is not None:
                    current, current_metrics = best
                    report.accepted[name] += 1
                    pass_changed = True
            if not pass_changed:
                break

        report.changed = current != original
        report.after = current_metrics
        report.output_feasible = current_metrics.feasible
        if not report.changed:
            report.fallback_reason = "no_candidate_passed_all_gates"
        report.runtime_seconds = time.perf_counter() - started
        return (current, report) if return_report else current
    except Exception as error:
        report.fallback_reason = f"repair_exception:{type(error).__name__}"
        report.after = report.before
        report.output_feasible = report.input_feasible
        report.runtime_seconds = time.perf_counter() - started
        return (original, report) if return_report else original


__all__ = [
    "RepairConfig",
    "RepairMetrics",
    "RepairReport",
    "SoftCounts",
    "calculate_bbox_area",
    "calculate_hpwl",
    "enumerate_factor_shapes",
    "repair_fixed_topology",
    "soft_counts",
]
