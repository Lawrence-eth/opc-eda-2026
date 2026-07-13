"""Generator-agnostic dual-parent floorplan representation.

This module deliberately knows nothing about FloorSet source files, workers,
random seeds, or instance identifiers.  It operates only on a floorplan's
public geometry/constraint tensors and supervised labels:

* an oriented, exact factor-pair shape for every soft block;
* one horizontal B*-tree parent/side relation per non-root block; and
* one vertical support parent (or the floor) per block.

The oracle extraction helpers are research-only.  ``decode_dual_parent`` is
the deployable part: a predictor can supply the same labels without exposing
golden coordinates to inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence


EPSILON = 1e-6
Rect = tuple[float, float, float, float]
Shape = tuple[float, float]


class DualParentError(ValueError):
    """A stable, machine-classifiable representation or decoding failure."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        message = code if not detail else f"{code}: {detail}"
        super().__init__(message)


@dataclass(frozen=True)
class HorizontalRelation:
    """A B*-tree edge.

    ``side == 0`` means that the child starts at the parent's right edge.
    ``side == 1`` means that the child has the same left edge as the parent.
    """

    parent: int
    child: int
    side: int


@dataclass(frozen=True)
class DualParentLabels:
    """Complete structural labels needed by :func:`decode_dual_parent`."""

    root: int
    dimensions: tuple[Shape, ...]
    shape_options: tuple[tuple[Shape, ...], ...]
    selected_shape_indices: tuple[int, ...]
    horizontal: tuple[HorizontalRelation, ...]
    vertical_supports: tuple[int | None, ...]
    mib_inconsistent_groups: tuple[int, ...] = ()


@dataclass(frozen=True)
class GeometryComparison:
    """Exactness diagnostics for two rectangle lists."""

    translation_x: float
    translation_y: float
    max_coordinate_delta: float
    max_dimension_delta: float

    def is_exact(self, tolerance: float = EPSILON) -> bool:
        return (
            self.max_coordinate_delta <= tolerance
            and self.max_dimension_delta <= tolerance
        )


def _scalar(value: Any) -> float:
    if hasattr(value, "item"):
        value = value.item()
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DualParentError("non_numeric_value", repr(value)) from exc
    if not math.isfinite(result):
        raise DualParentError("nonfinite_value", repr(value))
    return result


def _rows(value: Any) -> list[list[float]]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    rows: list[list[float]] = []
    for row in value:
        if not isinstance(row, (list, tuple)):
            row = [row]
        rows.append([_scalar(item) for item in row])
    return rows


def _values(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    return [_scalar(item) for item in value]


def _integer(value: float, *, code: str, tolerance: float = EPSILON) -> int:
    rounded = round(value)
    if abs(value - rounded) > tolerance:
        raise DualParentError(code, f"expected integer, got {value!r}")
    return int(rounded)


def enumerate_oriented_factor_shapes(
    area: Any,
    *,
    max_aspect_ratio: float = 3.0,
    tolerance: float = EPSILON,
) -> tuple[Shape, ...]:
    """Enumerate every legal integer ``(width, height)`` with exact area.

    Orientations are distinct.  Results are deterministic: near-square shapes
    come first, followed by increasing width.  The function returns an empty
    tuple when the target is not a positive integer or has no aspect-legal
    factor pair.
    """

    numeric_area = _scalar(area)
    rounded_area = round(numeric_area)
    if numeric_area <= 0 or abs(numeric_area - rounded_area) > tolerance:
        return ()
    integer_area = int(rounded_area)
    shapes: set[Shape] = set()
    for width in range(1, math.isqrt(integer_area) + 1):
        if integer_area % width:
            continue
        height = integer_area // width
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


def training_rectangles(fp_solution: Any, block_count: int | None = None) -> list[Rect]:
    """Convert FloorSet training-label rows ``(w,h,x,y)`` to rectangles."""

    rows = _rows(fp_solution)
    count = len(rows) if block_count is None else block_count
    if count < 1 or len(rows) < count:
        raise DualParentError(
            "malformed_golden_geometry", f"need {count} rows, found {len(rows)}"
        )
    rectangles: list[Rect] = []
    for index, row in enumerate(rows[:count]):
        if len(row) < 4:
            raise DualParentError(
                "malformed_golden_geometry", f"block {index} has {len(row)} fields"
            )
        width, height, x, y = row[:4]
        if width <= 0 or height <= 0:
            raise DualParentError(
                "invalid_golden_shape", f"block {index}: {(width, height)!r}"
            )
        rectangles.append((x, y, width, height))
    return rectangles


def hard_targets_from_golden(
    constraints: Any, golden_rectangles: Sequence[Rect]
) -> list[Rect]:
    """Build the optimizer's fixed/preplaced target rows for an oracle audit."""

    rows = _rows(constraints)
    targets: list[Rect] = [(-1.0, -1.0, -1.0, -1.0) for _ in golden_rectangles]
    for index, rectangle in enumerate(golden_rectangles):
        row = rows[index] if index < len(rows) else []
        fixed = len(row) > 0 and row[0] != 0.0
        preplaced = len(row) > 1 and row[1] != 0.0
        x, y, width, height = rectangle
        if preplaced:
            targets[index] = rectangle
        elif fixed:
            targets[index] = (-1.0, -1.0, width, height)
    return targets


def _constraint_groups(
    constraint_rows: Sequence[Sequence[float]], column: int, block_count: int
) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for index in range(block_count):
        row = constraint_rows[index] if index < len(constraint_rows) else ()
        group = int(row[column]) if len(row) > column else 0
        if group > 0:
            groups.setdefault(group, []).append(index)
    return groups


def _parse_horizontal_tree(
    tree: Any, block_count: int
) -> tuple[int, tuple[HorizontalRelation, ...]]:
    rows = _rows(tree)
    expected_edges = max(0, block_count - 1)
    if len(rows) != expected_edges:
        raise DualParentError(
            "horizontal_edge_count",
            f"expected {expected_edges}, found {len(rows)}",
        )
    if block_count == 1:
        return 0, ()

    child_parent: dict[int, int] = {}
    occupied_slots: set[tuple[int, int]] = set()
    relations: list[HorizontalRelation] = []
    adjacency: list[list[int]] = [[] for _ in range(block_count)]
    for edge_index, row in enumerate(rows):
        if len(row) < 3:
            raise DualParentError(
                "malformed_horizontal_edge", f"edge {edge_index} has {len(row)} fields"
            )
        parent = _integer(row[0], code="noninteger_horizontal_parent")
        child = _integer(row[1], code="noninteger_horizontal_child")
        side = _integer(row[2], code="noninteger_horizontal_side")
        if not (0 <= parent < block_count and 0 <= child < block_count):
            raise DualParentError(
                "horizontal_index_out_of_range", f"edge {(parent, child, side)!r}"
            )
        if parent == child:
            raise DualParentError("horizontal_self_cycle", f"block {parent}")
        if side not in (0, 1):
            raise DualParentError("invalid_horizontal_side", str(side))
        if child in child_parent:
            raise DualParentError("multiple_horizontal_parents", f"block {child}")
        if (parent, side) in occupied_slots:
            raise DualParentError(
                "duplicate_bstar_child_slot", f"parent {parent}, side {side}"
            )
        child_parent[child] = parent
        occupied_slots.add((parent, side))
        adjacency[parent].append(child)
        relations.append(HorizontalRelation(parent, child, side))

    roots = sorted(set(range(block_count)) - set(child_parent))
    if len(roots) != 1:
        raise DualParentError("horizontal_root_count", repr(roots))
    root = roots[0]
    visiting: set[int] = set()
    visited: set[int] = set()

    def visit(block: int) -> None:
        if block in visiting:
            raise DualParentError("horizontal_cycle", f"block {block}")
        if block in visited:
            return
        visiting.add(block)
        for child in adjacency[block]:
            visit(child)
        visiting.remove(block)
        visited.add(block)

    visit(root)
    if len(visited) != block_count:
        missing = sorted(set(range(block_count)) - visited)
        raise DualParentError("horizontal_disconnected", repr(missing))
    return root, tuple(relations)


def _interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return min(a1, b1) - max(a0, b0)


def extract_oracle_labels(
    area_targets: Any,
    constraints: Any,
    tree: Any,
    golden_rectangles: Sequence[Rect],
    *,
    max_aspect_ratio: float = 3.0,
    tolerance: float = EPSILON,
) -> DualParentLabels:
    """Extract structural supervision from a labeled floorplan.

    This is intentionally a label-analysis function, not an inference path.
    It verifies that every free shape belongs to the complete exact factor-pair
    set and that every horizontal/vertical relation is geometrically true.
    """

    rectangles = [tuple(map(_scalar, rectangle)) for rectangle in golden_rectangles]
    block_count = len(rectangles)
    if block_count < 1:
        raise DualParentError("empty_layout")
    areas = _values(area_targets)
    if len(areas) < block_count:
        raise DualParentError(
            "area_count", f"need {block_count}, found {len(areas)}"
        )
    constraint_rows = _rows(constraints)
    if constraint_rows and len(constraint_rows) < block_count:
        raise DualParentError(
            "constraint_count", f"need {block_count}, found {len(constraint_rows)}"
        )

    dimensions: list[Shape] = []
    all_options: list[tuple[Shape, ...]] = []
    selected_indices: list[int] = []
    for index, rectangle in enumerate(rectangles):
        _x, _y, width, height = rectangle
        if width <= 0 or height <= 0:
            raise DualParentError(
                "invalid_golden_shape", f"block {index}: {(width, height)!r}"
            )
        row = constraint_rows[index] if index < len(constraint_rows) else []
        hard = (len(row) > 0 and row[0] != 0.0) or (
            len(row) > 1 and row[1] != 0.0
        )
        dimensions.append((width, height))
        if hard:
            options = ((width, height),)
            selected = 0
        else:
            options = enumerate_oriented_factor_shapes(
                areas[index],
                max_aspect_ratio=max_aspect_ratio,
                tolerance=tolerance,
            )
            if not options:
                raise DualParentError(
                    "soft_area_has_no_exact_factor_shape",
                    f"block {index}, area {areas[index]!r}",
                )
            matches = [
                option_index
                for option_index, (option_width, option_height) in enumerate(options)
                if abs(width - option_width) <= tolerance
                and abs(height - option_height) <= tolerance
            ]
            if len(matches) != 1:
                raise DualParentError(
                    "soft_shape_not_exact_factor",
                    f"block {index}, area {areas[index]!r}, shape {(width, height)!r}",
                )
            selected = matches[0]
        all_options.append(options)
        selected_indices.append(selected)

    mib_inconsistent: list[int] = []
    for group, members in _constraint_groups(
        constraint_rows, 2, block_count
    ).items():
        first = dimensions[members[0]]
        if any(
            abs(dimensions[index][0] - first[0]) > tolerance
            or abs(dimensions[index][1] - first[1]) > tolerance
            for index in members[1:]
        ):
            mib_inconsistent.append(group)

    root, horizontal = _parse_horizontal_tree(tree, block_count)
    for relation in horizontal:
        parent_x, _parent_y, parent_width, _parent_height = rectangles[
            relation.parent
        ]
        child_x = rectangles[relation.child][0]
        expected_x = parent_x + (parent_width if relation.side == 0 else 0.0)
        if abs(child_x - expected_x) > tolerance:
            raise DualParentError(
                "horizontal_relation_mismatch",
                f"edge {relation}, expected x={expected_x}, got {child_x}",
            )

    floor_y = min(rectangle[1] for rectangle in rectangles)
    supports: list[int | None] = []
    for child, (x, y, width, _height) in enumerate(rectangles):
        if abs(y - floor_y) <= tolerance:
            supports.append(None)
            continue
        candidates: list[tuple[float, float, int]] = []
        for parent, (px, py, pwidth, pheight) in enumerate(rectangles):
            if parent == child or abs(py + pheight - y) > tolerance:
                continue
            overlap = _interval_overlap(x, x + width, px, px + pwidth)
            if overlap > tolerance:
                child_center = x + width / 2.0
                parent_center = px + pwidth / 2.0
                candidates.append((-overlap, abs(child_center - parent_center), parent))
        if not candidates:
            raise DualParentError(
                "unsupported_above_floor", f"block {child}, y={y}, floor={floor_y}"
            )
        supports.append(min(candidates)[2])

    return DualParentLabels(
        root=root,
        dimensions=tuple(dimensions),
        shape_options=tuple(all_options),
        selected_shape_indices=tuple(selected_indices),
        horizontal=horizontal,
        vertical_supports=tuple(supports),
        mib_inconsistent_groups=tuple(mib_inconsistent),
    )


def _preplaced_shift(
    positions: Sequence[Rect],
    constraint_rows: Sequence[Sequence[float]],
    hard_targets: Sequence[Sequence[float]],
    tolerance: float,
) -> tuple[float, float] | None:
    shifts: list[tuple[float, float, int]] = []
    for index, position in enumerate(positions):
        row = constraint_rows[index] if index < len(constraint_rows) else ()
        preplaced = len(row) > 1 and row[1] != 0.0
        if not preplaced:
            continue
        if index >= len(hard_targets) or len(hard_targets[index]) < 4:
            raise DualParentError("missing_preplaced_target", f"block {index}")
        target = [_scalar(value) for value in hard_targets[index][:4]]
        if target[0] < 0 or target[1] < 0:
            raise DualParentError("missing_preplaced_target", f"block {index}")
        shifts.append((target[0] - position[0], target[1] - position[1], index))
    if not shifts:
        return None
    dx, dy, _index = shifts[0]
    for other_dx, other_dy, index in shifts[1:]:
        if abs(other_dx - dx) > tolerance or abs(other_dy - dy) > tolerance:
            raise DualParentError(
                "preplaced_anchor_conflict",
                f"block {index} implies shift {(other_dx, other_dy)}, expected {(dx, dy)}",
            )
    return dx, dy


def decode_dual_parent(
    labels: DualParentLabels,
    *,
    constraints: Any = None,
    hard_targets: Sequence[Sequence[float]] | None = None,
    origin: tuple[float, float] = (0.0, 0.0),
    enforce_mib: bool = True,
    tolerance: float = EPSILON,
) -> list[Rect]:
    """Decode dimensions plus horizontal/vertical parents into rectangles.

    Preplaced targets, when supplied, may determine one global translation.
    They must all agree on that translation.  Fixed/preplaced dimensions and
    MIB uniformity are then checked before returning the placement.
    """

    block_count = len(labels.dimensions)
    if block_count < 1:
        raise DualParentError("empty_layout")
    if (
        len(labels.shape_options) != block_count
        or len(labels.selected_shape_indices) != block_count
    ):
        raise DualParentError("shape_category_count")
    if len(labels.vertical_supports) != block_count:
        raise DualParentError(
            "vertical_support_count",
            f"need {block_count}, found {len(labels.vertical_supports)}",
        )
    if not 0 <= labels.root < block_count:
        raise DualParentError("horizontal_root_out_of_range", str(labels.root))
    for index, (width, height) in enumerate(labels.dimensions):
        if not all(math.isfinite(value) and value > 0 for value in (width, height)):
            raise DualParentError(
                "invalid_decoded_shape", f"block {index}: {(width, height)!r}"
            )
        selected = labels.selected_shape_indices[index]
        options = labels.shape_options[index]
        if not 0 <= selected < len(options):
            raise DualParentError(
                "selected_shape_out_of_range", f"block {index}: {selected}"
            )
        selected_width, selected_height = options[selected]
        if (
            abs(width - selected_width) > tolerance
            or abs(height - selected_height) > tolerance
        ):
            raise DualParentError(
                "selected_shape_dimension_mismatch", f"block {index}"
            )

    horizontal_children: list[list[HorizontalRelation]] = [
        [] for _ in range(block_count)
    ]
    horizontal_parent_count = [0] * block_count
    occupied_slots: set[tuple[int, int]] = set()
    for relation in labels.horizontal:
        if not (
            0 <= relation.parent < block_count
            and 0 <= relation.child < block_count
            and relation.side in (0, 1)
        ):
            raise DualParentError("invalid_horizontal_relation", repr(relation))
        slot = (relation.parent, relation.side)
        if slot in occupied_slots:
            raise DualParentError("duplicate_bstar_child_slot", repr(slot))
        occupied_slots.add(slot)
        horizontal_children[relation.parent].append(relation)
        horizontal_parent_count[relation.child] += 1
    if len(labels.horizontal) != block_count - 1:
        raise DualParentError("horizontal_edge_count", str(len(labels.horizontal)))
    if horizontal_parent_count[labels.root] != 0 or any(
        horizontal_parent_count[index] != 1
        for index in range(block_count)
        if index != labels.root
    ):
        raise DualParentError("invalid_horizontal_parent_counts")

    x_values: list[float | None] = [None] * block_count
    x_values[labels.root] = _scalar(origin[0])
    stack = [labels.root]
    while stack:
        parent = stack.pop()
        parent_x = x_values[parent]
        assert parent_x is not None
        parent_width = labels.dimensions[parent][0]
        for relation in horizontal_children[parent]:
            if x_values[relation.child] is not None:
                raise DualParentError("horizontal_cycle", f"block {relation.child}")
            x_values[relation.child] = parent_x + (
                parent_width if relation.side == 0 else 0.0
            )
            stack.append(relation.child)
    if any(value is None for value in x_values):
        raise DualParentError("horizontal_disconnected")

    y_values: list[float | None] = [None] * block_count
    visiting: set[int] = set()

    def resolve_y(block: int) -> float:
        known = y_values[block]
        if known is not None:
            return known
        if block in visiting:
            raise DualParentError("vertical_support_cycle", f"block {block}")
        visiting.add(block)
        support = labels.vertical_supports[block]
        if support is None:
            value = _scalar(origin[1])
        else:
            if not 0 <= support < block_count or support == block:
                raise DualParentError(
                    "invalid_vertical_support", f"block {block}: {support!r}"
                )
            value = resolve_y(support) + labels.dimensions[support][1]
        visiting.remove(block)
        y_values[block] = value
        return value

    for index in range(block_count):
        resolve_y(index)

    positions: list[Rect] = [
        (
            float(x_values[index]),
            float(y_values[index]),
            labels.dimensions[index][0],
            labels.dimensions[index][1],
        )
        for index in range(block_count)
    ]
    constraint_rows = _rows(constraints)
    target_rows = list(hard_targets or [])
    anchor = _preplaced_shift(
        positions, constraint_rows, target_rows, tolerance
    ) if hard_targets is not None else None
    if anchor is not None:
        dx, dy = anchor
        positions = [
            (x + dx, y + dy, width, height)
            for x, y, width, height in positions
        ]

    if constraint_rows:
        if len(constraint_rows) < block_count:
            raise DualParentError("constraint_count")
        for index, position in enumerate(positions):
            row = constraint_rows[index]
            fixed = len(row) > 0 and row[0] != 0.0
            preplaced = len(row) > 1 and row[1] != 0.0
            if not (fixed or preplaced):
                continue
            if index >= len(target_rows) or len(target_rows[index]) < 4:
                raise DualParentError("missing_hard_target", f"block {index}")
            target = [_scalar(value) for value in target_rows[index][:4]]
            if abs(position[2] - target[2]) > tolerance or abs(
                position[3] - target[3]
            ) > tolerance:
                raise DualParentError("hard_dimension_mismatch", f"block {index}")
            if preplaced and (
                abs(position[0] - target[0]) > tolerance
                or abs(position[1] - target[1]) > tolerance
            ):
                raise DualParentError("preplaced_position_mismatch", f"block {index}")

        if enforce_mib:
            for group, members in _constraint_groups(
                constraint_rows, 2, block_count
            ).items():
                first = positions[members[0]][2:4]
                if any(
                    abs(positions[index][2] - first[0]) > tolerance
                    or abs(positions[index][3] - first[1]) > tolerance
                    for index in members[1:]
                ):
                    raise DualParentError("mib_shape_mismatch", f"group {group}")

    return positions


def compare_geometry(
    actual: Sequence[Rect],
    expected: Sequence[Rect],
    *,
    allow_global_translation: bool = False,
) -> GeometryComparison:
    """Return exact coordinate/dimension deltas for two placements."""

    if len(actual) != len(expected):
        raise DualParentError(
            "geometry_count_mismatch", f"{len(actual)} != {len(expected)}"
        )
    if not actual:
        return GeometryComparison(0.0, 0.0, 0.0, 0.0)
    translation_x = (
        _scalar(expected[0][0]) - _scalar(actual[0][0])
        if allow_global_translation
        else 0.0
    )
    translation_y = (
        _scalar(expected[0][1]) - _scalar(actual[0][1])
        if allow_global_translation
        else 0.0
    )
    max_coordinate_delta = 0.0
    max_dimension_delta = 0.0
    for actual_rect, expected_rect in zip(actual, expected):
        ax, ay, aw, ah = map(_scalar, actual_rect)
        ex, ey, ew, eh = map(_scalar, expected_rect)
        max_coordinate_delta = max(
            max_coordinate_delta,
            abs(ax + translation_x - ex),
            abs(ay + translation_y - ey),
        )
        max_dimension_delta = max(
            max_dimension_delta, abs(aw - ew), abs(ah - eh)
        )
    return GeometryComparison(
        translation_x,
        translation_y,
        max_coordinate_delta,
        max_dimension_delta,
    )
