"""Pure-stdlib inference for the first learned dual-parent baseline.

The model consumes only the 60 inference-visible, permutation-equivariant
features from :mod:`learned_order`.  Supervised FloorSet labels are used by the
trainer, never by this module.  Predicted local scores are projected into a
valid binary B*-tree and an acyclic vertical-support forest before the exact
dual-parent decoder is called.

This is intentionally a conservative research interface: malformed artifacts,
low confidence, decoding failures, and hard-infeasible placements return
``None`` so a caller can retain the proven v32 placement.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence

try:  # package import in tests
    from .dual_parent_decoder import (
        DualParentError,
        DualParentLabels,
        HorizontalRelation,
        decode_dual_parent,
        enumerate_oriented_factor_shapes,
    )
    from .learned_order import FEATURE_NAMES, FEATURE_VERSION, extract_order_features
except ImportError:  # flat import in the packaged solver
    from dual_parent_decoder import (  # type: ignore
        DualParentError,
        DualParentLabels,
        HorizontalRelation,
        decode_dual_parent,
        enumerate_oriented_factor_shapes,
    )
    from learned_order import (  # type: ignore
        FEATURE_NAMES,
        FEATURE_VERSION,
        extract_order_features,
    )


ARTIFACT_SCHEMA_VERSION = 1
MODEL_TYPE = "dual_parent_linear_projection_v1"
MAX_SHAPE_OPTIONS = 8
PAIR_HEAD_NAMES = (
    "horizontal_side_0",
    "horizontal_side_1",
    "vertical_support",
)
PAIR_DIRECT_FEATURE_NAMES = (
    "log_b2b_weight_norm",
    "has_b2b_edge",
    "same_mib_group",
    "same_cluster_group",
    "log_area_ratio",
    "abs_log_area_ratio",
    "boundary_bit_compatibility",
    "common_pin_similarity",
    "preplaced_anchor_similarity",
)
NODE_TARGET_NAMES = (
    "golden_center_x_rank",
    "golden_center_y_rank",
    *(f"shape_option_{index}" for index in range(MAX_SHAPE_OPTIONS)),
    "horizontal_root",
    "vertical_floor",
)


@dataclass(frozen=True)
class StructuredPrediction:
    """A candidate and stable diagnostics suitable for an experiment ledger."""

    positions: tuple[tuple[float, float, float, float], ...] | None
    confidence: float
    reason: str
    hard_feasible: bool
    root: int | None = None
    shape_correctness_proxy: float = 0.0
    horizontal_margin: float = 0.0
    vertical_margin: float = 0.0
    vertical_mode: str = "skyline"
    labels: DualParentLabels | None = None


@dataclass(frozen=True)
class PairFeatureContext:
    """Inference-visible pair relations; group IDs are equality keys only."""

    areas: tuple[float, ...]
    b2b_weights: dict[tuple[int, int], float]
    b2b_log_scale: float
    mib_groups: tuple[int, ...]
    cluster_groups: tuple[int, ...]
    boundary_codes: tuple[int, ...]
    common_pin: dict[tuple[int, int], float]
    pin_load: tuple[float, ...]
    anchor_vectors: tuple[tuple[float, ...], ...]


def _number(value: Any, name: str) -> float:
    if hasattr(value, "item"):
        value = value.item()
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must contain finite values")
    return result


def _rows(value: Any) -> list[list[float]]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError("matrix inputs must contain rows")
    result = []
    for row in value:
        if not isinstance(row, (list, tuple)):
            row = [row]
        result.append([_number(item, "matrix") for item in row])
    return result


def _values(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    return [_number(item, "vector") for item in value]


def _canonical_payload_sha256(model: dict[str, Any]) -> str:
    payload = dict(model)
    payload.pop("payload_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def seal_artifact(model: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy carrying its canonical payload digest."""

    sealed = dict(model)
    sealed.pop("payload_sha256", None)
    sealed["payload_sha256"] = _canonical_payload_sha256(sealed)
    return sealed


def _numeric_matrix(
    value: Any, rows: int, columns: int, name: str
) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, (list, tuple)) or len(value) != rows:
        raise ValueError(f"{name} must have {rows} rows")
    matrix = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != columns:
            raise ValueError(f"{name} must have {columns} columns")
        matrix.append(tuple(_number(item, name) for item in row))
    return tuple(matrix)


def validate_artifact(model: Any) -> dict[str, Any]:
    """Fail closed on schema drift, non-finite values, or hash mismatch."""

    if not isinstance(model, dict):
        raise ValueError("structured model artifact must be a JSON object")
    if model.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("structured model schema_version is unsupported")
    if model.get("model_type") != MODEL_TYPE:
        raise ValueError("structured model model_type is unsupported")
    feature_schema = model.get("feature_schema")
    if not isinstance(feature_schema, dict):
        raise ValueError("structured model feature_schema must be an object")
    if feature_schema.get("version") != FEATURE_VERSION:
        raise ValueError("structured model feature version is unsupported")
    if feature_schema.get("names") != list(FEATURE_NAMES):
        raise ValueError("structured model feature names do not match inference")
    message_steps = feature_schema.get("message_steps")
    if not isinstance(message_steps, int) or not 0 <= message_steps <= 16:
        raise ValueError("structured model message_steps is invalid")

    structured = model.get("structured_schema")
    if not isinstance(structured, dict):
        raise ValueError("structured model structured_schema must be an object")
    if structured.get("node_targets") != list(NODE_TARGET_NAMES):
        raise ValueError("structured model node targets are unsupported")
    if structured.get("pair_heads") != list(PAIR_HEAD_NAMES):
        raise ValueError("structured model pair heads are unsupported")
    if structured.get("pair_direct_features") != list(PAIR_DIRECT_FEATURE_NAMES):
        raise ValueError("structured model direct pair features are unsupported")
    hidden_size = structured.get("hidden_size")
    if not isinstance(hidden_size, int) or not 1 <= hidden_size <= 32:
        raise ValueError("structured model hidden_size is invalid")
    pair_feature_count = 1 + 5 * hidden_size + len(PAIR_DIRECT_FEATURE_NAMES)
    if structured.get("pair_feature_count") != pair_feature_count:
        raise ValueError("structured model pair_feature_count is invalid")
    if structured.get("max_shape_options") != MAX_SHAPE_OPTIONS:
        raise ValueError("structured model shape option count is unsupported")

    normalization = model.get("normalization")
    if not isinstance(normalization, dict):
        raise ValueError("structured model normalization must be an object")
    center = normalization.get("center")
    scale = normalization.get("scale")
    if not isinstance(center, (list, tuple)) or len(center) != len(FEATURE_NAMES):
        raise ValueError("structured model center does not match features")
    if not isinstance(scale, (list, tuple)) or len(scale) != len(FEATURE_NAMES):
        raise ValueError("structured model scale does not match features")
    numeric_center = tuple(_number(item, "normalization center") for item in center)
    numeric_scale = tuple(_number(item, "normalization scale") for item in scale)
    if any(item <= 0.0 for item in numeric_scale):
        raise ValueError("structured model scale must be positive")

    projection = _numeric_matrix(
        model.get("hidden_projection"),
        len(FEATURE_NAMES),
        hidden_size,
        "hidden_projection",
    )
    node_coefficients = _numeric_matrix(
        model.get("node_coefficients"),
        len(FEATURE_NAMES),
        len(NODE_TARGET_NAMES),
        "node_coefficients",
    )
    pair_coefficients = _numeric_matrix(
        model.get("pair_coefficients"),
        pair_feature_count,
        len(PAIR_HEAD_NAMES),
        "pair_coefficients",
    )
    calibration = model.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("structured model calibration must be an object")
    confidence_threshold = _number(
        calibration.get("confidence_threshold"), "confidence_threshold"
    )
    margin_scale = _number(calibration.get("margin_scale"), "margin_scale")
    margin_bias = _number(calibration.get("margin_bias"), "margin_bias")
    if not 0.0 <= confidence_threshold <= 1.0 or margin_scale <= 0.0:
        raise ValueError("structured model calibration values are invalid")

    expected = model.get("payload_sha256")
    if not (
        isinstance(expected, str)
        and len(expected) == 64
        and all(character in "0123456789abcdef" for character in expected)
    ):
        raise ValueError("structured model payload_sha256 is invalid")
    actual = _canonical_payload_sha256(model)
    if not hmac.compare_digest(expected, actual):
        raise ValueError("structured model payload_sha256 integrity check failed")

    validated = dict(model)
    validated["_validated"] = {
        "center": numeric_center,
        "scale": numeric_scale,
        "projection": projection,
        "node_coefficients": node_coefficients,
        "pair_coefficients": pair_coefficients,
    }
    return validated


def load_artifact(path: str | Path) -> dict[str, Any]:
    return validate_artifact(json.loads(Path(path).read_text(encoding="utf-8")))


def _matmul(rows: Sequence[Sequence[float]], weights: Sequence[Sequence[float]]):
    output_count = len(weights[0])
    return [
        [
            sum(row[index] * weights[index][output] for index in range(len(weights)))
            for output in range(output_count)
        ]
        for row in rows
    ]


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    denominator = max(1, len(values) - 1)
    result = [0.0] * len(values)
    for ordinal, index in enumerate(order):
        result[index] = ordinal / denominator
    return result


def _margin(scores: Sequence[float]) -> float:
    if len(scores) < 2:
        return 20.0
    first, second = sorted(scores, reverse=True)[:2]
    return first - second


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        term = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + term)
    term = math.exp(max(value, -700.0))
    return term / (1.0 + term)


def _pair_features(
    child: Sequence[float],
    parent: Sequence[float],
    direct: Sequence[float] = (),
) -> list[float]:
    if len(direct) not in (0, len(PAIR_DIRECT_FEATURE_NAMES)):
        raise ValueError("direct pair feature count is invalid")
    difference = [left - right for left, right in zip(child, parent)]
    return [
        1.0,
        *child,
        *parent,
        *difference,
        *(abs(value) for value in difference),
        *(left * right for left, right in zip(child, parent)),
        *direct,
    ]


def extract_pair_feature_context(
    block_count: int,
    area_targets: Any,
    b2b_connectivity: Any,
    p2b_connectivity: Any,
    constraints: Any,
    node_features: Sequence[Sequence[float]],
) -> PairFeatureContext:
    """Precompute sparse pair relations without identity or label inputs."""

    areas = tuple(_values(area_targets)[:block_count])
    if len(areas) != block_count or any(area <= 0.0 for area in areas):
        raise ValueError("pair context requires positive block areas")
    b2b_weights: dict[tuple[int, int], float] = {}
    max_b2b = 0.0
    for row in _rows(b2b_connectivity):
        if len(row) < 3 or row[2] <= 0.0:
            continue
        left, right = int(row[0]), int(row[1])
        if not (0 <= left < block_count and 0 <= right < block_count) or left == right:
            continue
        key = (min(left, right), max(left, right))
        b2b_weights[key] = b2b_weights.get(key, 0.0) + row[2]
        max_b2b = max(max_b2b, b2b_weights[key])

    constraint_rows = _rows(constraints)
    mib_groups = []
    cluster_groups = []
    boundary_codes = []
    for index in range(block_count):
        row = constraint_rows[index] if index < len(constraint_rows) else []
        mib_groups.append(int(row[2]) if len(row) > 2 else 0)
        cluster_groups.append(int(row[3]) if len(row) > 3 else 0)
        boundary_codes.append(int(row[4]) if len(row) > 4 else 0)

    pins: dict[int, list[tuple[int, float]]] = {}
    pin_load = [0.0] * block_count
    for row in _rows(p2b_connectivity):
        if len(row) < 3 or row[2] <= 0.0:
            continue
        pin, block = int(row[0]), int(row[1])
        if pin < 0 or not 0 <= block < block_count:
            continue
        pins.setdefault(pin, []).append((block, row[2]))
        pin_load[block] += row[2]
    common_pin: dict[tuple[int, int], float] = {}
    for connected in pins.values():
        for left_index in range(len(connected)):
            left, left_weight = connected[left_index]
            for right_index in range(left_index + 1, len(connected)):
                right, right_weight = connected[right_index]
                if left == right:
                    continue
                key = (min(left, right), max(left, right))
                common_pin[key] = common_pin.get(key, 0.0) + min(
                    left_weight, right_weight
                )

    anchor_names = (
        "connected_preplaced_degree_fraction",
        "connected_preplaced_dx_norm",
        "connected_preplaced_dy_norm",
        "nearest_obstacle_gap_x_norm",
        "nearest_obstacle_gap_y_norm",
        "has_other_preplaced_obstacle",
    )
    anchor_indices = [FEATURE_NAMES.index(name) for name in anchor_names]
    anchor_vectors = tuple(
        tuple(float(row[index]) for index in anchor_indices)
        for row in node_features
    )
    return PairFeatureContext(
        areas=areas,
        b2b_weights=b2b_weights,
        b2b_log_scale=max(math.log1p(max_b2b), 1e-12),
        mib_groups=tuple(mib_groups),
        cluster_groups=tuple(cluster_groups),
        boundary_codes=tuple(boundary_codes),
        common_pin=common_pin,
        pin_load=tuple(pin_load),
        anchor_vectors=anchor_vectors,
    )


def direct_pair_features(
    context: PairFeatureContext, child: int, parent: int
) -> tuple[float, ...]:
    key = (min(child, parent), max(child, parent))
    b2b_weight = context.b2b_weights.get(key, 0.0)
    mib_child = context.mib_groups[child]
    mib_parent = context.mib_groups[parent]
    cluster_child = context.cluster_groups[child]
    cluster_parent = context.cluster_groups[parent]
    ratio = math.log(context.areas[child] / context.areas[parent])
    child_boundary = context.boundary_codes[child]
    parent_boundary = context.boundary_codes[parent]
    union = child_boundary | parent_boundary
    boundary_compatibility = (
        (child_boundary & parent_boundary).bit_count() / union.bit_count()
        if union
        else 0.0
    )
    common = context.common_pin.get(key, 0.0)
    denominator = math.sqrt(
        max(context.pin_load[child] * context.pin_load[parent], 0.0)
    )
    common_similarity = common / denominator if denominator > 0.0 else 0.0
    child_anchor = context.anchor_vectors[child]
    parent_anchor = context.anchor_vectors[parent]
    both_anchored = child_anchor[-1] > 0.0 and parent_anchor[-1] > 0.0
    anchor_similarity = (
        math.exp(
            -4.0
            * sum(
                abs(left - right)
                for left, right in zip(child_anchor[:-1], parent_anchor[:-1])
            )
        )
        if both_anchored
        else 0.0
    )
    return (
        math.log1p(b2b_weight) / context.b2b_log_scale if b2b_weight > 0.0 else 0.0,
        float(b2b_weight > 0.0),
        float(mib_child > 0 and mib_child == mib_parent),
        float(cluster_child > 0 and cluster_child == cluster_parent),
        ratio,
        abs(ratio),
        boundary_compatibility,
        common_similarity,
        anchor_similarity,
    )


def _hard_targets(target_positions: Any, block_count: int) -> list[list[float]]:
    rows = _rows(target_positions)
    result = [[-1.0, -1.0, -1.0, -1.0] for _ in range(block_count)]
    for index, row in enumerate(rows[:block_count]):
        if len(row) >= 4:
            result[index] = row[:4]
    return result


def _shape_options_and_indices(
    area_targets: Any,
    constraints: Any,
    target_positions: Any,
    node_scores: Sequence[Sequence[float]],
) -> tuple[list[tuple[tuple[float, float], ...]], list[int], list[tuple[float, float]], float]:
    areas = _values(area_targets)
    constraint_rows = _rows(constraints)
    block_count = len(node_scores)
    targets = _hard_targets(target_positions, block_count)
    options: list[tuple[tuple[float, float], ...]] = []
    for index in range(block_count):
        row = constraint_rows[index] if index < len(constraint_rows) else []
        hard = (len(row) > 0 and row[0] != 0.0) or (
            len(row) > 1 and row[1] != 0.0
        )
        if hard:
            width, height = targets[index][2:4]
            if width <= 0.0 or height <= 0.0:
                raise DualParentError("missing_hard_target", f"block {index}")
            block_options = ((width, height),)
        else:
            block_options = enumerate_oriented_factor_shapes(areas[index])
            if not block_options or len(block_options) > MAX_SHAPE_OPTIONS:
                raise DualParentError("unsupported_shape_option_count", f"block {index}")
        options.append(block_options)

    groups: dict[int, list[int]] = {}
    for index in range(block_count):
        row = constraint_rows[index] if index < len(constraint_rows) else []
        group = int(row[2]) if len(row) > 2 else 0
        if group > 0:
            groups.setdefault(group, []).append(index)

    selected = [-1] * block_count
    shape_margins: list[float] = []
    grouped = set()
    for members in groups.values():
        grouped.update(members)
        common = set(options[members[0]])
        for member in members[1:]:
            common &= set(options[member])
        if not common:
            raise DualParentError("mib_shape_option_intersection_empty")
        candidates = []
        for shape in sorted(common):
            score = 0.0
            indices = []
            for member in members:
                option_index = options[member].index(shape)
                indices.append(option_index)
                score += node_scores[member][2 + option_index]
            candidates.append((score / len(members), shape, indices))
        best = max(candidates, key=lambda row: (row[0], row[1]))
        shape_margins.append(_margin([row[0] for row in candidates]))
        for member, option_index in zip(members, best[2]):
            selected[member] = option_index

    for index in range(block_count):
        if index in grouped:
            continue
        scores = [node_scores[index][2 + option] for option in range(len(options[index]))]
        selected[index] = max(range(len(scores)), key=lambda option: (scores[option], -option))
        shape_margins.append(_margin(scores))
    dimensions = [options[index][selected[index]] for index in range(block_count)]
    return options, selected, dimensions, min(shape_margins, default=20.0)


def _project_horizontal(
    node_scores: Sequence[Sequence[float]],
    pair_scores: Sequence[Sequence[Sequence[float]]],
) -> tuple[int, tuple[HorizontalRelation, ...], float]:
    count = len(node_scores)
    root_output = len(NODE_TARGET_NAMES) - 2
    root = max(range(count), key=lambda index: (node_scores[index][root_output], -index))
    assigned = {root}
    remaining = set(range(count)) - assigned
    slots = {(root, 0), (root, 1)}
    relations = []
    margins = []
    while remaining:
        ranked = []
        for parent, side in slots:
            head = side
            for child in remaining:
                ranked.append((pair_scores[child][parent][head], parent, child, side))
        if not ranked:
            raise DualParentError("horizontal_projection_stalled")
        ranked.sort(key=lambda row: (-row[0], row[1], row[2], row[3]))
        score, parent, child, side = ranked[0]
        margins.append(score - ranked[1][0] if len(ranked) > 1 else 20.0)
        relations.append(HorizontalRelation(parent, child, side))
        remaining.remove(child)
        assigned.add(child)
        slots.remove((parent, side))
        slots.add((child, 0))
        slots.add((child, 1))
    return root, tuple(relations), min(margins, default=20.0)


def _horizontal_x(
    root: int,
    relations: Sequence[HorizontalRelation],
    dimensions: Sequence[tuple[float, float]],
) -> tuple[float, ...]:
    children: list[list[HorizontalRelation]] = [[] for _ in dimensions]
    for relation in relations:
        children[relation.parent].append(relation)
    values: list[float | None] = [None] * len(dimensions)
    values[root] = 0.0
    stack = [root]
    while stack:
        parent = stack.pop()
        assert values[parent] is not None
        for relation in children[parent]:
            values[relation.child] = values[parent] + (
                dimensions[parent][0] if relation.side == 0 else 0.0
            )
            stack.append(relation.child)
    if any(value is None for value in values):
        raise DualParentError("horizontal_projection_disconnected")
    return tuple(float(value) for value in values)


def _x_overlaps(
    child: int,
    parent: int,
    x_values: Sequence[float],
    dimensions: Sequence[tuple[float, float]],
    tolerance: float = 1e-6,
) -> bool:
    child_x = x_values[child]
    parent_x = x_values[parent]
    return min(
        child_x + dimensions[child][0], parent_x + dimensions[parent][0]
    ) - max(child_x, parent_x) > tolerance


def _project_vertical_learned(
    node_scores: Sequence[Sequence[float]],
    pair_scores: Sequence[Sequence[Sequence[float]]],
    x_values: Sequence[float],
    dimensions: Sequence[tuple[float, float]],
) -> tuple[tuple[int | None, ...], float]:
    count = len(node_scores)
    y_rank = _rank([row[1] for row in node_scores])
    order = sorted(range(count), key=lambda index: (y_rank[index], index))
    floor_output = len(NODE_TARGET_NAMES) - 1
    supports: list[int | None] = [None] * count
    earlier: list[int] = []
    margins = []
    for child in order:
        choices = [(node_scores[child][floor_output], None)]
        choices.extend(
            (pair_scores[child][parent][2], parent)
            for parent in earlier
            if _x_overlaps(child, parent, x_values, dimensions)
        )
        choices.sort(key=lambda row: (-row[0], -1 if row[1] is None else row[1]))
        supports[child] = choices[0][1]
        margins.append(choices[0][0] - choices[1][0] if len(choices) > 1 else 20.0)
        earlier.append(child)
    return tuple(supports), min(margins, default=20.0)


def _project_vertical_skyline(
    node_scores: Sequence[Sequence[float]],
    pair_scores: Sequence[Sequence[Sequence[float]]],
    x_values: Sequence[float],
    dimensions: Sequence[tuple[float, float]],
) -> tuple[tuple[int | None, ...], float]:
    """Build an acyclic support forest that is overlap-free by construction.

    Predicted y rank chooses insertion order.  A block is put on the highest
    already-placed rectangle whose x interval intersects it, or on the floor
    when none intersects.  The learned support score breaks equal-height ties;
    it never overrides the feasibility-preserving highest-top rule.
    """

    count = len(node_scores)
    order = sorted(range(count), key=lambda index: (node_scores[index][1], index))
    supports: list[int | None] = [None] * count
    y_values = [0.0] * count
    earlier: list[int] = []
    margins = []
    for child in order:
        overlapping = [
            parent
            for parent in earlier
            if _x_overlaps(child, parent, x_values, dimensions)
        ]
        if not overlapping:
            supports[child] = None
            margins.append(20.0)
            earlier.append(child)
            continue
        tops = {
            parent: y_values[parent] + dimensions[parent][1]
            for parent in overlapping
        }
        highest = max(tops.values())
        candidates = [
            parent for parent in overlapping if abs(tops[parent] - highest) <= 1e-6
        ]
        candidates.sort(
            key=lambda parent: (-pair_scores[child][parent][2], parent)
        )
        support = candidates[0]
        supports[child] = support
        y_values[child] = highest
        ranked_scores = sorted(
            (pair_scores[child][parent][2] for parent in candidates), reverse=True
        )
        margins.append(
            ranked_scores[0] - ranked_scores[1] if len(ranked_scores) > 1 else 20.0
        )
        earlier.append(child)
    return tuple(supports), min(margins, default=20.0)


def hard_feasible(
    positions: Sequence[Sequence[float]],
    area_targets: Any,
    constraints: Any,
    target_positions: Any,
    *,
    tolerance: float = 1e-6,
) -> bool:
    """Mirror the official hard checks without needing golden score baselines."""

    count = len(positions)
    areas = _values(area_targets)
    constraint_rows = _rows(constraints)
    targets = _hard_targets(target_positions, count)
    rectangles = []
    for index, row in enumerate(positions):
        if len(row) != 4:
            return False
        x, y, width, height = (_number(item, "positions") for item in row)
        if width <= 0.0 or height <= 0.0:
            return False
        rectangles.append((x, y, width, height))
        constraint = constraint_rows[index] if index < len(constraint_rows) else []
        fixed = len(constraint) > 0 and constraint[0] != 0.0
        preplaced = len(constraint) > 1 and constraint[1] != 0.0
        if fixed or preplaced:
            target = targets[index]
            if target[2] <= 0.0 or target[3] <= 0.0:
                return False
            if abs(width - target[2]) > 1e-4 or abs(height - target[3]) > 1e-4:
                return False
            if preplaced and (
                abs(x - target[0]) > 1e-4 or abs(y - target[1]) > 1e-4
            ):
                return False
        elif index >= len(areas) or areas[index] <= 0.0 or (
            abs(width * height - areas[index]) / areas[index] > 0.01 + tolerance
        ):
            return False
    for left in range(count):
        ax, ay, aw, ah = rectangles[left]
        for right in range(left + 1, count):
            bx, by, bw, bh = rectangles[right]
            if min(ax + aw, bx + bw) - max(ax, bx) > tolerance and (
                min(ay + ah, by + bh) - max(ay, by) > tolerance
            ):
                return False
    return True


def predict_candidate(
    model: dict[str, Any],
    block_count: int,
    area_targets: Any,
    b2b_connectivity: Any,
    p2b_connectivity: Any,
    pins_pos: Any,
    constraints: Any,
    target_positions: Any,
    *,
    vertical_mode: str = "skyline",
) -> StructuredPrediction:
    """Predict, project, decode, and hard-validate one floorplan."""

    labels = None
    try:
        checked = model if "_validated" in model else validate_artifact(model)
        feature_schema = checked["feature_schema"]
        features = extract_order_features(
            block_count,
            area_targets,
            b2b_connectivity,
            p2b_connectivity,
            pins_pos,
            constraints,
            target_positions,
            message_steps=feature_schema["message_steps"],
        )
        validated = checked["_validated"]
        center = validated["center"]
        scale = validated["scale"]
        normalized = [
            [(row[i] - center[i]) / scale[i] for i in range(len(FEATURE_NAMES))]
            for row in features
        ]
        node_scores = _matmul(normalized, validated["node_coefficients"])
        hidden = _matmul(normalized, validated["projection"])
        pair_context = extract_pair_feature_context(
            block_count,
            area_targets,
            b2b_connectivity,
            p2b_connectivity,
            constraints,
            features,
        )
        pair_scores = []
        for child in range(block_count):
            child_scores = []
            for parent in range(block_count):
                if child == parent:
                    child_scores.append([-math.inf] * len(PAIR_HEAD_NAMES))
                else:
                    child_scores.append(
                        _matmul(
                            [
                                _pair_features(
                                    hidden[child],
                                    hidden[parent],
                                    direct_pair_features(pair_context, child, parent),
                                )
                            ],
                            validated["pair_coefficients"],
                        )[0]
                    )
            pair_scores.append(child_scores)

        options, selected, dimensions, shape_margin = _shape_options_and_indices(
            area_targets, constraints, target_positions, node_scores
        )
        root, horizontal, horizontal_margin = _project_horizontal(
            node_scores, pair_scores
        )
        x_values = _horizontal_x(root, horizontal, dimensions)
        if vertical_mode == "skyline":
            vertical, vertical_margin = _project_vertical_skyline(
                node_scores, pair_scores, x_values, dimensions
            )
        elif vertical_mode == "learned":
            vertical, vertical_margin = _project_vertical_learned(
                node_scores, pair_scores, x_values, dimensions
            )
        else:
            raise ValueError("vertical_mode must be 'skyline' or 'learned'")
        labels = DualParentLabels(
            root=root,
            dimensions=tuple(dimensions),
            shape_options=tuple(options),
            selected_shape_indices=tuple(selected),
            horizontal=horizontal,
            vertical_supports=vertical,
        )
        positions = decode_dual_parent(
            labels,
            constraints=constraints,
            hard_targets=_hard_targets(target_positions, block_count),
        )
        feasible = hard_feasible(
            positions, area_targets, constraints, target_positions
        )
        raw_margin = min(shape_margin, horizontal_margin, vertical_margin)
        calibration = checked["calibration"]
        confidence = _sigmoid(
            calibration["margin_scale"] * raw_margin + calibration["margin_bias"]
        )
        return StructuredPrediction(
            positions=tuple(positions) if feasible else None,
            confidence=confidence,
            reason="candidate" if feasible else "hard_infeasible",
            hard_feasible=feasible,
            root=root,
            shape_correctness_proxy=_sigmoid(shape_margin),
            horizontal_margin=horizontal_margin,
            vertical_margin=vertical_margin,
            vertical_mode=vertical_mode,
            labels=labels,
        )
    except (DualParentError, IndexError, KeyError, TypeError, ValueError, OverflowError) as exc:
        code = exc.code if isinstance(exc, DualParentError) else type(exc).__name__
        return StructuredPrediction(
            None, 0.0, f"prediction_error:{code}", False,
            vertical_mode=vertical_mode,
            labels=labels,
        )


def predict_or_fallback(
    model: dict[str, Any],
    fallback_positions: Sequence[Sequence[float]],
    block_count: int,
    area_targets: Any,
    b2b_connectivity: Any,
    p2b_connectivity: Any,
    pins_pos: Any,
    constraints: Any,
    target_positions: Any,
    *,
    validator: Callable[[Sequence[Sequence[float]]], bool] | None = None,
    vertical_mode: str = "skyline",
) -> tuple[list[tuple[float, float, float, float]], StructuredPrediction, bool]:
    """Return a candidate only after confidence and optional caller validation."""

    prediction = predict_candidate(
        model,
        block_count,
        area_targets,
        b2b_connectivity,
        p2b_connectivity,
        pins_pos,
        constraints,
        target_positions,
        vertical_mode=vertical_mode,
    )
    threshold = float(model.get("calibration", {}).get("confidence_threshold", 1.0))
    accepted = (
        prediction.positions is not None
        and prediction.hard_feasible
        and prediction.confidence >= threshold
        and (validator is None or validator(prediction.positions))
    )
    if accepted:
        return list(prediction.positions), prediction, True
    fallback = [tuple(_number(item, "fallback") for item in row) for row in fallback_positions]
    return fallback, prediction, False
