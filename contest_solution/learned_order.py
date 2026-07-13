"""Input-only features for learned FloorSet topology/order guidance.

Training and inference share this pure-stdlib feature path so an experimental
model cannot accidentally consume golden-only information.  Only dimensions
of fixed blocks and the complete rectangles of preplaced blocks are read from
``target_positions``.

The schema is block-permutation equivariant.  MIB and cluster identifiers are
used solely as equality keys: their numeric values never enter a feature.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math


FEATURE_VERSION = 3
MAX_BLOCKS = 120
MAX_MESSAGE_STEPS = 16
MODEL_SCHEMA_VERSION = 1
MODEL_TYPE = "standardized_linear_ridge"
TARGET_NAMES = (
    "golden_center_x_bbox_norm",
    "golden_center_y_bbox_norm",
)

FEATURE_NAMES = (
    # Global and local graph load.
    "bias",
    "block_count_norm",
    "log_area_over_mean",
    "sqrt_area_fraction",
    "log_b2b_degree",
    "log_p2b_degree",
    "neighbor_area_over_mean",
    "neighbor_boundary_fraction",
    # Robust pin geometry.
    "pin_centroid_x_norm",
    "pin_centroid_y_norm",
    "pin_median_x_norm",
    "pin_median_y_norm",
    "pin_mad_x_norm",
    "pin_mad_y_norm",
    "has_pin",
    # Multi-hop graph messages.
    "message_x",
    "message_y",
    "multihop_area_over_mean",
    "multihop_boundary_fraction",
    "multihop_pin_fraction",
    "multihop_preplaced_fraction",
    # Direct constraint state.
    "fixed",
    "preplaced",
    "mib_member",
    "cluster_member",
    "boundary_left",
    "boundary_right",
    "boundary_top",
    "boundary_bottom",
    "hard_width_norm",
    "hard_height_norm",
    "preplaced_center_x_norm",
    "preplaced_center_y_norm",
    # MIB hyperedge aggregate/message.  No raw MIB ID is exposed.
    "mib_group_size_fraction",
    "mib_group_area_fraction",
    "mib_member_area_share",
    "mib_group_boundary_fraction",
    "mib_group_preplaced_fraction",
    "mib_internal_degree_fraction",
    "mib_message_x",
    "mib_message_y",
    # Cluster hyperedge aggregate/message.  No raw cluster ID is exposed.
    "cluster_group_size_fraction",
    "cluster_group_area_fraction",
    "cluster_member_area_share",
    "cluster_group_boundary_fraction",
    "cluster_group_preplaced_fraction",
    "cluster_internal_degree_fraction",
    "cluster_message_x",
    "cluster_message_y",
    # Connected and geometrically nearest preplaced obstacles.
    "connected_preplaced_degree_fraction",
    "connected_preplaced_dx_norm",
    "connected_preplaced_dy_norm",
    "connected_preplaced_width_norm",
    "connected_preplaced_height_norm",
    "nearest_obstacle_gap_x_norm",
    "nearest_obstacle_gap_y_norm",
    "nearest_obstacle_distance_norm",
    "nearest_obstacle_width_norm",
    "nearest_obstacle_height_norm",
    "has_other_preplaced_obstacle",
)

MIB_FEATURE_NAMES = frozenset(
    name for name in FEATURE_NAMES if name == "mib_member" or name.startswith("mib_")
)
MIB_FEATURE_INDICES = tuple(
    index for index, name in enumerate(FEATURE_NAMES) if name in MIB_FEATURE_NAMES
)
MIB_POLICIES = frozenset(("unmasked", "mask_incompatible", "mask_all"))


def _number(value):
    return float(value.item()) if hasattr(value, "item") else float(value)


def _finite_number(value, name):
    try:
        number = _number(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain numbers") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must contain only finite values")
    return number


def _rows(value):
    if value is None:
        return []
    rows = value.tolist() if hasattr(value, "tolist") else value
    if not isinstance(rows, (list, tuple)):
        raise ValueError("matrix inputs must contain rows")
    return rows


def _integer(value, name):
    number = _finite_number(value, name)
    integer = int(number)
    if number != integer:
        raise ValueError(f"{name} must contain integer values")
    return integer


def mib_is_input_compatible(
    block_count, area_targets, constraints, target_positions
):
    """Return whether every MIB group can share one legal input-visible shape.

    FloorSet's later training configurations contain MIB groups whose soft
    area intervals do not intersect.  Fixed/preplaced dimensions are part of
    the optimizer input and therefore may also constrain the common shape;
    free-block golden dimensions are deliberately never consulted.
    """
    n = _integer(block_count, "block_count")
    if n < 1:
        raise ValueError("block_count must be positive")
    raw_areas = _rows(area_targets)
    raw_constraints = _rows(constraints)
    raw_targets = _rows(target_positions)
    if len(raw_areas) < n:
        raise ValueError("area_targets is shorter than block_count")
    if not raw_constraints:
        return True
    if len(raw_constraints) < n:
        raise ValueError("constraints is shorter than block_count")
    if not raw_targets:
        raw_targets = [[-1.0] * 4 for _ in range(n)]
    if len(raw_targets) < n:
        raise ValueError("target_positions is shorter than block_count")

    groups = {}
    for block, row in enumerate(raw_constraints[:n]):
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            raise ValueError("constraints rows must have at least three values")
        identifier = _integer(row[2], "constraints")
        if identifier < 0:
            raise ValueError("constraint group identifiers must be non-negative")
        if identifier:
            groups.setdefault(identifier, []).append(block)

    for members in groups.values():
        if len(members) < 2:
            continue
        soft_areas = []
        hard_shapes = []
        for block in members:
            row = raw_constraints[block]
            fixed = len(row) > 0 and _finite_number(row[0], "constraints") != 0.0
            preplaced = len(row) > 1 and _finite_number(row[1], "constraints") != 0.0
            if fixed or preplaced:
                target = raw_targets[block]
                if not isinstance(target, (list, tuple)) or len(target) < 4:
                    raise ValueError("target_positions rows must have four values")
                width = _finite_number(target[2], "target_positions")
                height = _finite_number(target[3], "target_positions")
                if width <= 0.0 or height <= 0.0:
                    return False
                hard_shapes.append((width, height))
            else:
                area = _finite_number(raw_areas[block], "area_targets")
                if area <= 0.0:
                    return False
                soft_areas.append(area)

        lower = max((0.99 * area for area in soft_areas), default=None)
        upper = min((1.01 * area for area in soft_areas), default=None)
        if lower is not None and lower > upper + 1e-9:
            return False
        if hard_shapes:
            width, height = hard_shapes[0]
            if any(
                round(other_width, 4) != round(width, 4)
                or round(other_height, 4) != round(height, 4)
                for other_width, other_height in hard_shapes[1:]
            ):
                return False
            if lower is not None:
                area = width * height
                if area < lower - 1e-7 or area > upper + 1e-7:
                    return False
    return True


def apply_mib_feature_policy(
    features,
    *,
    policy,
    block_count,
    area_targets,
    constraints,
    target_positions,
):
    """Apply a declared MIB corruption policy and return rows plus metadata."""
    if policy not in MIB_POLICIES:
        raise ValueError(f"unsupported MIB feature policy: {policy!r}")
    compatible = mib_is_input_compatible(
        block_count, area_targets, constraints, target_positions
    )
    masked = policy == "mask_all" or (
        policy == "mask_incompatible" and not compatible
    )
    rows = [list(row) for row in features]
    if masked:
        for row in rows:
            for index in MIB_FEATURE_INDICES:
                row[index] = 0.0
    return rows, {"input_compatible": compatible, "masked": masked}


def _valid_edges(value, *, first_limit: int, second_limit: int, name: str):
    """Validate connectivity rows and discard only non-positive sentinels."""
    edges = []
    for row_index, row in enumerate(_rows(value)):
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            raise ValueError(f"{name} row {row_index} must have three values")
        # Connectivity dominates row count, so validate inline rather than
        # paying three Python helper calls for every (usually dense) edge.
        try:
            first_number = float(row[0])
            second_number = float(row[1])
            weight = float(row[2])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must contain numbers") from exc
        if not (
            math.isfinite(first_number)
            and math.isfinite(second_number)
            and math.isfinite(weight)
        ):
            raise ValueError(f"{name} must contain only finite values")
        first = int(first_number)
        second = int(second_number)
        if first_number != first or second_number != second:
            raise ValueError(f"{name} must contain integer endpoint values")
        if weight <= 0.0:
            continue
        if not (0 <= first < first_limit and 0 <= second < second_limit):
            raise ValueError(f"{name} has a positive edge with an invalid endpoint")
        edges.append((first, second, weight))
    return edges


def _weighted_median(samples, total_weight):
    if total_weight <= 0.0:
        return 0.0
    threshold = 0.5 * total_weight
    cumulative = 0.0
    for value, weight in sorted(samples, key=lambda sample: sample[0]):
        cumulative += weight
        if cumulative >= threshold:
            return value
    return samples[-1][0]


def _propagate_messages(
    initial_x,
    initial_y,
    anchor_x_sum,
    anchor_y_sum,
    anchor_weight_x,
    anchor_weight_y,
    scalar_channels,
    adjacency,
    degree,
    steps,
):
    """Fuse all graph messages into one bounded edge traversal per hop."""
    if len(scalar_channels) != 4:
        raise AssertionError("the fused propagation kernel expects four scalar channels")
    message_x = list(initial_x)
    message_y = list(initial_y)
    scalars = [list(channel) for channel in scalar_channels]
    for _ in range(steps):
        next_x = list(message_x)
        next_y = list(message_y)
        next_scalars = [list(channel) for channel in scalars]
        for block, neighbors in enumerate(adjacency):
            graph_weight = degree[block]
            self_weight = max(graph_weight, 1.0)
            neighbor_x = neighbor_y = 0.0
            neighbor_area = neighbor_boundary = 0.0
            neighbor_pin = neighbor_preplaced = 0.0
            for neighbor, weight in neighbors:
                neighbor_x += weight * message_x[neighbor]
                neighbor_y += weight * message_y[neighbor]
                neighbor_area += weight * scalars[0][neighbor]
                neighbor_boundary += weight * scalars[1][neighbor]
                neighbor_pin += weight * scalars[2][neighbor]
                neighbor_preplaced += weight * scalars[3][neighbor]

            x_weight = anchor_weight_x[block] + self_weight + graph_weight
            y_weight = anchor_weight_y[block] + self_weight + graph_weight
            next_x[block] = (
                anchor_x_sum[block]
                + self_weight * message_x[block]
                + neighbor_x
            ) / x_weight
            next_y[block] = (
                anchor_y_sum[block]
                + self_weight * message_y[block]
                + neighbor_y
            ) / y_weight
            if graph_weight > 0.0:
                next_scalars[0][block] = 0.5 * (
                    scalars[0][block] + neighbor_area / graph_weight
                )
                next_scalars[1][block] = 0.5 * (
                    scalars[1][block] + neighbor_boundary / graph_weight
                )
                next_scalars[2][block] = 0.5 * (
                    scalars[2][block] + neighbor_pin / graph_weight
                )
                next_scalars[3][block] = 0.5 * (
                    scalars[3][block] + neighbor_preplaced / graph_weight
                )
        message_x, message_y, scalars = next_x, next_y, next_scalars
    return message_x, message_y, scalars


def _group_ids(constraint_rows, column):
    ids = []
    for row in constraint_rows:
        identifier = _integer(row[column], "constraints")
        if identifier < 0:
            raise ValueError("constraint group identifiers must be non-negative")
        ids.append(identifier)
    return ids


def _hyperedge_features(
    group_ids,
    areas,
    total_area,
    boundary_flags,
    preplaced_flags,
    internal_degree_fraction,
    message_x,
    message_y,
):
    """Return eight invariant aggregate/message channels for each block."""
    n = len(areas)
    groups = {}
    for block, identifier in enumerate(group_ids):
        if identifier > 0:
            groups.setdefault(identifier, []).append(block)

    result = [[0.0] * 8 for _ in range(n)]
    for members in groups.values():
        group_area = sum(areas[member] for member in members)
        boundary_fraction = sum(boundary_flags[member] for member in members) / len(members)
        preplaced_fraction = sum(preplaced_flags[member] for member in members) / len(members)
        aggregate_x = sum(areas[member] * message_x[member] for member in members) / group_area
        aggregate_y = sum(areas[member] * message_y[member] for member in members) / group_area
        for member in members:
            result[member] = [
                len(members) / n,
                group_area / total_area,
                areas[member] / group_area,
                boundary_fraction,
                preplaced_fraction,
                internal_degree_fraction[member],
                aggregate_x,
                aggregate_y,
            ]
    return result


def _signed_axis_gap(point, low, high):
    if point < low:
        return low - point
    if point > high:
        return high - point
    return 0.0


def extract_order_features(
    block_count,
    area_targets,
    b2b_connectivity,
    p2b_connectivity,
    pins_pos,
    constraints,
    target_positions,
    *,
    message_steps=4,
):
    """Return one 60-value, input-only feature row per block.

    Runtime is ``O(message_steps * (blocks + edges) + blocks * preplaced)``;
    ``blocks`` and ``message_steps`` are capped at the official contest bounds.
    """
    n = _integer(block_count, "block_count")
    if not 1 <= n <= MAX_BLOCKS:
        raise ValueError(f"block_count must be between 1 and {MAX_BLOCKS}")
    steps = _integer(message_steps, "message_steps")
    if not 0 <= steps <= MAX_MESSAGE_STEPS:
        raise ValueError(
            f"message_steps must be between 0 and {MAX_MESSAGE_STEPS}"
        )

    areas_raw = _rows(area_targets)
    if len(areas_raw) < n:
        raise ValueError("area_targets is shorter than block_count")
    areas = [_finite_number(areas_raw[i], "area_targets") for i in range(n)]
    if any(area <= 0.0 for area in areas):
        raise ValueError("active block areas must be positive")
    total_area = sum(areas)
    if not math.isfinite(total_area):
        raise ValueError("total block area must be finite")
    mean_area = total_area / n
    root_area = math.sqrt(total_area)

    raw_constraints = _rows(constraints)
    if not raw_constraints:
        constraint_rows = [[0.0] * 5 for _ in range(n)]
    else:
        if len(raw_constraints) < n:
            raise ValueError("constraints is shorter than block_count")
        widths = []
        constraint_rows = []
        for row_index, row in enumerate(raw_constraints[:n]):
            if not isinstance(row, (list, tuple)):
                raise ValueError(f"constraints row {row_index} is not a row")
            widths.append(len(row))
            constraint_rows.append(
                [_finite_number(value, "constraints") for value in row]
            )
        if len(set(widths)) != 1 or widths[0] < 5:
            raise ValueError("constraints rows must have a uniform width of at least five")

    raw_targets = _rows(target_positions)
    if not raw_targets:
        target_rows = [[-1.0] * 4 for _ in range(n)]
    else:
        if len(raw_targets) < n:
            raise ValueError("target_positions is shorter than block_count")
        target_rows = []
        for row_index, row in enumerate(raw_targets[:n]):
            if not isinstance(row, (list, tuple)) or len(row) < 4:
                raise ValueError(
                    f"target_positions row {row_index} must have four values"
                )
            target_rows.append(
                [_finite_number(value, "target_positions") for value in row[:4]]
            )

    raw_pins = _rows(pins_pos)
    pins = []
    for row_index, row in enumerate(raw_pins):
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            raise ValueError(f"pins_pos row {row_index} must have two values")
        pins.append(
            [
                _finite_number(row[0], "pins_pos"),
                _finite_number(row[1], "pins_pos"),
            ]
        )

    fixed_flags = [row[0] != 0.0 for row in constraint_rows]
    preplaced_flags = [row[1] != 0.0 for row in constraint_rows]
    mib_ids = _group_ids(constraint_rows, 2)
    cluster_ids = _group_ids(constraint_rows, 3)
    boundaries = [_integer(row[4], "constraints") for row in constraint_rows]
    if any(boundary < 0 or boundary & ~15 for boundary in boundaries):
        raise ValueError("boundary masks must use only the low four bits")

    for block in range(n):
        if fixed_flags[block] or preplaced_flags[block]:
            x, y, width, height = target_rows[block]
            if width <= 0.0 or height <= 0.0:
                raise ValueError("fixed and preplaced blocks need positive dimensions")
            if preplaced_flags[block] and (x < 0.0 or y < 0.0):
                raise ValueError("preplaced blocks need non-negative coordinates")

    b2b = _valid_edges(
        b2b_connectivity, first_limit=n, second_limit=n, name="b2b_connectivity"
    )
    p2b = _valid_edges(
        p2b_connectivity,
        first_limit=len(pins),
        second_limit=n,
        name="p2b_connectivity",
    )

    x_extent = [root_area] + [abs(pin[0]) for pin in pins]
    y_extent = [root_area] + [abs(pin[1]) for pin in pins]
    for block in range(n):
        if preplaced_flags[block]:
            x, y, width, height = target_rows[block]
            x_extent.extend((abs(x), abs(x + width)))
            y_extent.extend((abs(y), abs(y + height)))
    x_scale = max(x_extent)
    y_scale = max(y_extent)

    degree = [0.0] * n
    neighbor_area = [0.0] * n
    neighbor_boundary = [0.0] * n
    mib_internal_degree = [0.0] * n
    cluster_internal_degree = [0.0] * n
    connected_weight = [0.0] * n
    connected_center_x = [0.0] * n
    connected_center_y = [0.0] * n
    connected_width = [0.0] * n
    connected_height = [0.0] * n
    adjacency = [[] for _ in range(n)]
    for first, second, weight in b2b:
        adjacency[first].append((second, weight))
        adjacency[second].append((first, weight))
        degree[first] += weight
        degree[second] += weight
        neighbor_area[first] += weight * areas[second]
        neighbor_area[second] += weight * areas[first]
        neighbor_boundary[first] += weight * bool(boundaries[second])
        neighbor_boundary[second] += weight * bool(boundaries[first])
        if mib_ids[first] > 0 and mib_ids[first] == mib_ids[second]:
            mib_internal_degree[first] += weight
            mib_internal_degree[second] += weight
        if cluster_ids[first] > 0 and cluster_ids[first] == cluster_ids[second]:
            cluster_internal_degree[first] += weight
            cluster_internal_degree[second] += weight
        if preplaced_flags[second]:
            x, y, width, height = target_rows[second]
            connected_weight[first] += weight
            connected_center_x[first] += weight * (x + width / 2.0)
            connected_center_y[first] += weight * (y + height / 2.0)
            connected_width[first] += weight * width
            connected_height[first] += weight * height
        if preplaced_flags[first]:
            x, y, width, height = target_rows[first]
            connected_weight[second] += weight
            connected_center_x[second] += weight * (x + width / 2.0)
            connected_center_y[second] += weight * (y + height / 2.0)
            connected_width[second] += weight * width
            connected_height[second] += weight * height
    for block in range(n):
        if degree[block] > 0.0:
            neighbor_area[block] /= degree[block]
            neighbor_boundary[block] /= degree[block]

    pin_degree = [0.0] * n
    pin_x = [0.0] * n
    pin_y = [0.0] * n
    pin_samples_x = [[] for _ in range(n)]
    pin_samples_y = [[] for _ in range(n)]
    for pin, block, weight in p2b:
        x, y = pins[pin]
        pin_degree[block] += weight
        pin_x[block] += weight * x
        pin_y[block] += weight * y
        pin_samples_x[block].append((x, weight))
        pin_samples_y[block].append((y, weight))

    pin_median_x = [0.0] * n
    pin_median_y = [0.0] * n
    pin_mad_x = [0.0] * n
    pin_mad_y = [0.0] * n
    for block in range(n):
        if pin_degree[block] <= 0.0:
            continue
        pin_x[block] /= pin_degree[block]
        pin_y[block] /= pin_degree[block]
        pin_median_x[block] = _weighted_median(
            pin_samples_x[block], pin_degree[block]
        )
        pin_median_y[block] = _weighted_median(
            pin_samples_y[block], pin_degree[block]
        )
        pin_mad_x[block] = sum(
            weight * abs(value - pin_median_x[block])
            for value, weight in pin_samples_x[block]
        ) / pin_degree[block]
        pin_mad_y[block] = sum(
            weight * abs(value - pin_median_y[block])
            for value, weight in pin_samples_y[block]
        ) / pin_degree[block]

    anchor_x_sum = [0.0] * n
    anchor_y_sum = [0.0] * n
    anchor_weight_x = [0.0] * n
    anchor_weight_y = [0.0] * n
    for block in range(n):
        if pin_degree[block] > 0.0:
            anchor_x_sum[block] += pin_degree[block] * pin_x[block] / x_scale
            anchor_y_sum[block] += pin_degree[block] * pin_y[block] / y_scale
            anchor_weight_x[block] += pin_degree[block]
            anchor_weight_y[block] += pin_degree[block]

        boundary_weight = max(1.0, 0.25 * (degree[block] + pin_degree[block]))
        if boundaries[block] & 1:
            anchor_weight_x[block] += boundary_weight
        if boundaries[block] & 2:
            anchor_x_sum[block] += boundary_weight
            anchor_weight_x[block] += boundary_weight
        if boundaries[block] & 8:
            anchor_weight_y[block] += boundary_weight
        if boundaries[block] & 4:
            anchor_y_sum[block] += boundary_weight
            anchor_weight_y[block] += boundary_weight

        if preplaced_flags[block]:
            x, y, width, height = target_rows[block]
            preplaced_weight = 10.0 * (degree[block] + pin_degree[block] + 1.0)
            anchor_x_sum[block] += preplaced_weight * (x + width / 2.0) / x_scale
            anchor_y_sum[block] += preplaced_weight * (y + height / 2.0) / y_scale
            anchor_weight_x[block] += preplaced_weight
            anchor_weight_y[block] += preplaced_weight

    initial_x = [
        anchor_x_sum[block] / anchor_weight_x[block]
        if anchor_weight_x[block] > 0.0
        else 0.5
        for block in range(n)
    ]
    initial_y = [
        anchor_y_sum[block] / anchor_weight_y[block]
        if anchor_weight_y[block] > 0.0
        else 0.5
        for block in range(n)
    ]
    message_x, message_y, scalar_messages = _propagate_messages(
        initial_x,
        initial_y,
        anchor_x_sum,
        anchor_y_sum,
        anchor_weight_x,
        anchor_weight_y,
        (
            [area / mean_area for area in areas],
            [float(bool(boundary)) for boundary in boundaries],
            [float(weight > 0.0) for weight in pin_degree],
            [float(value) for value in preplaced_flags],
        ),
        adjacency,
        degree,
        steps,
    )
    (
        multihop_area,
        multihop_boundary,
        multihop_pin,
        multihop_preplaced,
    ) = scalar_messages

    boundary_flags = [float(bool(boundary)) for boundary in boundaries]
    preplaced_float = [float(value) for value in preplaced_flags]
    mib_features = _hyperedge_features(
        mib_ids,
        areas,
        total_area,
        boundary_flags,
        preplaced_float,
        [
            mib_internal_degree[block] / degree[block]
            if degree[block] > 0.0
            else 0.0
            for block in range(n)
        ],
        message_x,
        message_y,
    )
    cluster_features = _hyperedge_features(
        cluster_ids,
        areas,
        total_area,
        boundary_flags,
        preplaced_float,
        [
            cluster_internal_degree[block] / degree[block]
            if degree[block] > 0.0
            else 0.0
            for block in range(n)
        ],
        message_x,
        message_y,
    )

    connected_obstacle = [[0.0] * 5 for _ in range(n)]
    for block in range(n):
        weight_sum = connected_weight[block]
        if weight_sum > 0.0:
            connected_obstacle[block] = [
                weight_sum / degree[block],
                connected_center_x[block] / weight_sum / x_scale - message_x[block],
                connected_center_y[block] / weight_sum / y_scale - message_y[block],
                connected_width[block] / weight_sum / root_area,
                connected_height[block] / weight_sum / root_area,
            ]

    obstacles = [
        (block, *target_rows[block])
        for block in range(n)
        if preplaced_flags[block]
    ]
    nearest_obstacle = [[0.0] * 6 for _ in range(n)]
    for block in range(n):
        query_x = message_x[block] * x_scale
        query_y = message_y[block] * y_scale
        candidates = []
        for obstacle, x, y, width, height in obstacles:
            if obstacle == block:
                continue
            gap_x = _signed_axis_gap(query_x, x, x + width) / x_scale
            gap_y = _signed_axis_gap(query_y, y, y + height) / y_scale
            distance = math.hypot(gap_x, gap_y)
            candidate = [
                gap_x,
                gap_y,
                distance,
                width / root_area,
                height / root_area,
                1.0,
            ]
            # The complete geometric key makes ties independent of block order.
            key = (
                distance,
                gap_x,
                gap_y,
                width / root_area,
                height / root_area,
                x / x_scale,
                y / y_scale,
            )
            candidates.append((key, candidate))
        if candidates:
            nearest_obstacle[block] = min(candidates, key=lambda item: item[0])[1]

    features = []
    for block in range(n):
        fixed = fixed_flags[block]
        preplaced = preplaced_flags[block]
        boundary = boundaries[block]
        width = target_rows[block][2] if fixed or preplaced else 0.0
        height = target_rows[block][3] if fixed or preplaced else 0.0
        if preplaced:
            x, y, pre_width, pre_height = target_rows[block]
            pre_x = (x + pre_width / 2.0) / x_scale
            pre_y = (y + pre_height / 2.0) / y_scale
        else:
            pre_x = pre_y = 0.0

        feature_row = [
            1.0,
            n / MAX_BLOCKS,
            math.log1p(areas[block] / mean_area),
            math.sqrt(areas[block] / total_area),
            math.log1p(degree[block]),
            math.log1p(pin_degree[block]),
            neighbor_area[block] / mean_area,
            neighbor_boundary[block],
            pin_x[block] / x_scale,
            pin_y[block] / y_scale,
            pin_median_x[block] / x_scale,
            pin_median_y[block] / y_scale,
            pin_mad_x[block] / x_scale,
            pin_mad_y[block] / y_scale,
            float(pin_degree[block] > 0.0),
            message_x[block],
            message_y[block],
            multihop_area[block],
            multihop_boundary[block],
            multihop_pin[block],
            multihop_preplaced[block],
            float(fixed),
            float(preplaced),
            float(mib_ids[block] > 0),
            float(cluster_ids[block] > 0),
            float(bool(boundary & 1)),
            float(bool(boundary & 2)),
            float(bool(boundary & 4)),
            float(bool(boundary & 8)),
            width / root_area,
            height / root_area,
            pre_x,
            pre_y,
            *mib_features[block],
            *cluster_features[block],
            *connected_obstacle[block],
            *nearest_obstacle[block],
        ]
        if len(feature_row) != len(FEATURE_NAMES):
            raise AssertionError("feature implementation does not match FEATURE_NAMES")
        if not all(math.isfinite(value) for value in feature_row):
            raise ValueError("feature aggregation overflowed to a non-finite value")
        features.append(feature_row)
    return features


def linear_predictions(features, weights):
    """Apply a [feature][output] coefficient matrix without numpy/torch."""
    if len(weights) != len(FEATURE_NAMES):
        raise ValueError("model feature count does not match FEATURE_NAMES")
    output_count = len(weights[0]) if weights else 0
    if output_count == 0 or any(len(row) != output_count for row in weights):
        raise ValueError("model weights must have a non-empty uniform output width")
    if any(len(row) != len(FEATURE_NAMES) for row in features):
        raise ValueError("feature row count does not match FEATURE_NAMES")
    return [
        [
            sum(row[column] * weights[column][output] for column in range(len(weights)))
            for output in range(output_count)
        ]
        for row in features
    ]


def standardized_linear_predictions(features, center, scale, weights):
    """Apply a trainer artifact's normalization and coefficient matrix."""
    feature_count = len(FEATURE_NAMES)
    if len(center) != feature_count or len(scale) != feature_count:
        raise ValueError("model normalization does not match FEATURE_NAMES")
    numeric_center = [_finite_number(value, "model center") for value in center]
    numeric_scale = [_finite_number(value, "model scale") for value in scale]
    if any(value <= 0.0 for value in numeric_scale):
        raise ValueError("model scale values must be positive")
    normalized = []
    for row in features:
        if len(row) != feature_count:
            raise ValueError("feature row count does not match FEATURE_NAMES")
        normalized.append(
            [
                (_finite_number(value, "features") - numeric_center[column])
                / numeric_scale[column]
                for column, value in enumerate(row)
            ]
        )
    return linear_predictions(normalized, weights)


def artifact_predictions(features, model, *, message_steps):
    """Validate and apply a ``train_order_model.py`` artifact."""
    if not isinstance(model, dict):
        raise ValueError("model artifact must be a JSON object")
    if model.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError("model artifact schema_version is unsupported")
    if model.get("model_type") != MODEL_TYPE:
        raise ValueError("model artifact model_type is unsupported")

    schema = model.get("feature_schema")
    if not isinstance(schema, dict):
        raise ValueError("model feature_schema must be an object")
    if schema.get("version") != FEATURE_VERSION:
        raise ValueError("model feature version does not match inference schema")
    if schema.get("names") != list(FEATURE_NAMES):
        raise ValueError("model feature names do not match inference schema")
    try:
        artifact_steps = _integer(schema.get("message_steps"), "model message_steps")
        inference_steps = _integer(message_steps, "message_steps")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("model message_steps must be an integer") from exc
    if artifact_steps != inference_steps:
        raise ValueError("model message_steps does not match feature extraction")

    target_schema = model.get("target_schema")
    if not isinstance(target_schema, dict):
        raise ValueError("model target_schema must be an object")
    if target_schema.get("names") != list(TARGET_NAMES):
        raise ValueError("model target names do not match inference outputs")
    output_count = len(TARGET_NAMES)

    normalization = model.get("normalization")
    if not isinstance(normalization, dict):
        raise ValueError("model normalization must be an object")
    center = normalization.get("center")
    scale = normalization.get("scale")
    if not isinstance(center, (list, tuple)) or not isinstance(scale, (list, tuple)):
        raise ValueError("model normalization vectors must be arrays")
    if len(center) != len(FEATURE_NAMES) or len(scale) != len(FEATURE_NAMES):
        raise ValueError("model normalization does not match FEATURE_NAMES")
    numeric_center = [_finite_number(value, "model center") for value in center]
    numeric_scale = [_finite_number(value, "model scale") for value in scale]
    if any(value <= 0.0 for value in numeric_scale):
        raise ValueError("model scale values must be positive")

    coefficients = model.get("coefficients")
    if not isinstance(coefficients, (list, tuple)):
        raise ValueError("model coefficients must be an array")
    if len(coefficients) != len(FEATURE_NAMES):
        raise ValueError("model feature count does not match FEATURE_NAMES")
    numeric_coefficients = []
    for row in coefficients:
        if not isinstance(row, (list, tuple)) or len(row) != output_count:
            raise ValueError("model coefficient output width does not match target_schema")
        numeric_coefficients.append(
            [_finite_number(value, "model coefficients") for value in row]
        )

    expected_payload = model.get("payload_sha256")
    if not (
        isinstance(expected_payload, str)
        and len(expected_payload) == 64
        and all(character in "0123456789abcdef" for character in expected_payload)
    ):
        raise ValueError("model payload_sha256 must be a lowercase SHA-256 digest")
    payload_model = dict(model)
    del payload_model["payload_sha256"]
    try:
        payload = json.dumps(
            payload_model,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("model artifact is not canonical JSON") from exc
    actual_payload = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(expected_payload, actual_payload):
        raise ValueError("model payload_sha256 integrity check failed")

    predictions = standardized_linear_predictions(
        features,
        numeric_center,
        numeric_scale,
        numeric_coefficients,
    )
    if any(len(row) != output_count for row in predictions):
        raise ValueError("model prediction output width does not match target_schema")
    if any(not math.isfinite(value) for row in predictions for value in row):
        raise ValueError("model predictions must be finite")
    return predictions


def extract_artifact_predictions(
    block_count,
    area_targets,
    b2b_connectivity,
    p2b_connectivity,
    pins_pos,
    constraints,
    target_positions,
    model,
):
    """Extract, corruption-mask, and score features under one artifact contract."""
    if not isinstance(model, dict):
        raise ValueError("model artifact must be a JSON object")
    schema = model.get("feature_schema")
    if not isinstance(schema, dict):
        raise ValueError("model feature_schema must be an object")
    message_steps = _integer(schema.get("message_steps"), "model message_steps")
    mib_policy = schema.get("mib_policy", "unmasked")
    features = extract_order_features(
        block_count,
        area_targets,
        b2b_connectivity,
        p2b_connectivity,
        pins_pos,
        constraints,
        target_positions,
        message_steps=message_steps,
    )
    features, mask_metadata = apply_mib_feature_policy(
        features,
        policy=mib_policy,
        block_count=block_count,
        area_targets=area_targets,
        constraints=constraints,
        target_positions=target_positions,
    )
    return artifact_predictions(
        features, model, message_steps=message_steps
    ), mask_metadata
