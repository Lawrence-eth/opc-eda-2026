#!/usr/bin/env python3
"""Mine structural priors from the validation golden layouts.

The output is intentionally public-safe: it summarizes validation labels that
are already shipped with the contest checkout, and it mirrors the evaluator's
boundary/grouping/MIB soft-constraint definitions.

Usage:
    python3 scripts/mine_golden.py
    python3 scripts/mine_golden.py --output results/golden_structure.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTEST_DIR = ROOT / "external" / "FloorSet" / "iccad2026contest"
DEFAULT_DATA_PATH = ROOT / "external" / "FloorSet"
DEFAULT_OUTPUT = ROOT / "results" / "golden_structure.json"
EPS = 1e-6

sys.path.insert(0, str(CONTEST_DIR))
sys.path.insert(0, str(CONTEST_DIR.parent))

try:
    from iccad2026_evaluate import ContestEvaluator
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    if exc.name != "iccad2026_evaluate":
        raise
    raise SystemExit(
        "Cannot import iccad2026_evaluate.py. Expected FloorSet at "
        f"{CONTEST_DIR}"
    ) from exc


Rect = tuple[float, float, float, float]


def _scalar(value: Any) -> float:
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _constraint_rows(constraints: Any, block_count: int) -> list[list[float]]:
    if constraints is None:
        return []
    if hasattr(constraints, "detach"):
        rows = constraints.detach().cpu().tolist()
    elif hasattr(constraints, "tolist"):
        rows = constraints.tolist()
    else:
        rows = constraints
    out: list[list[float]] = []
    for row in rows[:block_count]:
        if not isinstance(row, (list, tuple)):
            row = [row]
        out.append([_scalar(v) for v in row])
    return out


def _col(row: list[float], idx: int, default: float = 0.0) -> float:
    return row[idx] if idx < len(row) else default


def _flag_indices(rows: list[list[float]], col: int) -> list[int]:
    return [i for i, row in enumerate(rows) if int(_col(row, col)) != 0]


def _positive_groups(rows: list[list[float]], col: int) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        gid = int(_col(row, col))
        if gid > 0:
            groups[gid].append(i)
    return dict(sorted(groups.items()))


def _extract_positions(polygons: Any, block_count: int) -> list[Rect]:
    positions: list[Rect] = []
    for i in range(block_count):
        block = polygons[i]
        valid = block[block[:, 0] != -1]
        if len(valid) > 0:
            x_min, y_min = valid.min(dim=0).values
            x_max, y_max = valid.max(dim=0).values
            positions.append(
                (
                    float(x_min),
                    float(y_min),
                    float(x_max - x_min),
                    float(y_max - y_min),
                )
            )
        else:
            positions.append((0.0, 0.0, 1.0, 1.0))
    return positions


def _bbox(positions: list[Rect]) -> tuple[float, float, float, float, float]:
    x_min = min(x for x, _y, _w, _h in positions)
    y_min = min(y for _x, y, _w, _h in positions)
    x_max = max(x + w for x, _y, w, _h in positions)
    y_max = max(y + h for _x, y, _w, h in positions)
    return x_min, y_min, x_max, y_max, (x_max - x_min) * (y_max - y_min)


def _interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return min(a1, b1) - max(a0, b0)


def _touches_edge(a: Rect, b: Rect, eps: float = EPS) -> bool:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    vertical = (
        (abs(ax1 - bx0) <= eps or abs(bx1 - ax0) <= eps)
        and _interval_overlap(ay0, ay1, by0, by1) > eps
    )
    horizontal = (
        (abs(ay1 - by0) <= eps or abs(by1 - ay0) <= eps)
        and _interval_overlap(ax0, ax1, bx0, bx1) > eps
    )
    return vertical or horizontal


def _component_stats(
    positions: list[Rect], members: list[int]
) -> tuple[int, int, list[int]]:
    if not members:
        return 0, 0, []
    parent = list(range(len(members)))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    edge_contacts = 0
    for local_i, block_i in enumerate(members):
        for local_j in range(local_i + 1, len(members)):
            block_j = members[local_j]
            if _touches_edge(positions[block_i], positions[block_j]):
                edge_contacts += 1
                union(local_i, local_j)

    sizes: Counter[int] = Counter(find(i) for i in range(len(members)))
    return len(sizes), edge_contacts, sorted(sizes.values(), reverse=True)


def _boundary_ok(rect: Rect, code: int, bbox: tuple[float, float, float, float, float]) -> bool:
    x_min, y_min, x_max, y_max, _area = bbox
    x, y, w, h = rect
    touches = {
        1: abs(x - x_min) < EPS,
        2: abs(x + w - x_max) < EPS,
        4: abs(y + h - y_max) < EPS,
        8: abs(y - y_min) < EPS,
    }
    return all(touches[bit] for bit in (1, 2, 4, 8) if code & bit)


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = pct * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _stats(values: list[float]) -> dict[str, float | int]:
    clean = sorted(float(v) for v in values)
    if not clean:
        return {"count": 0}
    return {
        "count": len(clean),
        "min": clean[0],
        "p10": _percentile(clean, 0.10),
        "median": _percentile(clean, 0.50),
        "mean": sum(clean) / len(clean),
        "p90": _percentile(clean, 0.90),
        "p95": _percentile(clean, 0.95),
        "max": clean[-1],
    }


def _unique_lines(values: list[float]) -> list[float]:
    lines: list[float] = []
    for value in sorted(values):
        if not lines or abs(value - lines[-1]) > EPS:
            lines.append(value)
    return lines


def _matches_line(value: float, lines: list[float]) -> bool:
    return any(abs(value - line) <= EPS for line in lines)


def _anchor_alignment(positions: list[Rect], anchor_indices: set[int]) -> dict[str, Any]:
    target_indices = [i for i in range(len(positions)) if i not in anchor_indices]
    if not anchor_indices:
        return {
            "anchor_blocks": 0,
            "anchor_x_lines": 0,
            "anchor_y_lines": 0,
            "target_blocks": len(target_indices),
            "x_aligned_blocks": 0,
            "y_aligned_blocks": 0,
            "any_aligned_blocks": 0,
            "adjacent_blocks": 0,
            "any_alignment_rate": None,
            "adjacent_rate": None,
        }

    x_lines = _unique_lines(
        [positions[i][0] for i in anchor_indices]
        + [positions[i][0] + positions[i][2] for i in anchor_indices]
    )
    y_lines = _unique_lines(
        [positions[i][1] for i in anchor_indices]
        + [positions[i][1] + positions[i][3] for i in anchor_indices]
    )
    x_aligned = 0
    y_aligned = 0
    any_aligned = 0
    adjacent = 0
    for i in target_indices:
        x, y, w, h = positions[i]
        x_hit = _matches_line(x, x_lines) or _matches_line(x + w, x_lines)
        y_hit = _matches_line(y, y_lines) or _matches_line(y + h, y_lines)
        if x_hit:
            x_aligned += 1
        if y_hit:
            y_aligned += 1
        if x_hit or y_hit:
            any_aligned += 1
        if any(_touches_edge(positions[i], positions[j]) for j in anchor_indices):
            adjacent += 1

    target_count = len(target_indices)
    return {
        "anchor_blocks": len(anchor_indices),
        "anchor_x_lines": len(x_lines),
        "anchor_y_lines": len(y_lines),
        "target_blocks": target_count,
        "x_aligned_blocks": x_aligned,
        "y_aligned_blocks": y_aligned,
        "any_aligned_blocks": any_aligned,
        "adjacent_blocks": adjacent,
        "any_alignment_rate": any_aligned / target_count if target_count else None,
        "adjacent_rate": adjacent / target_count if target_count else None,
    }


def _support_stats(positions: list[Rect], bbox: tuple[float, float, float, float, float]) -> dict[str, Any]:
    _x_min, y_min, _x_max, _y_max, _area = bbox
    floor_supported = 0
    block_supported = 0
    unsupported = 0
    edge_touched = 0
    for i, (x, y, w, _h) in enumerate(positions):
        if abs(y - y_min) <= EPS:
            floor_supported += 1
            supported = True
        else:
            supported = any(
                j != i
                and abs(positions[j][1] + positions[j][3] - y) <= EPS
                and _interval_overlap(x, x + w, positions[j][0], positions[j][0] + positions[j][2]) > EPS
                for j in range(len(positions))
            )
            if supported:
                block_supported += 1
            else:
                unsupported += 1
        if any(j != i and _touches_edge(positions[i], positions[j]) for j in range(len(positions))):
            edge_touched += 1

    return {
        "floor_supported_blocks": floor_supported,
        "block_supported_blocks": block_supported,
        "unsupported_above_floor_blocks": unsupported,
        "edge_touched_blocks": edge_touched,
        "unsupported_above_floor_rate": unsupported / len(positions) if positions else 0.0,
        "edge_touched_rate": edge_touched / len(positions) if positions else 0.0,
    }


def _mine_case(idx: int, sample: dict[str, Any]) -> tuple[dict[str, Any], list[float]]:
    inputs, labels = sample["input"], sample["label"]
    area_target, _b2b_conn, _p2b_conn, _pins_pos, constraints = inputs
    polygons = labels[0]
    block_count = int((area_target != -1).sum().item())

    positions = _extract_positions(polygons, block_count)
    rows = _constraint_rows(constraints, block_count)
    bbox = _bbox(positions)
    bbox_area = bbox[4]
    total_block_area = sum(
        _scalar(area_target[i]) for i in range(block_count) if _scalar(area_target[i]) > 0
    )
    utilization = total_block_area / max(bbox_area, EPS)

    fixed = set(_flag_indices(rows, 0))
    preplaced = set(_flag_indices(rows, 1))
    boundary_codes = {
        i: int(_col(row, 4))
        for i, row in enumerate(rows)
        if int(_col(row, 4)) > 0
    }
    cluster_groups = _positive_groups(rows, 3)
    mib_groups = _positive_groups(rows, 2)

    soft_aspects: list[float] = []
    for i, (_x, _y, w, h) in enumerate(positions):
        if i in fixed or i in preplaced:
            continue
        soft_aspects.append(max(w, h) / max(min(w, h), EPS))

    boundary_by_code: dict[str, dict[str, int]] = {}
    boundary_violations = 0
    preplaced_boundary_blocks = 0
    preplaced_boundary_violations = 0
    fixed_boundary_blocks = 0
    fixed_boundary_violations = 0
    for i, code in boundary_codes.items():
        ok = _boundary_ok(positions[i], code, bbox)
        code_key = str(code)
        bucket = boundary_by_code.setdefault(
            code_key,
            {
                "blocks": 0,
                "satisfied": 0,
                "violations": 0,
                "preplaced_blocks": 0,
                "preplaced_violations": 0,
                "fixed_blocks": 0,
                "fixed_violations": 0,
            },
        )
        bucket["blocks"] += 1
        if ok:
            bucket["satisfied"] += 1
        else:
            bucket["violations"] += 1
            boundary_violations += 1
        if i in preplaced:
            preplaced_boundary_blocks += 1
            bucket["preplaced_blocks"] += 1
            if not ok:
                preplaced_boundary_violations += 1
                bucket["preplaced_violations"] += 1
        if i in fixed:
            fixed_boundary_blocks += 1
            bucket["fixed_blocks"] += 1
            if not ok:
                fixed_boundary_violations += 1
                bucket["fixed_violations"] += 1

    cluster_component_hist: Counter[str] = Counter()
    cluster_connected = 0
    cluster_contact_edges = 0
    cluster_groups_with_boundary = 0
    cluster_groups_with_preplaced = 0
    preplaced_cluster_groups_connected = 0
    preplaced_cluster_groups_with_anchor_touch = 0
    preplaced_cluster_members = 0
    preplaced_cluster_members_touching = 0
    grouping_violations = 0
    for members in cluster_groups.values():
        component_count, contact_edges, _component_sizes = _component_stats(positions, members)
        cluster_component_hist[str(component_count)] += 1
        cluster_contact_edges += contact_edges
        grouping_violations += max(0, component_count - 1)
        if component_count == 1:
            cluster_connected += 1
        if any(i in boundary_codes for i in members):
            cluster_groups_with_boundary += 1

        preplaced_members = [i for i in members if i in preplaced]
        if preplaced_members:
            cluster_groups_with_preplaced += 1
            if component_count == 1:
                preplaced_cluster_groups_connected += 1
            group_has_anchor_touch = False
            for i in preplaced_members:
                touches = any(j != i and _touches_edge(positions[i], positions[j]) for j in members)
                preplaced_cluster_members += 1
                if touches:
                    preplaced_cluster_members_touching += 1
                    group_has_anchor_touch = True
            if group_has_anchor_touch:
                preplaced_cluster_groups_with_anchor_touch += 1

    mib_uniform = 0
    mib_shape_hist: Counter[str] = Counter()
    mib_violations = 0
    for members in mib_groups.values():
        shapes = {
            (round(positions[i][2], 4), round(positions[i][3], 4))
            for i in members
        }
        mib_shape_hist[str(len(shapes))] += 1
        mib_violations += max(0, len(shapes) - 1)
        if len(shapes) == 1:
            mib_uniform += 1

    n_soft = (
        len(boundary_codes)
        + sum(max(0, len(members) - 1) for members in cluster_groups.values())
        + sum(max(0, len(members) - 1) for members in mib_groups.values())
    )
    soft_violations = boundary_violations + grouping_violations + mib_violations

    case = {
        "test_id": idx,
        "block_count": block_count,
        "bbox": {
            "x_min": bbox[0],
            "y_min": bbox[1],
            "x_max": bbox[2],
            "y_max": bbox[3],
            "area": bbox_area,
        },
        "total_block_area": total_block_area,
        "utilization": utilization,
        "soft_block_aspect_ratio": _stats(soft_aspects),
        "constraints": {
            "fixed_blocks": len(fixed),
            "preplaced_blocks": len(preplaced),
            "boundary_blocks": len(boundary_codes),
            "cluster_blocks": sum(len(members) for members in cluster_groups.values()),
            "cluster_groups": len(cluster_groups),
            "mib_blocks": sum(len(members) for members in mib_groups.values()),
            "mib_groups": len(mib_groups),
        },
        "soft": {
            "n_soft": n_soft,
            "violations": soft_violations,
            "violations_relative": soft_violations / n_soft if n_soft else 0.0,
            "boundary_violations": boundary_violations,
            "grouping_violations": grouping_violations,
            "mib_violations": mib_violations,
        },
        "boundary": {
            "blocks": len(boundary_codes),
            "satisfied": len(boundary_codes) - boundary_violations,
            "violations": boundary_violations,
            "by_code": dict(sorted(boundary_by_code.items(), key=lambda item: int(item[0]))),
            "preplaced_blocks": preplaced_boundary_blocks,
            "preplaced_violations": preplaced_boundary_violations,
            "fixed_blocks": fixed_boundary_blocks,
            "fixed_violations": fixed_boundary_violations,
        },
        "clusters": {
            "groups": len(cluster_groups),
            "blocks": sum(len(members) for members in cluster_groups.values()),
            "connected_groups": cluster_connected,
            "component_histogram": dict(sorted(cluster_component_hist.items(), key=lambda item: int(item[0]))),
            "contact_edges": cluster_contact_edges,
            "groups_with_boundary": cluster_groups_with_boundary,
            "groups_with_preplaced": cluster_groups_with_preplaced,
            "preplaced_groups_connected": preplaced_cluster_groups_connected,
            "preplaced_groups_with_anchor_touch": preplaced_cluster_groups_with_anchor_touch,
            "preplaced_members": preplaced_cluster_members,
            "preplaced_members_touching_cluster": preplaced_cluster_members_touching,
        },
        "mib": {
            "groups": len(mib_groups),
            "blocks": sum(len(members) for members in mib_groups.values()),
            "uniform_groups": mib_uniform,
            "distinct_shape_histogram": dict(sorted(mib_shape_hist.items(), key=lambda item: int(item[0]))),
        },
        "preplaced_alignment": _anchor_alignment(positions, preplaced),
        "fixed_alignment": _anchor_alignment(positions, fixed),
        "support": _support_stats(positions, bbox),
    }
    return case, soft_aspects


def _load_golden_score_check(cases: list[dict[str, Any]]) -> dict[str, Any] | None:
    path = ROOT / "results" / "golden_scored.json"
    if not path.exists():
        return None
    scored = json.loads(path.read_text())
    if not isinstance(scored, list) or len(scored) < len(cases):
        return {"status": "skipped", "reason": "golden_scored.json has unexpected shape"}

    diffs = [
        abs(float(case["soft"]["violations_relative"]) - float(scored[i].get("vr", 0.0)))
        for i, case in enumerate(cases)
    ]
    mismatches = [
        {
            "test_id": cases[i]["test_id"],
            "mined_vr": cases[i]["soft"]["violations_relative"],
            "golden_scored_vr": float(scored[i].get("vr", 0.0)),
            "abs_diff": diff,
        }
        for i, diff in enumerate(diffs)
        if diff > 1e-9
    ]
    return {
        "status": "ok" if not mismatches else "mismatch",
        "max_abs_vr_diff": max(diffs) if diffs else 0.0,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:10],
    }


def _rate(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _summarize(cases: list[dict[str, Any]], all_soft_aspects: list[float]) -> dict[str, Any]:
    boundary_by_code: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "blocks": 0,
            "satisfied": 0,
            "violations": 0,
            "preplaced_blocks": 0,
            "preplaced_violations": 0,
            "fixed_blocks": 0,
            "fixed_violations": 0,
        }
    )
    cluster_component_hist: Counter[str] = Counter()
    mib_shape_hist: Counter[str] = Counter()
    for case in cases:
        for code, bucket in case["boundary"]["by_code"].items():
            for key, value in bucket.items():
                boundary_by_code[code][key] += int(value)
        cluster_component_hist.update(case["clusters"]["component_histogram"])
        mib_shape_hist.update(case["mib"]["distinct_shape_histogram"])

    boundary_blocks = sum(case["boundary"]["blocks"] for case in cases)
    boundary_violations = sum(case["boundary"]["violations"] for case in cases)
    preplaced_boundary_blocks = sum(case["boundary"]["preplaced_blocks"] for case in cases)
    preplaced_boundary_violations = sum(case["boundary"]["preplaced_violations"] for case in cases)
    fixed_boundary_blocks = sum(case["boundary"]["fixed_blocks"] for case in cases)
    fixed_boundary_violations = sum(case["boundary"]["fixed_violations"] for case in cases)

    cluster_groups = sum(case["clusters"]["groups"] for case in cases)
    cluster_connected = sum(case["clusters"]["connected_groups"] for case in cases)
    preplaced_cluster_groups = sum(case["clusters"]["groups_with_preplaced"] for case in cases)
    preplaced_cluster_connected = sum(case["clusters"]["preplaced_groups_connected"] for case in cases)
    preplaced_cluster_anchor_touch = sum(case["clusters"]["preplaced_groups_with_anchor_touch"] for case in cases)
    preplaced_cluster_members = sum(case["clusters"]["preplaced_members"] for case in cases)
    preplaced_cluster_members_touching = sum(
        case["clusters"]["preplaced_members_touching_cluster"] for case in cases
    )

    mib_groups = sum(case["mib"]["groups"] for case in cases)
    mib_uniform = sum(case["mib"]["uniform_groups"] for case in cases)

    preplaced_anchor_cases = [
        case for case in cases if case["preplaced_alignment"]["anchor_blocks"] > 0
    ]
    fixed_anchor_cases = [
        case for case in cases if case["fixed_alignment"]["anchor_blocks"] > 0
    ]
    unsupported_blocks = sum(case["support"]["unsupported_above_floor_blocks"] for case in cases)
    total_blocks = sum(case["block_count"] for case in cases)

    return {
        "cases": len(cases),
        "block_count": _stats([case["block_count"] for case in cases]),
        "utilization": _stats([case["utilization"] for case in cases]),
        "soft_block_aspect_ratio": _stats(all_soft_aspects),
        "soft_constraints": {
            "total_n_soft": sum(case["soft"]["n_soft"] for case in cases),
            "total_violations": sum(case["soft"]["violations"] for case in cases),
            "cases_with_any_violation": sum(1 for case in cases if case["soft"]["violations"] > 0),
            "violations_relative": _stats([case["soft"]["violations_relative"] for case in cases]),
            "boundary_violations": boundary_violations,
            "grouping_violations": sum(case["soft"]["grouping_violations"] for case in cases),
            "mib_violations": sum(case["soft"]["mib_violations"] for case in cases),
        },
        "boundary": {
            "blocks": boundary_blocks,
            "satisfied": boundary_blocks - boundary_violations,
            "violations": boundary_violations,
            "satisfaction_rate": _rate(boundary_blocks - boundary_violations, boundary_blocks),
            "preplaced_blocks": preplaced_boundary_blocks,
            "preplaced_violations": preplaced_boundary_violations,
            "preplaced_violation_rate": _rate(preplaced_boundary_violations, preplaced_boundary_blocks),
            "fixed_blocks": fixed_boundary_blocks,
            "fixed_violations": fixed_boundary_violations,
            "fixed_violation_rate": _rate(fixed_boundary_violations, fixed_boundary_blocks),
            "by_code": dict(sorted(boundary_by_code.items(), key=lambda item: int(item[0]))),
        },
        "clusters": {
            "groups": cluster_groups,
            "connected_groups": cluster_connected,
            "connected_rate": _rate(cluster_connected, cluster_groups),
            "component_histogram": dict(sorted(cluster_component_hist.items(), key=lambda item: int(item[0]))),
            "groups_with_preplaced": preplaced_cluster_groups,
            "preplaced_groups_connected": preplaced_cluster_connected,
            "preplaced_groups_connected_rate": _rate(
                preplaced_cluster_connected, preplaced_cluster_groups
            ),
            "preplaced_groups_with_anchor_touch": preplaced_cluster_anchor_touch,
            "preplaced_groups_anchor_touch_rate": _rate(
                preplaced_cluster_anchor_touch, preplaced_cluster_groups
            ),
            "preplaced_members": preplaced_cluster_members,
            "preplaced_members_touching_cluster": preplaced_cluster_members_touching,
            "preplaced_member_touch_rate": _rate(
                preplaced_cluster_members_touching, preplaced_cluster_members
            ),
        },
        "mib": {
            "groups": mib_groups,
            "uniform_groups": mib_uniform,
            "uniform_rate": _rate(mib_uniform, mib_groups),
            "distinct_shape_histogram": dict(sorted(mib_shape_hist.items(), key=lambda item: int(item[0]))),
        },
        "preplaced_alignment": {
            "cases_with_preplaced": len(preplaced_anchor_cases),
            "any_alignment_rate_by_case": _stats(
                [
                    case["preplaced_alignment"]["any_alignment_rate"]
                    for case in preplaced_anchor_cases
                    if case["preplaced_alignment"]["any_alignment_rate"] is not None
                ]
            ),
            "adjacent_rate_by_case": _stats(
                [
                    case["preplaced_alignment"]["adjacent_rate"]
                    for case in preplaced_anchor_cases
                    if case["preplaced_alignment"]["adjacent_rate"] is not None
                ]
            ),
            "aligned_blocks": sum(case["preplaced_alignment"]["any_aligned_blocks"] for case in cases),
            "adjacent_blocks": sum(case["preplaced_alignment"]["adjacent_blocks"] for case in cases),
        },
        "fixed_alignment": {
            "cases_with_fixed": len(fixed_anchor_cases),
            "any_alignment_rate_by_case": _stats(
                [
                    case["fixed_alignment"]["any_alignment_rate"]
                    for case in fixed_anchor_cases
                    if case["fixed_alignment"]["any_alignment_rate"] is not None
                ]
            ),
            "adjacent_rate_by_case": _stats(
                [
                    case["fixed_alignment"]["adjacent_rate"]
                    for case in fixed_anchor_cases
                    if case["fixed_alignment"]["adjacent_rate"] is not None
                ]
            ),
            "aligned_blocks": sum(case["fixed_alignment"]["any_aligned_blocks"] for case in cases),
            "adjacent_blocks": sum(case["fixed_alignment"]["adjacent_blocks"] for case in cases),
        },
        "support": {
            "unsupported_above_floor_blocks": unsupported_blocks,
            "unsupported_above_floor_rate": _rate(unsupported_blocks, total_blocks),
            "unsupported_above_floor_rate_by_case": _stats(
                [case["support"]["unsupported_above_floor_rate"] for case in cases]
            ),
            "edge_touched_rate_by_case": _stats(
                [case["support"]["edge_touched_rate"] for case in cases]
            ),
        },
        "diagnostics": {
            "highest_soft_violation_cases": [
                {
                    "test_id": case["test_id"],
                    "block_count": case["block_count"],
                    "violations_relative": case["soft"]["violations_relative"],
                    "boundary": case["soft"]["boundary_violations"],
                    "grouping": case["soft"]["grouping_violations"],
                    "mib": case["soft"]["mib_violations"],
                }
                for case in sorted(
                    cases,
                    key=lambda item: (
                        item["soft"]["violations_relative"],
                        item["soft"]["violations"],
                    ),
                    reverse=True,
                )[:10]
            ],
            "most_disconnected_cluster_cases": [
                {
                    "test_id": case["test_id"],
                    "block_count": case["block_count"],
                    "cluster_groups": case["clusters"]["groups"],
                    "connected_groups": case["clusters"]["connected_groups"],
                    "grouping_violations": case["soft"]["grouping_violations"],
                    "component_histogram": case["clusters"]["component_histogram"],
                }
                for case in sorted(
                    cases,
                    key=lambda item: item["soft"]["grouping_violations"],
                    reverse=True,
                )[:10]
            ],
            "most_unsupported_cases": [
                {
                    "test_id": case["test_id"],
                    "block_count": case["block_count"],
                    "unsupported_above_floor_blocks": case["support"]["unsupported_above_floor_blocks"],
                    "unsupported_above_floor_rate": case["support"]["unsupported_above_floor_rate"],
                }
                for case in sorted(
                    cases,
                    key=lambda item: item["support"]["unsupported_above_floor_blocks"],
                    reverse=True,
                )[:10]
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data_path = args.data_path.resolve()
    if not (data_path / "LiteTensorDataTest").exists():
        raise SystemExit(
            f"Cannot find LiteTensorDataTest under {data_path}. "
            "Run from a checkout with external/FloorSet synced."
        )

    evaluator = ContestEvaluator(data_path=str(data_path), verbose=False)
    evaluator._load_dataset()

    cases: list[dict[str, Any]] = []
    all_soft_aspects: list[float] = []
    for idx in range(len(evaluator.dataset)):
        case, aspects = _mine_case(idx, evaluator.dataset[idx])
        cases.append(case)
        all_soft_aspects.extend(aspects)

    output = {
        "schema_version": 1,
        "source": {
            "dataset": "LiteTensorDataTest",
            "cases": len(cases),
            "data_path": str(data_path),
            "evaluator_soft_semantics": {
                "boundary": "bitmask bbox-edge contact, 1=left, 2=right, 4=top, 8=bottom",
                "grouping": "connected components from exact shared rectangle edges",
                "mib": "distinct rounded (w, h) pairs per group, 4 decimal places",
            },
        },
        "summary": _summarize(cases, all_soft_aspects),
        "golden_scored_check": _load_golden_score_check(cases),
        "cases": cases,
    }

    out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    summary = output["summary"]
    print(f"wrote {out_path}")
    print(f"cases: {summary['cases']}")
    print(
        "soft violations: "
        f"{summary['soft_constraints']['total_violations']} / "
        f"{summary['soft_constraints']['total_n_soft']} "
        f"across {summary['soft_constraints']['cases_with_any_violation']} cases"
    )
    print(
        "cluster connected rate: "
        f"{summary['clusters']['connected_groups']} / {summary['clusters']['groups']} "
        f"({summary['clusters']['connected_rate']:.3f})"
    )
    print(
        "boundary satisfaction: "
        f"{summary['boundary']['satisfied']} / {summary['boundary']['blocks']} "
        f"({summary['boundary']['satisfaction_rate']:.3f})"
    )
    check = output["golden_scored_check"]
    if check:
        print(
            "golden_scored vr check: "
            f"{check['status']} max_abs_diff={check['max_abs_vr_diff']:.3g}"
        )


if __name__ == "__main__":
    main()
