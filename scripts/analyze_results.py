#!/usr/bin/env python3
"""Analyze FloorSet validation result JSON files.

The official evaluator stores per-case quality metrics in results/boundary_full.json.
This script makes the next optimization step less blind by highlighting the cases
that dominate the score and by aggregating metrics by block-count range.

Usage:
    python scripts/analyze_results.py
    python scripts/analyze_results.py results/boundary_full.json --top 30
    python scripts/analyze_results.py results/boundary_full.json --contest-dir external/FloorSet/iccad2026contest
    python scripts/analyze_results.py results/boundary_full.json --contest-dir external/FloorSet/iccad2026contest --write-enriched results/enriched_full.json
    python scripts/analyze_results.py results/boundary_full.json --write-focus-json results/focus_cases.json
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = ROOT / "results" / "boundary_full.json"
RANGES = [(21, 40), (41, 60), (61, 80), (81, 100), (101, 120)]
ALPHA = 0.5
BETA = 2.0
ENRICHED_DIAGNOSTIC_FIELDS = (
    "boundary_violations",
    "grouping_violations",
    "mib_violations",
    "total_soft_violations",
    "max_possible_violations",
    "recomputed_violations_relative",
    "constraint_fixed_blocks",
    "constraint_preplaced_blocks",
    "constraint_boundary_blocks",
    "constraint_boundary_codes",
    "constraint_cluster_blocks",
    "constraint_cluster_groups",
    "constraint_mib_blocks",
    "constraint_mib_groups",
    "b2b_edges",
    "p2b_edges",
)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _block_weight(block_count: int, max_blocks: int) -> float:
    return math.exp((block_count - max_blocks) / 12)


def _get_any(case: dict[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in case:
            return case[name]
    details = case.get("violations") or case.get("violation_counts") or case.get("soft_violations") or {}
    if isinstance(details, dict):
        for name in names:
            if name in details:
                return details[name]
    return default


def _violation(case: dict[str, Any], kind: str) -> Any:
    aliases = {
        "boundary": ["boundary_violations", "boundary", "num_boundary_violations"],
        "grouping": ["grouping_violations", "grouping", "cluster_violations", "num_grouping_violations"],
        "mib": ["mib_violations", "mib", "num_mib_violations"],
    }
    value = _get_any(case, aliases[kind], None)
    return "N/A" if value is None else value


def _extract_gt_positions(polygons: Any, block_count: int) -> list[tuple[float, float, float, float]]:
    positions = []
    for i in range(block_count):
        block = polygons[i]
        valid = block[block[:, 0] != -1]
        if len(valid) > 0:
            x_min, y_min = valid.min(dim=0).values
            x_max, y_max = valid.max(dim=0).values
            positions.append((float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)))
        else:
            positions.append((0.0, 0.0, 1.0, 1.0))
    return positions


def _cell(value: Any) -> float:
    """Return a numeric scalar from tensors, numpy values, or plain numbers."""

    if hasattr(value, "item"):
        value = value.item()
    return _num(value)


def _matrix_rows(matrix: Any, rows: int | None = None) -> list[list[float]]:
    if matrix is None:
        return []
    if hasattr(matrix, "detach"):
        data = matrix.detach().cpu().tolist()
    elif hasattr(matrix, "tolist"):
        data = matrix.tolist()
    else:
        data = matrix
    if data is None:
        return []
    out = []
    limit = len(data) if rows is None else min(rows, len(data))
    for idx in range(limit):
        row = data[idx]
        if not isinstance(row, (list, tuple)):
            row = [row]
        out.append([_cell(value) for value in row])
    return out


def _valid_edge_count(edges: Any) -> int:
    return sum(1 for row in _matrix_rows(edges) if row and int(row[0]) != -1)


def add_constraint_profile(
    case: dict[str, Any],
    constraints: Any,
    b2b_connectivity: Any = None,
    p2b_connectivity: Any = None,
) -> None:
    """Attach public-safe structural counts for the case inputs.

    These fields explain why a high-weight case is difficult without exposing
    private solution details: fixed/preplaced blocks, boundary demand, cluster
    and MIB group pressure, and net counts.
    """

    block_count = int(_num(case.get("block_count")))
    rows = _matrix_rows(constraints, block_count)
    if not rows:
        return

    def nonzero_count(col: int) -> int:
        return sum(1 for row in rows if len(row) > col and int(row[col]) != 0)

    def positive_values(col: int) -> list[int]:
        return [int(row[col]) for row in rows if len(row) > col and int(row[col]) > 0]

    boundary_codes = positive_values(4)
    cluster_ids = positive_values(3)
    mib_ids = positive_values(2)
    case["constraint_fixed_blocks"] = nonzero_count(0)
    case["constraint_preplaced_blocks"] = nonzero_count(1)
    case["constraint_mib_blocks"] = len(mib_ids)
    case["constraint_mib_groups"] = len(set(mib_ids))
    case["constraint_cluster_blocks"] = len(cluster_ids)
    case["constraint_cluster_groups"] = len(set(cluster_ids))
    case["constraint_boundary_blocks"] = len(boundary_codes)
    code_counts: dict[str, int] = {}
    for code in boundary_codes:
        code_counts[str(code)] = code_counts.get(str(code), 0) + 1
    case["constraint_boundary_codes"] = dict(sorted(code_counts.items(), key=lambda item: int(item[0])))
    if b2b_connectivity is not None:
        case["b2b_edges"] = _valid_edge_count(b2b_connectivity)
    if p2b_connectivity is not None:
        case["p2b_edges"] = _valid_edge_count(p2b_connectivity)


def _load_official_module(contest_dir: Path) -> Any:
    evaluator_path = contest_dir / "iccad2026_evaluate.py"
    if not evaluator_path.exists():
        raise SystemExit(f"Cannot find official evaluator at {evaluator_path}")
    sys.path.insert(0, str(contest_dir))
    sys.path.insert(0, str(contest_dir.parent))
    spec = importlib.util.spec_from_file_location("official_iccad2026_evaluate", evaluator_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot import official evaluator at {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Official evaluator dependency is missing ({exc.name}). "
            "Run this command with the same Python environment used for contest evaluation."
        ) from exc
    return module


def annotate_with_official_constraint_profile(
    cases: list[dict[str, Any]],
    contest_dir: Path,
    data_path: Path | None = None,
) -> None:
    official = _load_official_module(contest_dir)
    dataset_root = data_path if data_path is not None else contest_dir.parent

    with contextlib.redirect_stdout(sys.stderr):
        dataset = official.FloorplanDatasetLiteTest(str(dataset_root))

    annotated = 0
    for case in cases:
        if case.get("error") is not None:
            continue
        test_id = int(_num(case.get("test_id"), -1))
        if test_id < 0:
            continue
        sample = dataset[test_id]
        area_target, b2b_conn, p2b_conn, _pins_pos, constraints = sample["input"]
        block_count = int(_num(case.get("block_count"))) or int((area_target != -1).sum().item())
        case["block_count"] = block_count
        add_constraint_profile(case, constraints, b2b_conn, p2b_conn)
        annotated += 1

    print(f"Annotated constraint profiles for {annotated}/{len(cases)} cases using {contest_dir}", file=sys.stderr)


def enrich_with_official_soft_counts(
    cases: list[dict[str, Any]],
    contest_dir: Path,
    data_path: Path | None = None,
) -> None:
    """Recompute official soft-violation attribution for saved positions.

    The public result JSON stores aggregate violation ratios but not the
    boundary/grouping/MIB numerator split.  When the official contest checkout
    and validation data are available, this function loads each matching case
    and reuses the official evaluator to recover those counts without rerunning
    the optimizer.
    """

    official = _load_official_module(contest_dir)
    dataset_root = data_path if data_path is not None else contest_dir.parent

    with contextlib.redirect_stdout(sys.stderr):
        dataset = official.FloorplanDatasetLiteTest(str(dataset_root))
        evaluator = official.ContestEvaluator(str(dataset_root), verbose=False)

    median_runtime = 1.0
    runtimes = sorted(_num(c.get("runtime_seconds")) for c in cases if c.get("error") is None)
    if runtimes:
        median_runtime = runtimes[len(runtimes) // 2]

    enriched = 0
    for case in cases:
        if case.get("error") is not None:
            continue
        test_id = int(_num(case.get("test_id"), -1))
        if test_id < 0:
            continue
        sample = dataset[test_id]
        inputs, labels = sample["input"], sample["label"]
        area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
        add_constraint_profile(case, constraints, b2b_conn, p2b_conn)
        if "positions" not in case:
            continue
        polygons, _ = labels
        block_count = int(_num(case.get("block_count")))
        baseline, target_pos = evaluator._extract_baseline(test_id, labels, b2b_conn, p2b_conn, pins_pos, block_count)
        positions = [tuple(float(v) for v in p) for p in case["positions"]]
        metrics = official.evaluate_solution(
            {"positions": positions, "runtime": _num(case.get("runtime_seconds"), 1.0)},
            baseline,
            constraints,
            b2b_conn,
            p2b_conn,
            pins_pos,
            area_target,
            target_pos or _extract_gt_positions(polygons, block_count),
            median_runtime=median_runtime,
        )
        case["boundary_violations"] = metrics.boundary_violations
        case["grouping_violations"] = metrics.grouping_violations
        case["mib_violations"] = metrics.mib_violations
        case["total_soft_violations"] = metrics.total_soft_violations
        case["max_possible_violations"] = metrics.max_possible_violations
        case["recomputed_violations_relative"] = metrics.violations_relative
        enriched += 1

    print(f"Enriched soft-violation counts for {enriched}/{len(cases)} cases using {contest_dir}", file=sys.stderr)


def write_enriched_result(original: dict[str, Any], cases: list[dict[str, Any]], output_path: Path, source_path: Path) -> None:
    """Write a result JSON copy with recomputed per-case soft counts.

    The published baseline result should remain immutable unless a new solver
    result is verified.  This helper writes a separate diagnostic artifact that
    preserves the original score and summary while adding attribution fields to
    each case.
    """

    enriched = dict(original)
    enriched["test_results"] = cases
    diagnostics = dict(enriched.get("diagnostics") or {})
    diagnostics["enriched_soft_counts"] = {
        "source_result": str(source_path),
        "fields": [
            "boundary_violations",
            "grouping_violations",
            "mib_violations",
            "total_soft_violations",
            "max_possible_violations",
            "recomputed_violations_relative",
            "constraint_fixed_blocks",
            "constraint_preplaced_blocks",
            "constraint_boundary_blocks",
            "constraint_boundary_codes",
            "constraint_cluster_blocks",
            "constraint_cluster_groups",
            "constraint_mib_blocks",
            "constraint_mib_groups",
            "b2b_edges",
            "p2b_edges",
        ],
    }
    enriched["diagnostics"] = diagnostics
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(enriched, f, indent=2)
        f.write("\n")


def load_result(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if "test_results" not in data or not isinstance(data["test_results"], list):
        raise SystemExit(f"No test_results list found in {path}")
    return data


def _source_matches_result(source: str, result_path: Path) -> bool:
    """Return True when a diagnostic sidecar names this result file."""

    source_path = Path(source)
    result_norm = Path(result_path)
    candidates = {
        str(result_norm),
        str(result_norm.as_posix()),
        result_norm.name,
    }
    with contextlib.suppress(ValueError):
        candidates.add(str(result_norm.resolve().relative_to(ROOT)))
        candidates.add(result_norm.resolve().relative_to(ROOT).as_posix())
    return source in candidates or source_path.name == result_norm.name


def _compatible_sidecar(base: dict[str, Any], sidecar: dict[str, Any], result_path: Path) -> bool:
    diagnostics = sidecar.get("diagnostics") or {}
    enriched = diagnostics.get("enriched_soft_counts") or {}
    source = enriched.get("source_result")
    if source and not _source_matches_result(str(source), result_path):
        return False
    if not math.isclose(_num(base.get("total_score")), _num(sidecar.get("total_score")), rel_tol=0.0, abs_tol=1e-9):
        return False
    base_cases = base.get("test_results") or []
    sidecar_cases = sidecar.get("test_results") or []
    if len(base_cases) != len(sidecar_cases):
        return False
    for idx, (base_case, sidecar_case) in enumerate(zip(base_cases, sidecar_cases)):
        base_id = int(_num(base_case.get("test_id"), idx))
        sidecar_id = int(_num(sidecar_case.get("test_id"), idx))
        if base_id != sidecar_id:
            return False
        if not math.isclose(_num(base_case.get("cost")), _num(sidecar_case.get("cost")), rel_tol=0.0, abs_tol=1e-9):
            return False
    return True


def _default_sidecar_candidates(result_path: Path) -> list[Path]:
    return [
        result_path.with_name("enriched_diagnostics.json"),
        ROOT / "results" / "enriched_diagnostics.json",
    ]


def merge_enriched_sidecar(
    data: dict[str, Any],
    cases: list[dict[str, Any]],
    result_path: Path,
    sidecar_path: Path | None = None,
) -> tuple[int, Path | None]:
    """Merge matching diagnostic sidecar fields into in-memory case rows.

    The official full-result JSON is kept as the source of score truth.  This
    only copies derived diagnostic fields from a compatible enriched artifact so
    the default report can show soft-constraint attribution.
    """

    candidates = [sidecar_path] if sidecar_path is not None else _default_sidecar_candidates(result_path)
    for candidate in candidates:
        if candidate is None or not candidate.exists() or candidate.resolve() == result_path.resolve():
            continue
        sidecar = load_result(candidate)
        if not _compatible_sidecar(data, sidecar, result_path):
            continue
        by_id = {
            int(_num(case.get("test_id"), idx)): case
            for idx, case in enumerate(sidecar["test_results"])
        }
        merged = 0
        for idx, case in enumerate(cases):
            test_id = int(_num(case.get("test_id"), idx))
            enriched_case = by_id.get(test_id)
            if not enriched_case:
                continue
            changed = False
            for field in ENRICHED_DIAGNOSTIC_FIELDS:
                if field in enriched_case and (field not in case or case[field] is None):
                    case[field] = enriched_case[field]
                    changed = True
            if changed:
                merged += 1
        return merged, candidate
    return 0, None


def add_weights(cases: list[dict[str, Any]]) -> None:
    if not cases:
        return
    max_blocks = max(int(_num(c.get("block_count"))) for c in cases)
    raw_weights = [_block_weight(int(_num(c.get("block_count"))), max_blocks) for c in cases]
    total_weight = sum(raw_weights) or 1.0
    for case, weight in zip(cases, raw_weights):
        norm = weight / total_weight
        case["_score_weight"] = norm
        case["_weighted_contribution"] = _num(case.get("cost")) * norm


def fmt_weighted(value: float) -> str:
    """Print score contributions without rounding small cases to zero."""

    if abs(value) < 0.0001 and value != 0.0:
        return f"{value:.3e}"
    return f"{value:.6f}"


def fmt_case(case: dict[str, Any]) -> str:
    constraint_bits = ""
    if "constraint_boundary_blocks" in case:
        constraint_bits = (
            f" constraints=boundary:{case.get('constraint_boundary_blocks')}"
            f" cluster:{case.get('constraint_cluster_blocks')}/{case.get('constraint_cluster_groups')}"
            f" mib:{case.get('constraint_mib_blocks')}/{case.get('constraint_mib_groups')}"
            f" preplaced:{case.get('constraint_preplaced_blocks')}"
            f" nets:{case.get('b2b_edges', 'N/A')}/{case.get('p2b_edges', 'N/A')}"
        )
    return (
        f"test_id={case.get('test_id'):>3} "
        f"blocks={case.get('block_count'):>3} "
        f"cost={_num(case.get('cost')):7.4f} "
        f"weighted={fmt_weighted(_num(case.get('_weighted_contribution'))):>9} "
        f"hpwl={_num(case.get('hpwl_gap')):7.4f} "
        f"area={_num(case.get('area_gap')):7.4f} "
        f"soft={_num(case.get('violations_relative')):7.4f} "
        f"boundary={_violation(case, 'boundary')} "
        f"grouping={_violation(case, 'grouping')} "
        f"mib={_violation(case, 'mib')} "
        f"soft_count={_get_any(case, ['total_soft_violations'], 'N/A')}/"
        f"{_get_any(case, ['max_possible_violations'], 'N/A')} "
        f"runtime={_num(case.get('runtime_seconds')):6.3f}s"
        f"{constraint_bits}"
    )


def print_cases(title: str, cases: list[dict[str, Any]], top: int) -> None:
    print(f"\n## {title}")
    for case in cases[:top]:
        print("- " + fmt_case(case))


def range_label(block_count: int) -> str:
    for lo, hi in RANGES:
        if lo <= block_count <= hi:
            return f"{lo}-{hi}"
    return "other"


def avg(items: list[float]) -> float:
    return mean(items) if items else 0.0


def print_aggregates(cases: list[dict[str, Any]]) -> None:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        buckets[range_label(int(_num(case.get("block_count"))))].append(case)

    print("\n## Aggregate averages by block-count range")
    total_contribution = sum(_num(c.get("_weighted_contribution")) for c in cases) or 1.0
    total_weight = sum(_num(c.get("_score_weight")) for c in cases) or 1.0
    for lo, hi in RANGES:
        label = f"{lo}-{hi}"
        group = buckets.get(label, [])
        if not group:
            print(f"- {label}: no cases")
            continue
        feasible = sum(1 for c in group if c.get("is_feasible"))
        score_contribution = sum(_num(c.get("_weighted_contribution")) for c in group)
        weight_share = sum(_num(c.get("_score_weight")) for c in group) / total_weight
        score_share = score_contribution / total_contribution
        worst_weighted = max(group, key=lambda c: _num(c.get("_weighted_contribution")))
        print(
            f"- {label}: cases={len(group)}, feasible={feasible}/{len(group)}, "
            f"avg_cost={avg([_num(c.get('cost')) for c in group]):.4f}, "
            f"score_contribution={fmt_weighted(score_contribution)}, "
            f"score_share={score_share:.2%}, weight_share={weight_share:.2%}, "
            f"top_weighted_case={worst_weighted.get('test_id')}, "
            f"avg_hpwl={avg([_num(c.get('hpwl_gap')) for c in group]):.4f}, "
            f"avg_area={avg([_num(c.get('area_gap')) for c in group]):.4f}, "
            f"avg_soft={avg([_num(c.get('violations_relative')) for c in group]):.4f}, "
            f"avg_runtime={avg([_num(c.get('runtime_seconds')) for c in group]):.3f}s"
        )


def score_concentration(cases: list[dict[str, Any]], cutoffs: Iterable[int] = (1, 3, 5, 10, 20)) -> list[dict[str, Any]]:
    """Summarize how much total score is concentrated in the highest-weight cases."""

    ordered = sorted(cases, key=lambda c: _num(c.get("_score_weight")), reverse=True)
    total_weight = sum(_num(c.get("_score_weight")) for c in ordered) or 1.0
    total_contribution = sum(_num(c.get("_weighted_contribution")) for c in ordered) or 1.0
    rows = []
    for cutoff in cutoffs:
        if cutoff <= 0:
            continue
        selected = ordered[: min(cutoff, len(ordered))]
        if not selected:
            continue
        rows.append(
            {
                "top_n": len(selected),
                "test_ids": [case.get("test_id") for case in selected],
                "weight_share": sum(_num(c.get("_score_weight")) for c in selected) / total_weight,
                "score_share": sum(_num(c.get("_weighted_contribution")) for c in selected) / total_contribution,
                "avg_cost": avg([_num(c.get("cost")) for c in selected]),
                "avg_hpwl": avg([_num(c.get("hpwl_gap")) for c in selected]),
                "avg_area": avg([_num(c.get("area_gap")) for c in selected]),
                "avg_soft": avg([_num(c.get("violations_relative")) for c in selected]),
            }
        )
    return rows


def print_score_concentration(cases: list[dict[str, Any]]) -> None:
    print("\n## Score concentration")
    for row in score_concentration(cases):
        ids = ",".join(str(test_id) for test_id in row["test_ids"])
        print(
            f"- top_{row['top_n']}: test_ids={ids}, "
            f"weight_share={row['weight_share']:.2%}, "
            f"score_share={row['score_share']:.2%}, "
            f"avg_cost={row['avg_cost']:.4f}, "
            f"avg_hpwl={row['avg_hpwl']:.4f}, "
            f"avg_area={row['avg_area']:.4f}, "
            f"avg_soft={row['avg_soft']:.4f}"
        )


def print_constraint_profile(cases: list[dict[str, Any]], top: int) -> None:
    profiled = [case for case in cases if "constraint_boundary_blocks" in case]
    if not profiled:
        return
    ordered = sorted(profiled, key=lambda c: _num(c.get("_weighted_contribution")), reverse=True)
    focus = ordered[: max(1, min(top, len(ordered)))]
    print("\n## Constraint profile for weighted focus cases")
    print(
        f"- focus_cases={len(focus)}, "
        f"avg_boundary_blocks={avg([_num(c.get('constraint_boundary_blocks')) for c in focus]):.2f}, "
        f"avg_cluster_blocks={avg([_num(c.get('constraint_cluster_blocks')) for c in focus]):.2f}, "
        f"avg_cluster_groups={avg([_num(c.get('constraint_cluster_groups')) for c in focus]):.2f}, "
        f"avg_mib_blocks={avg([_num(c.get('constraint_mib_blocks')) for c in focus]):.2f}, "
        f"avg_preplaced={avg([_num(c.get('constraint_preplaced_blocks')) for c in focus]):.2f}, "
        f"avg_b2b_edges={avg([_num(c.get('b2b_edges')) for c in focus]):.1f}, "
        f"avg_p2b_edges={avg([_num(c.get('p2b_edges')) for c in focus]):.1f}"
    )
    boundary_codes: dict[str, float] = defaultdict(float)
    for case in focus:
        for code, count in (case.get("constraint_boundary_codes") or {}).items():
            boundary_codes[str(code)] += _num(count)
    if boundary_codes:
        ordered_codes = sorted(boundary_codes.items(), key=lambda item: (-item[1], int(item[0])))
        print(
            "- boundary_code_totals="
            + ", ".join(f"{code}:{int(count)}" for code, count in ordered_codes)
        )


def dominant_block_range(cases: list[dict[str, Any]]) -> tuple[str, float]:
    """Return the block-count range with the largest score contribution."""

    if not cases:
        return ("none", 0.0)
    totals: dict[str, float] = defaultdict(float)
    for case in cases:
        totals[range_label(int(_num(case.get("block_count"))))] += _num(case.get("_weighted_contribution"))
    label, contribution = max(totals.items(), key=lambda item: item[1])
    total = sum(totals.values()) or 1.0
    return label, contribution / total


def print_soft_totals(cases: list[dict[str, Any]]) -> None:
    if not cases or any(_violation(cases[0], k) == "N/A" for k in ("boundary", "grouping", "mib")):
        return
    boundary = sum(int(_num(_violation(case, "boundary"))) for case in cases)
    grouping = sum(int(_num(_violation(case, "grouping"))) for case in cases)
    mib = sum(int(_num(_violation(case, "mib"))) for case in cases)
    total = sum(int(_num(case.get("total_soft_violations"))) for case in cases)
    max_possible = sum(int(_num(case.get("max_possible_violations"))) for case in cases)
    print("\n## Soft-constraint violation totals")
    print(
        f"- boundary={boundary}, grouping={grouping}, mib={mib}, "
        f"total={total}/{max_possible}"
    )


def _quality_factor(case: dict[str, Any]) -> float:
    return 1.0 + ALPHA * (max(0.0, _num(case.get("hpwl_gap"))) + max(0.0, _num(case.get("area_gap"))))


def metric_pressure(cases: list[dict[str, Any]]) -> dict[str, float]:
    """Estimate reconstructed-score reduction from small metric improvements.

    For feasible cases, official cost is:
    (1 + ALPHA * (hpwl + area)) * exp(BETA * soft) * runtime_adjustment.
    Using the published per-case cost avoids reconstructing median-runtime
    details while still giving the local derivative of score with respect to
    HPWL, area, and soft-violation ratio.
    """

    totals = {
        "hpwl_per_0_1": 0.0,
        "area_per_0_1": 0.0,
        "soft_per_0_01": 0.0,
    }
    for case in cases:
        weight = _num(case.get("_score_weight"))
        cost = _num(case.get("cost"))
        quality = max(_quality_factor(case), 1e-12)
        totals["hpwl_per_0_1"] += weight * cost * (ALPHA / quality) * 0.1
        totals["area_per_0_1"] += weight * cost * (ALPHA / quality) * 0.1
        totals["soft_per_0_01"] += weight * cost * BETA * 0.01
    return totals


def soft_driver_pressure(cases: list[dict[str, Any]]) -> dict[str, float]:
    """Return score-weighted soft-violation counts by violation family."""

    totals = {"boundary": 0.0, "grouping": 0.0, "mib": 0.0}
    for case in cases:
        for kind in totals:
            value = _violation(case, kind)
            if value != "N/A":
                totals[kind] += _num(case.get("_score_weight")) * _num(value)
    return totals


def sensitivity_rows(cases: list[dict[str, Any]], top: int) -> list[str]:
    rows = []
    for case in cases:
        quality = max(_quality_factor(case), 1e-12)
        weight = _num(case.get("_score_weight"))
        cost = _num(case.get("cost"))
        quality_gain_0_1 = weight * cost * (ALPHA / quality) * 0.1
        soft_gain_0_01 = weight * cost * BETA * 0.01
        rows.append(
            {
                "case": case,
                "quality_gain_0_1": quality_gain_0_1,
                "soft_gain_0_01": soft_gain_0_01,
                "dominant_gain": max(quality_gain_0_1, soft_gain_0_01),
            }
        )
    rows.sort(key=lambda row: row["dominant_gain"], reverse=True)
    formatted = []
    for row in rows[:top]:
        case = row["case"]
        formatted.append(
            f"test_id={case.get('test_id'):>3} blocks={case.get('block_count'):>3} "
            f"score_weight={_num(case.get('_score_weight')):.6f} cost={_num(case.get('cost')):.4f} "
            f"gain_if_hpwl_or_area_-0.1={fmt_weighted(row['quality_gain_0_1'])} "
            f"gain_if_soft_-0.01={fmt_weighted(row['soft_gain_0_01'])} "
            f"hpwl={_num(case.get('hpwl_gap')):.4f} area={_num(case.get('area_gap')):.4f} "
            f"soft={_num(case.get('violations_relative')):.4f}"
        )
    return formatted


def print_metric_pressure(cases: list[dict[str, Any]], top: int) -> None:
    pressure = metric_pressure(cases)
    print("\n## Weighted metric pressure")
    print(
        "- Estimated total-score reduction if every case improved by: "
        f"HPWL -0.1 => {fmt_weighted(pressure['hpwl_per_0_1'])}, "
        f"area -0.1 => {fmt_weighted(pressure['area_per_0_1'])}, "
        f"soft ratio -0.01 => {fmt_weighted(pressure['soft_per_0_01'])}"
    )
    soft_pressure = soft_driver_pressure(cases)
    if any(value > 0 for value in soft_pressure.values()):
        ordered = sorted(soft_pressure.items(), key=lambda item: item[1], reverse=True)
        print(
            "- Score-weighted soft counts: "
            + ", ".join(f"{name}={value:.3f}" for name, value in ordered)
        )
    print("- Highest-impact local sensitivities:")
    for row in sensitivity_rows(cases, top):
        print(f"  {row}")


def compact_case(case: dict[str, Any]) -> dict[str, Any]:
    """Return stable, compact fields for focus-case JSON artifacts."""

    fields = [
        "test_id",
        "block_count",
        "cost",
        "hpwl_gap",
        "area_gap",
        "violations_relative",
        "runtime_seconds",
        "_score_weight",
        "_weighted_contribution",
        "boundary_violations",
        "grouping_violations",
        "mib_violations",
        "total_soft_violations",
        "max_possible_violations",
        "constraint_fixed_blocks",
        "constraint_preplaced_blocks",
        "constraint_boundary_blocks",
        "constraint_boundary_codes",
        "constraint_cluster_blocks",
        "constraint_cluster_groups",
        "constraint_mib_blocks",
        "constraint_mib_groups",
        "b2b_edges",
        "p2b_edges",
    ]
    out: dict[str, Any] = {}
    for field in fields:
        if field not in case:
            continue
        value = case[field]
        if isinstance(value, float):
            out[field] = round(value, 12)
        else:
            out[field] = value
    return out


def focus_report(data: dict[str, Any], cases: list[dict[str, Any]], source_path: Path, top: int) -> dict[str, Any]:
    """Build a compact repeatable planning artifact for score experiments."""

    weighted = sorted(cases, key=lambda c: _num(c.get("_weighted_contribution")), reverse=True)
    sensitivity = sorted(
        cases,
        key=lambda c: max(
            _num(c.get("_score_weight")) * _num(c.get("cost")) * (ALPHA / max(_quality_factor(c), 1e-12)) * 0.1,
            _num(c.get("_score_weight")) * _num(c.get("cost")) * BETA * 0.01,
        ),
        reverse=True,
    )
    range_name, range_share = dominant_block_range(cases)
    return {
        "source_result": str(source_path),
        "total_score": data.get("total_score"),
        "summary": data.get("summary") or {},
        "dominant_score_range": {"range": range_name, "score_share": round(range_share, 12)},
        "score_concentration": score_concentration(cases),
        "metric_pressure": {key: round(value, 12) for key, value in metric_pressure(cases).items()},
        "soft_driver_pressure": {
            key: round(value, 12) for key, value in soft_driver_pressure(cases).items()
        },
        "recommendation": recommendation(cases),
        "top_weighted_cases": [compact_case(case) for case in weighted[:top]],
        "top_sensitivity_cases": [compact_case(case) for case in sensitivity[:top]],
    }


def write_focus_report(data: dict[str, Any], cases: list[dict[str, Any]], output_path: Path, source_path: Path, top: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(focus_report(data, cases, source_path, top), f, indent=2)
        f.write("\n")


def recommendation(cases: list[dict[str, Any]]) -> str:
    weighted = sorted(cases, key=lambda c: _num(c.get("_weighted_contribution")), reverse=True)
    focus = weighted[: max(1, min(20, len(weighted)))]
    hpwl = avg([_num(c.get("hpwl_gap")) for c in focus])
    area = avg([_num(c.get("area_gap")) for c in focus])
    soft = avg([_num(c.get("violations_relative")) for c in focus])
    runtime = avg([_num(c.get("runtime_seconds")) for c in focus])
    boundary = sum(_num(_violation(c, "boundary")) for c in focus if _violation(c, "boundary") != "N/A")
    grouping = sum(_num(_violation(c, "grouping")) for c in focus if _violation(c, "grouping") != "N/A")
    mib = sum(_num(_violation(c, "mib")) for c in focus if _violation(c, "mib") != "N/A")

    # Cost uses (1 + 0.5*(HPWL+area)) * exp(2*soft) * runtime factor.
    # Use this as a simple directional diagnosis rather than a proof.
    quality_term = 0.5 * (hpwl + area)
    soft_term = math.exp(2 * soft) - 1
    if soft_term > quality_term * 1.15:
        if max(boundary, grouping, mib) > 0:
            driver = max((boundary, "boundary"), (grouping, "grouping"), (mib, "MIB"))[1]
            primary = f"soft constraints, led by {driver} violations in weighted cases"
        else:
            primary = "soft constraints, especially grouping/boundary if detailed counts confirm it"
    elif hpwl > area * 1.15:
        primary = "HPWL / connectivity-aware placement"
    elif area > hpwl * 1.15:
        primary = "bounding-box area compaction"
    else:
        primary = "combined HPWL and area improvement, while preserving current soft-constraint gains"

    if runtime > 5.0:
        primary += "; also watch runtime on the largest cases"
    profiled_focus = [c for c in focus if "constraint_boundary_blocks" in c]
    if profiled_focus:
        avg_blocks = avg([_num(c.get("block_count")) for c in profiled_focus])
        avg_boundary = avg([_num(c.get("constraint_boundary_blocks")) for c in profiled_focus])
        avg_cluster = avg([_num(c.get("constraint_cluster_blocks")) for c in profiled_focus])
        avg_b2b = avg([_num(c.get("b2b_edges")) for c in profiled_focus])
        if avg_blocks > 0 and avg_boundary >= 0.25 * avg_blocks:
            primary += "; high boundary density makes perimeter ordering/packing especially important"
        if avg_blocks > 0 and avg_cluster >= 0.20 * avg_blocks:
            primary += "; cluster packing remains a meaningful interaction with area/HPWL"
        if avg_b2b > 3000:
            primary += "; dense B2B connectivity favors connectivity-aware ordering changes"
    range_name, range_share = dominant_block_range(cases)
    return (
        f"Weighted worst-case averages: hpwl={hpwl:.4f}, area={area:.4f}, "
        f"soft={soft:.4f}, runtime={runtime:.3f}s. Dominant score range: "
        f"{range_name} ({range_share:.2%} of reconstructed score). "
        f"Suggested next target: {primary}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json", nargs="?", default=str(DEFAULT_RESULT), help="Path to full evaluator JSON")
    parser.add_argument("--top", type=int, default=20, help="Number of worst cases to print")
    parser.add_argument(
        "--contest-dir",
        type=Path,
        default=None,
        help="Optional official FloorSet/iccad2026contest path; enables boundary/grouping/MIB count reconstruction",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Optional FloorSet data root for --contest-dir; defaults to the official checkout parent",
    )
    parser.add_argument(
        "--write-enriched",
        type=Path,
        default=None,
        help="Write a diagnostic JSON copy containing recomputed boundary/grouping/MIB counts; requires --contest-dir",
    )
    parser.add_argument(
        "--write-focus-json",
        type=Path,
        default=None,
        help="Write a compact JSON planning artifact for the highest-impact weighted cases",
    )
    parser.add_argument(
        "--diagnostic-sidecar",
        type=Path,
        default=None,
        help="Optional enriched diagnostic JSON to merge before reporting soft-constraint attribution",
    )
    parser.add_argument(
        "--no-auto-sidecar",
        action="store_true",
        help="Disable automatic merge from results/enriched_diagnostics.json when it matches the result file",
    )
    args = parser.parse_args()

    path = Path(args.result_json)
    data = load_result(path)
    cases = [dict(c) for c in data["test_results"]]
    if args.write_enriched is not None and args.contest_dir is None:
        raise SystemExit("--write-enriched requires --contest-dir so official soft counts can be reconstructed")
    if args.diagnostic_sidecar is not None or not args.no_auto_sidecar:
        merged, merged_path = merge_enriched_sidecar(
            data,
            cases,
            path,
            args.diagnostic_sidecar,
        )
        if merged_path is not None:
            print(f"Merged diagnostic sidecar fields for {merged}/{len(cases)} cases from {merged_path}", file=sys.stderr)
    if args.contest_dir is not None:
        enrich_with_official_soft_counts(cases, args.contest_dir.resolve(), args.data_path.resolve() if args.data_path else None)
    if args.write_enriched is not None:
        write_enriched_result(data, cases, args.write_enriched, path)
        print(f"Wrote enriched diagnostic JSON: {args.write_enriched}", file=sys.stderr)
    add_weights(cases)
    if args.write_focus_json is not None:
        write_focus_report(data, cases, args.write_focus_json, path, args.top)
        print(f"Wrote focus-case diagnostic JSON: {args.write_focus_json}", file=sys.stderr)

    total_score = data.get("total_score")
    summary = data.get("summary") or {}
    feasible = summary.get("num_feasible", sum(1 for c in cases if c.get("is_feasible")))
    print(f"# FloorSet result analysis")
    print(f"Result file: {path}")
    print(f"Total score: {_num(total_score):.6f}")
    print(f"Feasible: {feasible}/{len(cases)}")

    print_cases("Worst cases by raw cost", sorted(cases, key=lambda c: _num(c.get("cost")), reverse=True), args.top)
    print_cases("Worst cases by weighted contribution", sorted(cases, key=lambda c: _num(c.get("_weighted_contribution")), reverse=True), args.top)
    print_aggregates(cases)
    print_score_concentration(cases)
    print_constraint_profile(cases, min(args.top, 20))
    print_soft_totals(cases)
    print_metric_pressure(cases, min(args.top, 20))
    print("\n## Recommendation")
    print("- " + recommendation(cases))
    if all(_violation(cases[0], k) == "N/A" for k in ("boundary", "grouping", "mib")) if cases else False:
        print("- Note: this result JSON does not include per-case boundary/grouping/MIB counts. Add evaluator instrumentation if exact soft-violation attribution is needed.")


if __name__ == "__main__":
    main()
