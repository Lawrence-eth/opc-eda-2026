#!/usr/bin/env python3
"""Audit exact reconstruction by the dual-parent representation.

The audit consumes only supervised layout labels and contest constraints.  It
does not inspect or infer dataset generators, PRNG state, worker IDs, source
ordering, or instance identity.  A manifest is used solely to locate an
already-frozen source-disjoint evaluation panel.

Example (existing clean fold 0):

    python scripts/audit_dual_parent_oracle.py \
      --data-root /path/to/FloorSet \
      --manifest results/folds/heavy_clean_v1.json --fold 0
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOLUTION_DIR = ROOT / "contest_solution"
sys.path.insert(0, str(SOLUTION_DIR))

from dual_parent_decoder import (  # noqa: E402
    DualParentError,
    compare_geometry,
    decode_dual_parent,
    extract_oracle_labels,
    hard_targets_from_golden,
    training_rectangles,
)


FLOAT_METRICS = (
    "hpwl_b2b",
    "hpwl_p2b",
    "hpwl_total",
    "hpwl_gap",
    "bbox_area",
    "area_gap",
    "violations_relative",
    "cost",
)
INTEGER_METRICS = (
    "overlap_violations",
    "area_violations",
    "dimension_violations",
    "boundary_violations",
    "grouping_violations",
    "mib_violations",
    "total_soft_violations",
    "max_possible_violations",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical_source(path: str | Path, data_root: Path) -> str:
    return Path(path).resolve().relative_to(data_root.resolve()).as_posix()


def _manifest_cases(
    dataset: Any,
    data_root: Path,
    manifest_path: Path,
    folds: set[int] | None,
    max_cases: int | None,
) -> list[tuple[str, int]]:
    manifest = json.loads(manifest_path.read_text())
    by_source = {
        _canonical_source(path, data_root): file_index
        for file_index, path in enumerate(dataset.all_files)
    }
    selected: list[tuple[str, int]] = []
    seen_sources: set[str] = set()
    for fold_manifest in manifest.get("manifests", [manifest]):
        fold = int(fold_manifest.get("fold", 0))
        if folds is not None and fold not in folds:
            continue
        for case in fold_manifest.get("cases", []):
            source = str(case["source_file"])
            offset = int(case["file_offset"])
            if source not in by_source:
                raise ValueError(f"manifest source is absent from dataset: {source}")
            if source in seen_sources:
                raise ValueError(
                    "oracle panel must be source-disjoint; repeated source: " + source
                )
            if not 0 <= offset < int(dataset.layouts_per_file):
                raise ValueError(f"invalid file offset for {source}: {offset}")
            selected.append((source, offset))
            seen_sources.add(source)
            if max_cases is not None and len(selected) >= max_cases:
                return selected
    if not selected:
        raise ValueError("manifest/fold selection produced no cases")
    return selected


def _baseline(stored_metrics: Any) -> dict[str, float]:
    return {
        "area_baseline": float(stored_metrics[0]),
        "hpwl_baseline": float(stored_metrics[-2]) + float(stored_metrics[-1]),
    }


def _metric_delta(actual: Any, expected: Any) -> tuple[float, list[str]]:
    max_delta = 0.0
    mismatches: list[str] = []
    if bool(actual.is_feasible) != bool(expected.is_feasible):
        mismatches.append("is_feasible")
    for name in INTEGER_METRICS:
        if int(getattr(actual, name)) != int(getattr(expected, name)):
            mismatches.append(name)
    for name in FLOAT_METRICS:
        delta = abs(float(getattr(actual, name)) - float(getattr(expected, name)))
        max_delta = max(max_delta, delta)
    return max_delta, mismatches


def _summary_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "max": ordered[-1],
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    data_root = args.data_root.resolve()
    if not (data_root / "floorset_lite").is_dir():
        raise SystemExit(f"FloorSet training data is absent: {data_root}")
    sys.path.insert(0, str(data_root))
    dataset_module = _load_module(data_root / "lite_dataset.py", "_oracle_lite_dataset")
    evaluator = _load_module(
        data_root / "iccad2026contest" / "iccad2026_evaluate.py",
        "_oracle_official_evaluator",
    )
    dataset = dataset_module.FloorplanDatasetLite(str(data_root))
    dataset.all_files = sorted(str(Path(path).resolve()) for path in dataset.all_files)
    source_to_index = {
        _canonical_source(path, data_root): index
        for index, path in enumerate(dataset.all_files)
    }
    folds = set(args.fold) if args.fold else None
    selected = _manifest_cases(
        dataset, data_root, args.manifest, folds, args.max_cases
    )

    failure_counts: Counter[str] = Counter()
    failure_examples: list[dict[str, Any]] = []
    geometry_exact = 0
    metrics_exact = 0
    strict_mib_decodable = 0
    golden_feasible = 0
    blocks = 0
    horizontal_edges = 0
    floor_supports = 0
    block_supports = 0
    mib_inconsistent_layouts = 0
    max_coordinate_delta = 0.0
    max_dimension_delta = 0.0
    max_metric_delta = 0.0
    shape_option_counts: list[float] = []
    decoded_costs: list[float] = []
    golden_costs: list[float] = []
    block_counts: list[int] = []

    for ordinal, (source, offset) in enumerate(selected):
        file_index = source_to_index[source]
        sample_index = file_index * int(dataset.layouts_per_file) + offset
        sample = dataset[sample_index]
        try:
            area_targets, b2b, p2b, pins, constraints = sample["input"]
            tree, fp_solution, stored_metrics = sample["label"]
            block_count = int((area_targets != -1).sum().item())
            golden = training_rectangles(fp_solution, block_count)
            labels = extract_oracle_labels(
                area_targets,
                constraints,
                tree,
                golden,
                max_aspect_ratio=args.max_aspect_ratio,
            )
            hard_targets = hard_targets_from_golden(constraints, golden)
            strict_mib = True
            try:
                decoded = decode_dual_parent(
                    labels,
                    constraints=constraints,
                    hard_targets=hard_targets,
                )
            except DualParentError as exc:
                if exc.code != "mib_shape_mismatch":
                    raise
                strict_mib = False
                decoded = decode_dual_parent(
                    labels,
                    constraints=constraints,
                    hard_targets=hard_targets,
                    enforce_mib=False,
                )

            comparison = compare_geometry(decoded, golden)
            max_coordinate_delta = max(
                max_coordinate_delta, comparison.max_coordinate_delta
            )
            max_dimension_delta = max(
                max_dimension_delta, comparison.max_dimension_delta
            )
            if comparison.is_exact(args.geometry_tolerance):
                geometry_exact += 1
            else:
                failure_counts["geometry_mismatch"] += 1
                failure_examples.append(
                    {
                        "ordinal": ordinal,
                        "code": "geometry_mismatch",
                        "comparison": asdict(comparison),
                    }
                )

            baseline = _baseline(stored_metrics)
            golden_metrics = evaluator.evaluate_solution(
                {"positions": golden, "runtime": 1.0},
                baseline,
                constraints,
                b2b,
                p2b,
                pins,
                area_targets,
                golden,
                median_runtime=1.0,
            )
            decoded_metrics = evaluator.evaluate_solution(
                {"positions": decoded, "runtime": 1.0},
                baseline,
                constraints,
                b2b,
                p2b,
                pins,
                area_targets,
                golden,
                median_runtime=1.0,
            )
            metric_delta, integer_mismatches = _metric_delta(
                decoded_metrics, golden_metrics
            )
            max_metric_delta = max(max_metric_delta, metric_delta)
            if metric_delta <= args.metric_tolerance and not integer_mismatches:
                metrics_exact += 1
            else:
                failure_counts["metric_mismatch"] += 1
                failure_examples.append(
                    {
                        "ordinal": ordinal,
                        "code": "metric_mismatch",
                        "max_float_delta": metric_delta,
                        "integer_fields": integer_mismatches,
                    }
                )

            strict_mib_decodable += int(strict_mib)
            golden_feasible += int(golden_metrics.is_feasible)
            mib_inconsistent_layouts += int(bool(labels.mib_inconsistent_groups))
            blocks += block_count
            horizontal_edges += len(labels.horizontal)
            floor_supports += sum(
                support is None for support in labels.vertical_supports
            )
            block_supports += sum(
                support is not None for support in labels.vertical_supports
            )
            constraint_rows = constraints.detach().cpu().tolist()
            for index, options in enumerate(labels.shape_options):
                row = constraint_rows[index]
                hard = row[0] != 0.0 or row[1] != 0.0
                if not hard:
                    shape_option_counts.append(float(len(options)))
            decoded_costs.append(float(decoded_metrics.cost))
            golden_costs.append(float(golden_metrics.cost))
            block_counts.append(block_count)
        except DualParentError as exc:
            failure_counts[exc.code] += 1
            if len(failure_examples) < 20:
                failure_examples.append(
                    {"ordinal": ordinal, "code": exc.code, "detail": exc.detail}
                )
        except Exception as exc:  # keep a stable audit artifact on bad data/API drift
            failure_counts["audit_exception"] += 1
            if len(failure_examples) < 20:
                failure_examples.append(
                    {
                        "ordinal": ordinal,
                        "code": "audit_exception",
                        "detail": f"{type(exc).__name__}: {exc}",
                    }
                )

        if args.progress_every and (ordinal + 1) % args.progress_every == 0:
            print(f"audited {ordinal + 1}/{len(selected)}", file=sys.stderr)

    cases = len(selected)
    representation_failures = sum(failure_counts.values())
    gate_passed = (
        representation_failures == 0
        and geometry_exact == cases
        and metrics_exact == cases
        and strict_mib_decodable == cases
        and golden_feasible == cases
    )
    result = {
        "schema_version": 1,
        "audit": "dual_parent_oracle_reconstruction",
        "contract": {
            "decoder_inputs": [
                "exact oriented factor-pair shape category",
                "horizontal B*-tree parent and side",
                "vertical support parent or floor",
                "fixed/preplaced targets and MIB groups",
            ],
            "prohibited_inputs": [
                "generator implementation or parameters",
                "PRNG state",
                "source/file/worker identity as a model feature",
                "instance lookup or retrieval",
            ],
            "metric_runtime_seconds": 1.0,
        },
        "provenance": {
            "manifest": args.manifest.as_posix(),
            "manifest_sha256": _sha256(args.manifest),
            "folds": sorted(folds) if folds is not None else "all",
            "official_evaluator": "iccad2026contest/iccad2026_evaluate.py",
            "official_evaluator_sha256": _sha256(
                data_root / "iccad2026contest" / "iccad2026_evaluate.py"
            ),
            "decoder_sha256": _sha256(
                SOLUTION_DIR / "dual_parent_decoder.py"
            ),
            "audit_script_sha256": _sha256(Path(__file__)),
            "source_selection": "pre-existing manifest; source-disjoint",
        },
        "summary": {
            "oracle_gate_passed": gate_passed,
            "cases": cases,
            "unique_sources": len({source for source, _offset in selected}),
            "blocks": blocks,
            "golden_feasible": golden_feasible,
            "geometry_exact": geometry_exact,
            "metrics_exact": metrics_exact,
            "strict_mib_decodable": strict_mib_decodable,
            "mib_inconsistent_layouts": mib_inconsistent_layouts,
            "horizontal_edges_verified": horizontal_edges,
            "floor_support_labels": floor_supports,
            "block_support_labels": block_supports,
            "max_coordinate_delta": max_coordinate_delta,
            "max_dimension_delta": max_dimension_delta,
            "max_metric_delta": max_metric_delta,
            "decoded_score_rf1": evaluator.compute_total_score(
                decoded_costs, block_counts
            ) if decoded_costs else None,
            "golden_score_rf1": evaluator.compute_total_score(
                golden_costs, block_counts
            ) if golden_costs else None,
            "soft_shape_option_count": _summary_stats(shape_option_counts),
            "failure_taxonomy": dict(sorted(failure_counts.items())),
        },
        "failure_examples": failure_examples[:20],
    }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "external" / "FloorSet",
        help="FloorSet checkout containing floorset_lite and contest evaluator",
    )
    parser.add_argument(
        "--manifest", type=Path, required=True, help="existing frozen fold manifest"
    )
    parser.add_argument(
        "--fold", type=int, action="append", help="fold to audit; repeat as needed"
    )
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--max-aspect-ratio", type=float, default=3.0)
    parser.add_argument("--geometry-tolerance", type=float, default=1e-6)
    parser.add_argument("--metric-tolerance", type=float, default=1e-9)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.max_cases is not None and args.max_cases < 1:
        raise SystemExit("--max-cases must be positive")
    result = run_audit(args)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    return 0 if result["summary"]["oracle_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
