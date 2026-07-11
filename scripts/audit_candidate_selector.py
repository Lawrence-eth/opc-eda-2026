#!/usr/bin/env python3
"""Measure deployable selector regret against hidden-baseline offline oracles.

The optimizer runs exactly as submitted: its golden baselines remain absent and
its normal proxy chooses each candidate.  This audit intercepts those choices,
scores the same candidate pool afterward with training-label HPWL/area
baselines, and reports false accepts, missed wins, tail regret, and the safe
Pareto-dominance subset.  Golden metrics are diagnostics only and never enter
the optimizer call or inference features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_training_holdout import (  # noqa: E402
    OFFICIAL_ROOT,
    SOLUTION_DIR,
    _baseline,
    _file_sha256,
    _git_state,
    _load_optimizer,
    _optimizer_targets,
    _positions_from_training_label,
    _portable_path,
    _resolve_manifest_cases,
    _solver_component_hashes,
    _source_tree_sha256,
    evaluate_solution,
)
from lite_dataset import FloorplanDatasetLite  # noqa: E402


ARTIFACT_SCHEMA_VERSION = 2
RAW_MODE = "deployed_proxy_vs_offline_golden_baseline_oracle"
COMBINED_MODE = "combined_deployed_proxy_vs_offline_oracle"
RUNTIME_FACTOR_MODE = "neutral_rf_1"
ORACLE_POLICY = "min_pinned_official_cost_on_primary_candidate_snapshots"


def _identity_index(items, selected):
    for index, item in enumerate(items):
        if item is selected:
            return index
    return None


def _snapshot_positions(positions, expected_count):
    """Copy a candidate immediately and reject malformed solver output."""
    if positions is None or len(positions) != expected_count:
        return None
    snapshot = []
    for row in positions:
        if row is None or len(row) != 4:
            return None
        try:
            values = tuple(float(value) for value in row)
        except (TypeError, ValueError, OverflowError):
            return None
        if not all(math.isfinite(value) for value in values):
            return None
        if values[2] <= 0.0 or values[3] <= 0.0:
            return None
        snapshot.append(values)
    return snapshot


def _install_audit_hook(optimizer):
    original = optimizer._select_candidate

    def audited(self, candidates, constraints, area_targets, b2b, p2b, pins_pos, target_positions):
        pristine = self._audit_pristine
        snapshots = [
            _snapshot_positions(candidate, pristine["block_count"])
            for candidate in candidates
        ]
        selected = original(
            candidates,
            constraints,
            area_targets,
            b2b,
            p2b,
            pins_pos,
            target_positions,
        )
        proxy_index = _identity_index(candidates, selected)
        rows = []
        for index, positions in enumerate(snapshots):
            if positions is None:
                rows.append(
                    {
                        "index": index,
                        "valid_output": False,
                        "feasible": False,
                        "hpwl": None,
                        "area": None,
                        "soft_violations": None,
                        "official_cost": 10.0,
                    }
                )
                continue
            metric = evaluate_solution(
                {"positions": positions, "runtime": 1.0},
                pristine["baseline"],
                pristine["constraints"].clone(),
                pristine["b2b"].clone(),
                pristine["p2b"].clone(),
                pristine["pins"].clone(),
                pristine["area"].clone(),
                list(pristine["golden"]),
                median_runtime=1.0,
            )
            rows.append(
                {
                    "index": index,
                    "valid_output": True,
                    "feasible": bool(metric.is_feasible),
                    "hpwl": float(metric.hpwl_total),
                    "area": float(metric.bbox_area),
                    "soft_violations": int(
                        metric.boundary_violations
                        + metric.grouping_violations
                        + metric.mib_violations
                    ),
                    "official_cost": float(metric.cost),
                }
            )

        oracle_index = min(range(len(rows)), key=lambda index: rows[index]["official_cost"])
        incumbent = rows[0]
        dominant = []
        for row in rows[1:]:
            weakly_better = (
                row["valid_output"]
                and incumbent["valid_output"]
                and row["feasible"]
                and row["hpwl"] <= incumbent["hpwl"] + 1e-9
                and row["area"] <= incumbent["area"] + 1e-9
                and row["soft_violations"] <= incumbent["soft_violations"]
            )
            strictly_better = weakly_better and (
                row["hpwl"] < incumbent["hpwl"] - 1e-9
                or row["area"] < incumbent["area"] - 1e-9
                or row["soft_violations"] < incumbent["soft_violations"]
            )
            if weakly_better and strictly_better:
                dominant.append(row["index"])
        proxy_cost = (
            rows[proxy_index]["official_cost"] if proxy_index is not None else 10.0
        )
        oracle_cost = rows[oracle_index]["official_cost"]
        self._audit_decisions.append(
            {
                "decision_ordinal": len(self._audit_decisions),
                "candidate_count": len(rows),
                "proxy_index": proxy_index,
                "oracle_index": oracle_index,
                "pareto_dominant_indices": dominant,
                "incumbent_cost": incumbent["official_cost"],
                "proxy_cost": proxy_cost,
                "oracle_cost": oracle_cost,
                "proxy_regret": proxy_cost - oracle_cost,
                "proxy_false_accept": bool(
                    proxy_index not in (None, 0)
                    and proxy_cost > incumbent["official_cost"] + 1e-12
                ),
                "proxy_missed_win": bool(
                    oracle_cost < proxy_cost - 1e-12
                    and oracle_index != proxy_index
                ),
                "candidates": rows,
            }
        )
        return selected

    optimizer._select_candidate = types.MethodType(audited, optimizer)


def _weighted(rows, key):
    max_blocks = max(row["block_count"] for row in rows)
    weighted = [
        (math.exp((row["block_count"] - max_blocks) / 12.0), row[key])
        for row in rows
    ]
    return sum(weight * value for weight, value in weighted) / sum(
        weight for weight, _value in weighted
    )


def summarize(rows):
    if not rows:
        raise ValueError("selector audit produced no decisions")
    regrets = sorted(row["proxy_regret"] for row in rows)
    tail_count = max(1, math.ceil(0.05 * len(regrets)))
    proxy_accepts = [row for row in rows if row["proxy_index"] not in (None, 0)]
    proxy_true_wins = [
        row
        for row in proxy_accepts
        if row["proxy_cost"] < row["incumbent_cost"] - 1e-12
    ]
    oracle_headroom = _weighted(rows, "incumbent_cost") - _weighted(
        rows, "oracle_cost"
    )
    weighted_regret = _weighted(rows, "proxy_regret")
    return {
        "cases": len(rows),
        "candidate_decisions": sum(row["candidate_count"] for row in rows),
        "proxy_score": _weighted(rows, "proxy_cost"),
        "oracle_score": _weighted(rows, "oracle_cost"),
        "incumbent_score": _weighted(rows, "incumbent_cost"),
        "weighted_proxy_regret": weighted_regret,
        "fraction_oracle_headroom_missed": (
            weighted_regret / oracle_headroom if oracle_headroom > 0.0 else 0.0
        ),
        "oracle_headroom_from_incumbent": _weighted(rows, "oracle_cost")
        - _weighted(rows, "incumbent_cost"),
        "proxy_false_accepts": sum(row["proxy_false_accept"] for row in rows),
        "proxy_accepts": len(proxy_accepts),
        "proxy_true_wins": len(proxy_true_wins),
        "proxy_accept_precision": (
            len(proxy_true_wins) / len(proxy_accepts) if proxy_accepts else 1.0
        ),
        "proxy_missed_wins": sum(row["proxy_missed_win"] for row in rows),
        "proxy_matches_oracle": sum(
            row["proxy_index"] == row["oracle_index"] for row in rows
        ),
        "cases_with_pareto_dominant_candidate": sum(
            bool(row["pareto_dominant_indices"]) for row in rows
        ),
        "worst_proxy_regret": regrets[-1],
        "proxy_regret_cvar_5pct": sum(regrets[-tail_count:]) / tail_count,
    }


def _require_mapping(value, context):
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def _require_field(mapping, field, context):
    if field not in mapping or mapping[field] is None:
        raise ValueError(f"{context} is missing required field {field}")
    return mapping[field]


def _require_sha256(value, context):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _require_git_commit(value, context):
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase full Git commit")
    return value


def _artifact_descriptor(path, payload):
    return {
        "path": _portable_path(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _validate_case(row, context):
    row = _require_mapping(row, context)
    for field in (
        "case_id",
        "source_file",
        "file_offset",
        "sample_index",
        "block_count",
        "input_sha256",
        "optimizer_target_sha256",
        "scoring_label_sha256",
        "candidate_count",
        "proxy_regret",
    ):
        _require_field(row, field, context)
    if not isinstance(row["case_id"], str) or not row["case_id"]:
        raise ValueError(f"{context}.case_id must be a non-empty string")
    if not isinstance(row["source_file"], str) or not row["source_file"]:
        raise ValueError(f"{context}.source_file must be a non-empty string")
    for field in ("file_offset", "sample_index", "block_count", "candidate_count"):
        if isinstance(row[field], bool) or not isinstance(row[field], int):
            raise ValueError(f"{context}.{field} must be an integer")
    if row["file_offset"] < 0 or row["sample_index"] < 0:
        raise ValueError(f"{context} has an out-of-range case identity")
    if row["block_count"] <= 0 or row["candidate_count"] <= 0:
        raise ValueError(f"{context} has an out-of-range count")
    if row["case_id"] != f"{row['source_file']}#{row['file_offset']}":
        raise ValueError(f"{context}.case_id does not match source_file/file_offset")
    for field in ("input_sha256", "optimizer_target_sha256", "scoring_label_sha256"):
        _require_sha256(row[field], f"{context}.{field}")
    try:
        regret = float(row["proxy_regret"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context}.proxy_regret must be numeric") from error
    if not math.isfinite(regret):
        raise ValueError(f"{context}.proxy_regret must be finite")


def _extract_contract(artifact, path):
    if artifact.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"{path} schema_version must be exactly {ARTIFACT_SCHEMA_VERSION}"
        )
    config = _require_mapping(artifact.get("config"), f"{path}.config")
    provenance = _require_mapping(artifact.get("provenance"), f"{path}.provenance")
    if _require_field(config, "mode", f"{path}.config") != RAW_MODE:
        raise ValueError(f"{path}.config.mode is not a raw selector audit")
    runtime_mode = _require_field(
        config, "runtime_factor_mode", f"{path}.config"
    )
    if runtime_mode != RUNTIME_FACTOR_MODE:
        raise ValueError(f"{path}.config.runtime_factor_mode is unsupported")
    oracle_policy = _require_field(config, "oracle_policy", f"{path}.config")
    if oracle_policy != ORACLE_POLICY:
        raise ValueError(f"{path}.config.oracle_policy is unsupported")

    manifest = _require_mapping(
        _require_field(config, "manifest", f"{path}.config"),
        f"{path}.config.manifest",
    )
    manifest_context = f"{path}.config.manifest"
    manifest_sha256 = _require_sha256(
        _require_field(manifest, "sha256", manifest_context),
        f"{manifest_context}.sha256",
    )
    manifest_schema = _require_field(manifest, "schema_version", manifest_context)
    if (
        isinstance(manifest_schema, bool)
        or not isinstance(manifest_schema, int)
        or manifest_schema < 3
    ):
        raise ValueError(f"{manifest_context}.schema_version must be an integer >= 3")
    config_fold = _require_field(config, "fold", f"{path}.config")
    manifest_fold = _require_field(manifest, "fold", manifest_context)
    if (
        isinstance(config_fold, bool)
        or not isinstance(config_fold, int)
        or isinstance(manifest_fold, bool)
        or not isinstance(manifest_fold, int)
        or config_fold != manifest_fold
    ):
        raise ValueError(f"{path} config/manifest fold mismatch")

    generation = _require_mapping(
        _require_field(manifest, "generation", manifest_context),
        f"{manifest_context}.generation",
    )
    for field in ("min_blocks", "max_blocks", "num_folds", "per_size", "seed"):
        _require_field(generation, field, f"{manifest_context}.generation")
    dataset = _require_mapping(
        _require_field(manifest, "dataset", manifest_context),
        f"{manifest_context}.dataset",
    )
    for field in (
        "name",
        "official_floorset_commit",
        "loader",
        "layouts_per_file",
        "source_file_count",
        "source_inventory_sha256",
    ):
        _require_field(dataset, field, f"{manifest_context}.dataset")
    fold_metadata = _require_mapping(
        _require_field(manifest, "fold_metadata", manifest_context),
        f"{manifest_context}.fold_metadata",
    )
    metadata_fold = _require_field(
        fold_metadata, "fold", f"{manifest_context}.fold_metadata"
    )
    if (
        isinstance(metadata_fold, bool)
        or not isinstance(metadata_fold, int)
        or metadata_fold != config_fold
    ):
        raise ValueError(f"{path} config/fold-metadata mismatch")
    inventory = _require_sha256(
        _require_field(dataset, "source_inventory_sha256", f"{manifest_context}.dataset"),
        f"{manifest_context}.dataset.source_inventory_sha256",
    )
    resolved_inventory = _require_sha256(
        _require_field(manifest, "resolved_inventory_sha256", manifest_context),
        f"{manifest_context}.resolved_inventory_sha256",
    )
    if inventory != resolved_inventory:
        raise ValueError(f"{path} resolved dataset inventory does not match manifest")
    dataset_commit = _require_git_commit(
        _require_field(
            dataset, "official_floorset_commit", f"{manifest_context}.dataset"
        ),
        f"{manifest_context}.dataset.official_floorset_commit",
    )
    resolved_commit = _require_git_commit(
        _require_field(manifest, "resolved_official_floorset_commit", manifest_context),
        f"{manifest_context}.resolved_official_floorset_commit",
    )
    if dataset_commit != resolved_commit:
        raise ValueError(f"{path} resolved FloorSet commit does not match manifest")

    official_git = _require_mapping(
        _require_field(provenance, "official_floorset_git", f"{path}.provenance"),
        f"{path}.provenance.official_floorset_git",
    )
    official_commit = _require_git_commit(
        _require_field(official_git, "commit", f"{path}.provenance.official_floorset_git"),
        f"{path}.provenance.official_floorset_git.commit",
    )
    if official_commit != dataset_commit:
        raise ValueError(f"{path} official FloorSet commit does not match manifest")
    tracked_dirty = _require_field(
        official_git, "tracked_dirty", f"{path}.provenance.official_floorset_git"
    )
    if tracked_dirty is not False:
        raise ValueError(f"{path} official FloorSet checkout has tracked changes")

    evaluator_sha256 = _require_sha256(
        _require_field(provenance, "evaluator_sha256", f"{path}.provenance"),
        f"{path}.provenance.evaluator_sha256",
    )
    harness_sha256 = _require_sha256(
        _require_field(
            provenance, "evaluation_harness_sha256", f"{path}.provenance"
        ),
        f"{path}.provenance.evaluation_harness_sha256",
    )
    solver_source_sha256 = _require_sha256(
        _require_field(provenance, "solver_source_sha256", f"{path}.provenance"),
        f"{path}.provenance.solver_source_sha256",
    )
    solver_components = _require_mapping(
        _require_field(provenance, "solver_component_sha256", f"{path}.provenance"),
        f"{path}.provenance.solver_component_sha256",
    )
    if not solver_components:
        raise ValueError(f"{path}.provenance.solver_component_sha256 must not be empty")
    for name, digest in solver_components.items():
        _require_sha256(digest, f"{path}.provenance.solver_component_sha256.{name}")

    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "mode": RAW_MODE,
        "manifest_sha256": manifest_sha256,
        "manifest_schema_version": manifest_schema,
        "manifest_generation": generation,
        "dataset": dataset,
        "resolved_inventory_sha256": resolved_inventory,
        "official_floorset_commit": official_commit,
        "official_floorset_tracked_dirty": tracked_dirty,
        "evaluator_sha256": evaluator_sha256,
        "evaluation_harness_sha256": harness_sha256,
        "runtime_factor_mode": runtime_mode,
        "oracle_policy": oracle_policy,
        "solver_source_sha256": solver_source_sha256,
        "solver_component_sha256": solver_components,
    }


def combine_artifacts(paths):
    rows = []
    seen = set()
    descriptors = []
    contract = None
    for raw_path in paths:
        path = Path(raw_path)
        payload = path.read_bytes()
        artifact = json.loads(payload)
        current_contract = _extract_contract(artifact, path)
        if contract is None:
            contract = current_contract
        elif current_contract != contract:
            raise ValueError(f"{path} evaluation contract mismatch")
        artifact_rows = artifact.get("cases")
        if not isinstance(artifact_rows, list) or not artifact_rows:
            raise ValueError(f"{path}.cases must be a non-empty list")
        for position, row in enumerate(artifact_rows):
            _validate_case(row, f"{path}.cases[{position}]")
            identity = row["case_id"]
            if identity in seen:
                raise ValueError(f"combined audits repeat case {identity}")
            seen.add(identity)
            rows.append(row)
        descriptors.append(_artifact_descriptor(path, payload))
    if contract is None:
        raise ValueError("at least one selector-audit artifact is required")
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "config": {
            "mode": COMBINED_MODE,
            "inputs": descriptors,
        },
        "evaluation_contract": contract,
        "summary": summarize(rows),
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=OFFICIAL_ROOT)
    parser.add_argument("--solver-dir", type=Path, default=SOLUTION_DIR)
    parser.add_argument(
        "--fold-manifest",
        type=Path,
        default=ROOT / "results" / "folds" / "heavy_clean_v1.json",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fold", type=int)
    mode.add_argument(
        "--combine",
        type=Path,
        nargs="+",
        help="combine existing raw selector-audit artifacts without solving",
    )
    parser.add_argument("--max-cases", type=int, help="bounded smoke/debug prefix")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.combine:
        result = combine_artifacts(args.combine)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["summary"], indent=2))
        print(f"wrote {args.output}")
        return

    dataset = FloorplanDatasetLite(str(args.data_root))
    selected, manifest = _resolve_manifest_cases(
        dataset, args.data_root, args.fold_manifest, args.fold
    )
    if args.max_cases is not None:
        if args.max_cases < 1:
            parser.error("--max-cases must be positive")
        selected = selected[: args.max_cases]
    optimizer = _load_optimizer(args.solver_dir)
    optimizer._baselines_by_n = {}
    _install_audit_hook(optimizer)
    rows = []
    for ordinal, (sample_index, sample, identity) in enumerate(selected, 1):
        area, b2b, p2b, pins, constraints = sample["input"]
        _tree, fp_sol, stored_metrics = sample["label"]
        n = int((area != -1).sum().item())
        golden = _positions_from_training_label(fp_sol, n)
        targets = _optimizer_targets(constraints, golden, n)
        optimizer._audit_baseline = _baseline(stored_metrics)
        optimizer._audit_pristine = {
            "block_count": n,
            "baseline": optimizer._audit_baseline,
            "area": area.clone(),
            "b2b": b2b.clone(),
            "p2b": p2b.clone(),
            "pins": pins.clone(),
            "constraints": constraints.clone(),
            "targets": targets.clone(),
            "golden": tuple(golden),
        }
        optimizer._audit_decisions = []
        optimizer._baselines_by_n = {}
        optimizer._hpwl_baseline = None
        optimizer._area_baseline = None
        started = time.perf_counter()
        optimizer.solve(
            n,
            area.clone(),
            b2b.clone(),
            p2b.clone(),
            pins.clone(),
            constraints.clone(),
            targets.clone(),
        )
        runtime = time.perf_counter() - started
        if not optimizer._audit_decisions:
            raise RuntimeError(f"no selector decision captured for {identity['case_id']}")
        # The first selector call is the primary shelf/dissection portfolio.
        # Later repairs depend on this proxy-selected path and therefore are not
        # a clean counterfactual pool (some can also contain more trial moves).
        primary = optimizer._audit_decisions[0]
        primary.update(
            {
                "case_id": identity["case_id"],
                "sample_index": sample_index,
                "source_file": identity["source_file"],
                "file_offset": identity["file_offset"],
                "input_sha256": identity["input_sha256"],
                "optimizer_target_sha256": identity["optimizer_target_sha256"],
                "scoring_label_sha256": identity["scoring_label_sha256"],
                "block_count": n,
                "solver_runtime_seconds": runtime,
                "all_decision_count": len(optimizer._audit_decisions),
            }
        )
        rows.append(primary)
        if ordinal % 10 == 0 or ordinal == len(selected):
            print(f"audited {ordinal}/{len(selected)}", flush=True)

    result = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "config": {
            "solver_dir": _portable_path(args.solver_dir),
            "fold_manifest": _portable_path(args.fold_manifest),
            "fold": args.fold,
            "max_cases": args.max_cases,
            "manifest": manifest,
            "mode": RAW_MODE,
            "runtime_factor_mode": RUNTIME_FACTOR_MODE,
            "oracle_policy": ORACLE_POLICY,
        },
        "provenance": {
            "evaluation_harness_sha256": _file_sha256(Path(__file__)),
            "solver_source_sha256": _source_tree_sha256(args.solver_dir),
            "solver_component_sha256": _solver_component_hashes(args.solver_dir),
            "solver_git": _git_state(args.solver_dir),
            "evaluator_sha256": _file_sha256(
                OFFICIAL_ROOT / "iccad2026contest" / "iccad2026_evaluate.py"
            ),
            "official_floorset_git": _git_state(OFFICIAL_ROOT),
        },
        "summary": summarize(rows),
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
