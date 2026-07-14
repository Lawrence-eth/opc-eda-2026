#!/usr/bin/env python3
"""Select the learned-policy mode from provenance-bound holdout evidence.

The selector consumes the six development comparison artifacts produced for
``clean/raw x replacement/additive/additive_first_pass``.  It does not trust
their reported statistics: every referenced holdout and manifest is rehashed,
the complete holdout contract is validated, and ``compare_fold_results`` is
replayed with the recorded seed and sample count.  An optional pair of fold-3
comparisons confirms exactly the one development finalist selected here.

This is an evidence compositor, not an evaluator.  It never imports the live
solver and never runs FloorSet cases.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import compare_fold_results as fold_compare  # noqa: E402
from solver_components import (  # noqa: E402
    LIVE_SOLVER_COMPONENTS,
    validate_live_solver_components,
)


SCHEMA_VERSION = 1
MODES = ("off", "replacement", "additive", "additive_first_pass")
CHALLENGERS = MODES[1:]
PANELS = ("clean", "raw")
DEV_FOLDS = (0, 1, 2)
CALIBRATION_FOLD = 3
EXPECTED_CASES_PER_FOLD = 105
EXPECTED_BLOCK_COUNTS = tuple(range(100, 121))
EXPECTED_PER_SIZE = 5
BOOTSTRAP_SAMPLES = 30_000
BOOTSTRAP_SEED = 20260710
PSEUDO_SEED = BOOTSTRAP_SEED + 1
COMPLEXITY_ORDER = {
    "replacement": 0,
    "additive_first_pass": 1,
    "additive": 2,
}

PANEL_MANIFEST_PATHS = {
    "clean": "results/folds/heavy_clean_v1.json",
    "raw": "results/folds/heavy_raw_hash_v1.json",
}

THRESHOLDS = {
    "development": {
        "clean": {
            "maximum_pooled_delta": -0.00025,
            "minimum_bootstrap_probability_improves": 1.0 - 0.05 / 3.0,
            "maximum_bootstrap_ci95_upper": 0.0,
            "maximum_pseudo_ci95_upper": 0.0,
            "maximum_fold_delta": 0.005,
            "maximum_worst_case_score_contribution": 0.00025,
            "maximum_regression_cvar_5pct": 0.00010,
            "maximum_worst_sampled_pseudo_delta": 0.002,
        },
        "raw": {
            "maximum_pooled_delta": 0.00025,
            "maximum_bootstrap_ci95_upper": 0.001,
            "maximum_pseudo_ci95_upper": 0.001,
            "maximum_fold_delta": 0.005,
            "maximum_worst_case_score_contribution": 0.00025,
            "maximum_regression_cvar_5pct": 0.00010,
            "maximum_worst_sampled_pseudo_delta": 0.002,
        },
        "direct_ranking": {
            "minimum_clean_improvement": 0.00025,
            "minimum_clean_bootstrap_probability_improves": 0.95,
            "maximum_raw_delta": 0.00025,
            "maximum_raw_bootstrap_ci95_upper": 0.001,
        },
    },
    "calibration": {
        "clean": {
            "maximum_pooled_delta_exclusive": 0.0,
            "minimum_bootstrap_probability_improves": 0.90,
            "maximum_pseudo_ci95_upper": 0.0,
            "maximum_worst_case_score_contribution": 0.00025,
            "maximum_regression_cvar_5pct": 0.00010,
            "maximum_worst_sampled_pseudo_delta": 0.002,
        },
        "raw": {
            "maximum_pooled_delta": 0.00025,
            "maximum_bootstrap_ci95_upper": 0.001,
            "maximum_pseudo_ci95_upper": 0.001,
            "maximum_worst_case_score_contribution": 0.00025,
            "maximum_regression_cvar_5pct": 0.00010,
            "maximum_worst_sampled_pseudo_delta": 0.002,
        },
    },
}

HOLDOUT_TOP_LEVEL_KEYS = {
    "config",
    "provenance",
    "solver_all",
    "golden_all",
    "solver_golden_mib_clean",
    "golden_mib_clean",
    "golden_mib_violation_cases",
    "solver_by_size",
    "cases",
}
HOLDOUT_CONFIG_KEYS = {
    "min_blocks",
    "max_blocks",
    "per_size",
    "seed",
    "scanned_files",
    "require_golden_mib_clean",
    "solver_dir",
    "learned_mode",
    "indices_from",
    "fold_manifest",
    "fold",
    "manifest",
    "oracle_baseline_selector",
    "runtime_factor_mode",
}
HOLDOUT_PROVENANCE_KEYS = {
    "evaluation_harness_sha256",
    "solver_source_sha256",
    "solver_component_sha256",
    "solver_git",
    "evaluator_sha256",
    "official_floorset_git",
}
CASE_KEYS = {
    "sample_index",
    "block_count",
    "is_feasible",
    "cost",
    "hpwl_gap",
    "hpwl_gap_clamped",
    "area_gap",
    "area_gap_clamped",
    "violations_relative",
    "boundary_violations",
    "grouping_violations",
    "mib_violations",
    "golden_mib_violations",
    "runtime_seconds",
    "case_id",
    "source_file",
    "file_offset",
    "input_sha256",
    "optimizer_target_sha256",
    "scoring_label_sha256",
}
SUMMARY_KEYS = {
    "cases",
    "feasible",
    "total_score",
    "weighted_hpwl_gap_clamped",
    "weighted_area_gap_clamped",
    "weighted_violations_relative",
    "runtime_mean",
    "runtime_p95",
    "runtime_max",
}
IDENTITY_FIELDS = (
    "case_id",
    "source_file",
    "file_offset",
    "sample_index",
    "block_count",
    "input_sha256",
    "optimizer_target_sha256",
    "scoring_label_sha256",
)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _decode_json(raw: bytes, context: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"cannot decode {context}: {exc}") from exc


def _load_object(path: Path, context: str) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read {context} {path}: {exc}") from exc
    value = _decode_json(raw, f"{context} {path}")
    if not isinstance(value, dict):
        raise ValueError(f"{context} {path} must contain a JSON object")
    return raw, value


def _require_exact_keys(value: Any, expected: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        observed = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(
            f"{context} must have exactly {sorted(expected)}; observed {observed}"
        )
    return value


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def _require_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    return value


def _require_int(value: Any, context: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{context} must be >= {minimum}")
    return value


def _require_bool(value: Any, context: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{context} must be boolean")
    return value


def _require_number(
    value: Any,
    context: str,
    *,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{context} must be >= {minimum}")
    return result


def _require_sha256(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _require_commit(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a full lowercase Git commit")
    return value


def _require_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a nonempty string")
    return value


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _artifact(path: Path, *, displayed_path: str | None = None) -> dict[str, Any]:
    return {
        "path": displayed_path if displayed_path is not None else _portable_path(path),
        "resolved_path": _portable_path(path),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _git_bytes(commit: str, relative_path: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{commit}:{relative_path}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(
            f"cannot read {relative_path} from commit {commit}: "
            f"{completed.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return completed.stdout


def _git_names(commit: str, relative_tree: str) -> list[str]:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), "ls-tree", "--name-only", f"{commit}:{relative_tree}"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(
            f"cannot list {relative_tree} at {commit}: {completed.stderr.strip()}"
        )
    return completed.stdout.splitlines()


def _source_tree_sha256_at_commit(commit: str) -> str:
    digest = hashlib.sha256()
    names = sorted(name for name in _git_names(commit, "contest_solution") if name.endswith(".py"))
    if not names:
        raise ValueError("frozen solver tree contains no Python sources")
    for name in names:
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(bytes.fromhex(_sha256_bytes(_git_bytes(commit, f"contest_solution/{name}"))))
    return digest.hexdigest()


def _campaign_contract(expected_commit: str) -> dict[str, Any]:
    expected_commit = _require_commit(expected_commit, "expected solver commit")
    # Resolve to a commit object, rejecting an arbitrary 40-hex object ID.
    completed = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--verify", f"{expected_commit}^{{commit}}"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0 or completed.stdout.strip() != expected_commit:
        raise ValueError(f"expected solver commit is unavailable: {expected_commit}")

    for relative in (
        "scripts/compare_fold_results.py",
        "scripts/evaluate_training_holdout.py",
        "scripts/solver_components.py",
    ):
        local = ROOT / relative
        if not local.is_file() or local.read_bytes() != _git_bytes(expected_commit, relative):
            raise ValueError(f"live campaign harness differs from {expected_commit}: {relative}")

    official_raw = _git_bytes(expected_commit, "docs/official_sources.json")
    official = _decode_json(official_raw, "frozen official-source manifest")
    official = _require_mapping(official, "frozen official-source manifest")
    floorset = _require_mapping(official.get("floorset"), "official_sources.floorset")
    official_commit = _require_commit(floorset.get("commit"), "official_sources.floorset.commit")
    files = _require_mapping(floorset.get("files"), "official_sources.floorset.files")
    evaluator_sha256 = _require_sha256(
        files.get("iccad2026contest/iccad2026_evaluate.py"),
        "official evaluator digest",
    )

    components = validate_live_solver_components(LIVE_SOLVER_COMPONENTS)
    component_sha256 = {
        name: _sha256_bytes(_git_bytes(expected_commit, f"contest_solution/{name}"))
        for name in components
    }
    manifests: dict[str, dict[str, Any]] = {}
    for panel, relative in PANEL_MANIFEST_PATHS.items():
        raw = _git_bytes(expected_commit, relative)
        payload = _decode_json(raw, f"frozen {panel} manifest")
        if not isinstance(payload, dict):
            raise ValueError(f"frozen {panel} manifest must be an object")
        manifest_dataset = _require_mapping(
            payload.get("dataset"), f"frozen {panel} manifest.dataset"
        )
        if _require_commit(
            manifest_dataset.get("official_floorset_commit"),
            f"frozen {panel} manifest.dataset.official_floorset_commit",
        ) != official_commit:
            raise ValueError(
                f"frozen {panel} manifest does not bind the official FloorSet commit"
            )
        manifests[panel] = {
            "relative_path": relative,
            "raw": raw,
            "sha256": _sha256_bytes(raw),
            "payload": payload,
        }

    return {
        "expected_commit": expected_commit,
        "official_sources": {
            "path": "docs/official_sources.json",
            "sha256": _sha256_bytes(official_raw),
            "size_bytes": len(official_raw),
        },
        "official_floorset_commit": official_commit,
        "evaluator_sha256": evaluator_sha256,
        "evaluation_harness_sha256": _sha256_bytes(
            _git_bytes(expected_commit, "scripts/evaluate_training_holdout.py")
        ),
        "comparison_harness_sha256": _sha256_bytes(
            _git_bytes(expected_commit, "scripts/compare_fold_results.py")
        ),
        "solver_source_sha256": _source_tree_sha256_at_commit(expected_commit),
        "solver_component_sha256": component_sha256,
        "manifests": manifests,
    }


def _resolve_recorded_path(
    raw_path: str,
    *,
    artifact_root: Path,
    owner_path: Path,
    context: str,
) -> Path:
    recorded = Path(_require_string(raw_path, f"{context}.path"))
    if recorded.is_absolute():
        candidates = [recorded.resolve()]
    else:
        candidates = [
            (artifact_root / recorded).resolve(),
            (owner_path.parent / recorded).resolve(),
            (ROOT / recorded).resolve(),
        ]
    existing: list[Path] = []
    for candidate in candidates:
        if candidate.is_file() and candidate not in existing:
            existing.append(candidate)
    if not existing:
        raise ValueError(f"{context} cannot be resolved: {recorded}")
    if len(existing) != 1:
        raise ValueError(f"{context} is ambiguous: {recorded} -> {existing}")
    return existing[0]


def _validate_descriptor(
    value: Any,
    *,
    artifact_root: Path,
    owner_path: Path,
    context: str,
) -> tuple[dict[str, Any], Path]:
    descriptor = _require_exact_keys(
        value, {"path", "sha256", "size_bytes"}, context
    )
    digest = _require_sha256(descriptor["sha256"], f"{context}.sha256")
    size = _require_int(descriptor["size_bytes"], f"{context}.size_bytes", minimum=1)
    path = _resolve_recorded_path(
        descriptor["path"], artifact_root=artifact_root, owner_path=owner_path, context=context
    )
    if path.stat().st_size != size or _sha256_file(path) != digest:
        raise ValueError(f"{context} does not match the referenced artifact bytes")
    return descriptor, path


def _manifest_fold(payload: dict[str, Any], fold: int, context: str) -> dict[str, Any]:
    if _require_int(payload.get("schema_version"), f"{context}.schema_version") != 3:
        raise ValueError(f"{context}.schema_version must be 3")
    if payload.get("split_unit") != "source_file":
        raise ValueError(f"{context}.split_unit must be source_file")
    manifests = _require_list(payload.get("manifests"), f"{context}.manifests")
    matches = []
    for index, row in enumerate(manifests):
        row = _require_mapping(row, f"{context}.manifests[{index}]")
        row_fold = _require_int(row.get("fold"), f"{context}.manifests[{index}].fold")
        if row_fold == fold:
            matches.append(row)
    if len(matches) != 1:
        raise ValueError(f"{context} has {len(matches)} entries for fold {fold}")
    return matches[0]


def _expected_manifest_provenance(
    manifest_raw: bytes,
    manifest: dict[str, Any],
    fold_row: dict[str, Any],
    fold: int,
) -> dict[str, Any]:
    dataset = _require_mapping(manifest.get("dataset"), "manifest.dataset")
    generation = _require_mapping(manifest.get("generation"), "manifest.generation")
    inventory = _require_sha256(
        dataset.get("source_inventory_sha256"), "manifest.dataset.source_inventory_sha256"
    )
    official_commit = _require_commit(
        dataset.get("official_floorset_commit"), "manifest.dataset.official_floorset_commit"
    )
    return {
        "sha256": _sha256_bytes(manifest_raw),
        "schema_version": 3,
        "fold": fold,
        "fold_metadata": {key: value for key, value in fold_row.items() if key != "cases"},
        "generation": generation,
        "dataset": dataset,
        "resolved_inventory_sha256": inventory,
        "resolved_official_floorset_commit": official_commit,
    }


def _official_cost(row: dict[str, Any]) -> float:
    if not row["is_feasible"]:
        return 10.0
    quality = 1.0 + 0.5 * (
        max(0.0, float(row["hpwl_gap"])) + max(0.0, float(row["area_gap"]))
    )
    try:
        violation = math.exp(2.0 * float(row["violations_relative"]))
    except OverflowError as exc:
        raise ValueError("violations_relative overflows official scoring") from exc
    return min(quality * violation, 10.0 - 1e-6)


def _weighted_score(rows: list[dict[str, Any]]) -> float:
    max_n = max(int(row["block_count"]) for row in rows)
    weighted = [
        (math.exp((int(row["block_count"]) - max_n) / 12.0), float(row["cost"]))
        for row in rows
    ]
    return sum(weight * cost for weight, cost in weighted) / sum(
        weight for weight, _cost in weighted
    )


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"cases": 0}
    max_n = max(int(row["block_count"]) for row in rows)
    weights = [math.exp((int(row["block_count"]) - max_n) / 12.0) for row in rows]
    denominator = sum(weights)

    def weighted(field: str) -> float:
        return sum(weight * float(row[field]) for weight, row in zip(weights, rows)) / denominator

    runtimes = [float(row["runtime_seconds"]) for row in rows]
    return {
        "cases": len(rows),
        "feasible": sum(bool(row["is_feasible"]) for row in rows),
        "total_score": _weighted_score(rows),
        "weighted_hpwl_gap_clamped": weighted("hpwl_gap_clamped"),
        "weighted_area_gap_clamped": weighted("area_gap_clamped"),
        "weighted_violations_relative": weighted("violations_relative"),
        "runtime_mean": statistics.fmean(runtimes),
        "runtime_p95": sorted(runtimes)[max(0, math.ceil(0.95 * len(runtimes)) - 1)],
        "runtime_max": max(runtimes),
    }


def _same_number(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    try:
        a = float(left)
        b = float(right)
    except (TypeError, ValueError):
        return False
    return math.isfinite(a) and math.isfinite(b) and math.isclose(
        a, b, rel_tol=0.0, abs_tol=tolerance
    )


def _validate_summary(value: Any, expected: dict[str, Any], context: str) -> None:
    if expected == {"cases": 0}:
        if value != expected:
            raise ValueError(f"{context} must be the empty summary")
        return
    summary = _require_exact_keys(value, SUMMARY_KEYS, context)
    for field in ("cases", "feasible"):
        actual = _require_int(summary[field], f"{context}.{field}", minimum=0)
        if actual != expected[field]:
            raise ValueError(f"{context}.{field} mismatch: {actual} != {expected[field]}")
    for field in SUMMARY_KEYS - {"cases", "feasible"}:
        _require_number(summary[field], f"{context}.{field}")
        if not _same_number(summary[field], expected[field]):
            raise ValueError(f"{context}.{field} does not match reconstructed value")


def _validate_unreconstructable_summary(
    value: Any, *, expected_cases: int, context: str
) -> None:
    if expected_cases == 0:
        if value != {"cases": 0}:
            raise ValueError(f"{context} must be the empty summary")
        return
    summary = _require_exact_keys(value, SUMMARY_KEYS, context)
    if _require_int(summary["cases"], f"{context}.cases") != expected_cases:
        raise ValueError(f"{context}.cases does not match the panel")
    feasible = _require_int(summary["feasible"], f"{context}.feasible", minimum=0)
    if feasible > expected_cases:
        raise ValueError(f"{context}.feasible exceeds its case count")
    for field in SUMMARY_KEYS - {"cases", "feasible"}:
        _require_number(summary[field], f"{context}.{field}", minimum=0.0)


def _validate_case(
    row: Any,
    manifest_case: dict[str, Any],
    context: str,
) -> dict[str, Any]:
    row = _require_exact_keys(row, CASE_KEYS, context)
    for field in IDENTITY_FIELDS:
        if field not in manifest_case or row[field] != manifest_case[field]:
            raise ValueError(f"{context}.{field} does not match the fold manifest")
    if row["case_id"] != f"{row['source_file']}#{row['file_offset']}":
        raise ValueError(f"{context}.case_id is not canonical")
    for field in ("sample_index", "block_count", "file_offset"):
        _require_int(row[field], f"{context}.{field}", minimum=0)
    for field in ("input_sha256", "optimizer_target_sha256", "scoring_label_sha256"):
        _require_sha256(row[field], f"{context}.{field}")
    _require_bool(row["is_feasible"], f"{context}.is_feasible")
    for field in (
        "cost",
        "hpwl_gap",
        "hpwl_gap_clamped",
        "area_gap",
        "area_gap_clamped",
        "violations_relative",
        "runtime_seconds",
    ):
        _require_number(
            row[field],
            f"{context}.{field}",
            minimum=0.0 if field in {"cost", "hpwl_gap_clamped", "area_gap_clamped", "violations_relative", "runtime_seconds"} else None,
        )
    for field in (
        "boundary_violations",
        "grouping_violations",
        "mib_violations",
        "golden_mib_violations",
    ):
        _require_int(row[field], f"{context}.{field}", minimum=0)
    if not _same_number(row["hpwl_gap_clamped"], max(0.0, float(row["hpwl_gap"]))):
        raise ValueError(f"{context}.hpwl_gap_clamped is inconsistent")
    if not _same_number(row["area_gap_clamped"], max(0.0, float(row["area_gap"]))):
        raise ValueError(f"{context}.area_gap_clamped is inconsistent")
    if not _same_number(row["cost"], _official_cost(row), tolerance=1e-10):
        raise ValueError(f"{context}.cost does not match official neutral-RF scoring")
    return row


def _validate_holdout(
    path: Path,
    *,
    panel: str,
    mode: str,
    fold: int,
    manifest_raw: bytes,
    manifest_payload: dict[str, Any],
    campaign: dict[str, Any],
) -> dict[str, Any]:
    raw, payload = _load_object(path, "holdout artifact")
    _require_exact_keys(payload, HOLDOUT_TOP_LEVEL_KEYS, f"{path}")
    config = _require_exact_keys(payload["config"], HOLDOUT_CONFIG_KEYS, f"{path}.config")
    provenance = _require_exact_keys(
        payload["provenance"], HOLDOUT_PROVENANCE_KEYS, f"{path}.provenance"
    )
    if config["learned_mode"] != mode:
        raise ValueError(
            f"{path}.config.learned_mode is {config['learned_mode']!r}; expected {mode!r}"
        )
    if config["runtime_factor_mode"] != "neutral_rf_1":
        raise ValueError(f"{path} does not use neutral RF=1 scoring")
    if _require_bool(config["oracle_baseline_selector"], f"{path}.config.oracle_baseline_selector"):
        raise ValueError(f"{path} enables the oracle baseline selector")
    if config["indices_from"] is not None or config["require_golden_mib_clean"] is not None:
        raise ValueError(f"{path} does not use the canonical manifest-only evaluation mode")
    if _require_int(config["scanned_files"], f"{path}.config.scanned_files") != 0:
        raise ValueError(f"{path}.config.scanned_files must be zero for a manifest run")
    _require_string(config["solver_dir"], f"{path}.config.solver_dir")
    _require_string(config["fold_manifest"], f"{path}.config.fold_manifest")
    if _require_int(config["fold"], f"{path}.config.fold") != fold:
        raise ValueError(f"{path} has the wrong fold")

    fold_row = _manifest_fold(manifest_payload, fold, f"{panel} manifest")
    expected_manifest = _expected_manifest_provenance(
        manifest_raw, manifest_payload, fold_row, fold
    )
    if config["manifest"] != expected_manifest:
        raise ValueError(f"{path}.config.manifest does not match canonical {panel} bytes")
    generation = expected_manifest["generation"]
    for field in ("min_blocks", "max_blocks", "per_size", "seed"):
        expected_value = _require_int(generation[field], f"{panel} manifest.generation.{field}")
        if _require_int(config[field], f"{path}.config.{field}") != expected_value:
            raise ValueError(f"{path}.config.{field} does not match the manifest")
    if (
        generation["min_blocks"] != 100
        or generation["max_blocks"] != 120
        or generation["per_size"] != EXPECTED_PER_SIZE
        or generation["num_folds"] != 5
    ):
        raise ValueError(f"{panel} manifest does not have the frozen heavy-panel quotas")

    if _require_sha256(
        provenance["evaluation_harness_sha256"],
        f"{path}.provenance.evaluation_harness_sha256",
    ) != campaign["evaluation_harness_sha256"]:
        raise ValueError(f"{path} uses the wrong holdout harness")
    if _require_sha256(
        provenance["evaluator_sha256"], f"{path}.provenance.evaluator_sha256"
    ) != campaign["evaluator_sha256"]:
        raise ValueError(f"{path} uses the wrong official evaluator")
    if _require_sha256(
        provenance["solver_source_sha256"], f"{path}.provenance.solver_source_sha256"
    ) != campaign["solver_source_sha256"]:
        raise ValueError(f"{path} solver source tree does not match the frozen commit")
    components = _require_mapping(
        provenance["solver_component_sha256"], f"{path}.provenance.solver_component_sha256"
    )
    if components != campaign["solver_component_sha256"]:
        raise ValueError(f"{path} solver component hashes do not match the frozen commit")

    solver_git = _require_exact_keys(
        provenance["solver_git"],
        {"commit", "dirty", "tracked_dirty", "has_untracked"},
        f"{path}.provenance.solver_git",
    )
    if _require_commit(solver_git["commit"], f"{path}.provenance.solver_git.commit") != campaign["expected_commit"]:
        raise ValueError(f"{path} solver commit does not match the campaign")
    for field in ("dirty", "tracked_dirty", "has_untracked"):
        if _require_bool(solver_git[field], f"{path}.provenance.solver_git.{field}"):
            raise ValueError(f"{path} solver Git state is not clean")

    official_git = _require_mapping(
        provenance["official_floorset_git"], f"{path}.provenance.official_floorset_git"
    )
    if _require_commit(
        official_git.get("commit"), f"{path}.provenance.official_floorset_git.commit"
    ) != campaign["official_floorset_commit"]:
        raise ValueError(f"{path} uses the wrong FloorSet commit")
    if _require_bool(
        official_git.get("tracked_dirty"),
        f"{path}.provenance.official_floorset_git.tracked_dirty",
    ):
        raise ValueError(f"{path} official FloorSet checkout has tracked changes")

    rows = _require_list(payload["cases"], f"{path}.cases")
    manifest_cases = _require_list(fold_row.get("cases"), f"{panel} manifest fold {fold}.cases")
    if len(rows) != EXPECTED_CASES_PER_FOLD or len(manifest_cases) != EXPECTED_CASES_PER_FOLD:
        raise ValueError(f"{path} must contain exactly {EXPECTED_CASES_PER_FOLD} cases")
    validated = [
        _validate_case(row, manifest_case, f"{path}.cases[{index}]")
        for index, (row, manifest_case) in enumerate(zip(rows, manifest_cases))
    ]
    identities = [row["case_id"] for row in validated]
    if len(identities) != len(set(identities)):
        raise ValueError(f"{path} repeats a case identity")
    counts = {size: 0 for size in EXPECTED_BLOCK_COUNTS}
    for row in validated:
        block_count = row["block_count"]
        if block_count not in counts:
            raise ValueError(f"{path} contains non-heavy block count {block_count}")
        counts[block_count] += 1
    if any(count != EXPECTED_PER_SIZE for count in counts.values()):
        raise ValueError(f"{path} does not have five cases for every heavy size")
    if not all(row["is_feasible"] for row in validated):
        raise ValueError(f"{path} is not fully feasible")

    _validate_summary(payload["solver_all"], _summary(validated), f"{path}.solver_all")
    clean_rows = [row for row in validated if row["golden_mib_violations"] == 0]
    _validate_summary(
        payload["solver_golden_mib_clean"],
        _summary(clean_rows),
        f"{path}.solver_golden_mib_clean",
    )
    expected_violation_cases = len(validated) - len(clean_rows)
    if _require_int(
        payload["golden_mib_violation_cases"], f"{path}.golden_mib_violation_cases"
    ) != expected_violation_cases:
        raise ValueError(f"{path}.golden_mib_violation_cases is inconsistent")

    solver_by_size = _require_mapping(payload["solver_by_size"], f"{path}.solver_by_size")
    if set(solver_by_size) != {str(size) for size in EXPECTED_BLOCK_COUNTS}:
        raise ValueError(f"{path}.solver_by_size has the wrong block counts")
    for size in EXPECTED_BLOCK_COUNTS:
        _validate_summary(
            solver_by_size[str(size)],
            _summary([row for row in validated if row["block_count"] == size]),
            f"{path}.solver_by_size[{size}]",
        )

    # Golden rows are not stored. Bind their summaries across modes later, and
    # validate the case counts here so an empty or unrelated summary cannot pass.
    _validate_unreconstructable_summary(
        payload["golden_all"], expected_cases=len(validated), context=f"{path}.golden_all"
    )
    _validate_unreconstructable_summary(
        payload["golden_mib_clean"],
        expected_cases=len(clean_rows),
        context=f"{path}.golden_mib_clean",
    )

    return {
        "path": path,
        "raw": raw,
        "payload": payload,
        "rows": validated,
        "artifact": _artifact(path),
        "identity_contract": [tuple(row[field] for field in IDENTITY_FIELDS) for row in validated],
        "golden_contract": {
            "golden_all": copy.deepcopy(payload["golden_all"]),
            "golden_mib_clean": copy.deepcopy(payload["golden_mib_clean"]),
            "golden_mib_violation_cases": payload["golden_mib_violation_cases"],
        },
    }


def _first_difference(left: Any, right: Any, path: str = "$" ) -> str | None:
    if type(left) is not type(right):
        return f"{path}: type {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return f"{path}: keys {sorted(left)} != {sorted(right)}"
        for key in left:
            difference = _first_difference(left[key], right[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: length {len(left)} != {len(right)}"
        for index, (a, b) in enumerate(zip(left, right)):
            difference = _first_difference(a, b, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if isinstance(left, float):
        if math.isnan(left) or math.isnan(right) or left != right:
            return f"{path}: {left!r} != {right!r}"
        return None
    if left != right:
        return f"{path}: {left!r} != {right!r}"
    return None


def _validate_comparison_shape(
    payload: dict[str, Any],
    *,
    folds: tuple[int, ...],
    context: str,
) -> None:
    if payload.get("schema_version") != 2:
        raise ValueError(f"{context}.schema_version must be 2")
    cases = EXPECTED_CASES_PER_FOLD * len(folds)
    if _require_int(payload.get("cases"), f"{context}.cases") != cases:
        raise ValueError(f"{context}.cases must be {cases}")
    for field in ("baseline_feasible", "candidate_feasible"):
        if _require_int(payload.get(field), f"{context}.{field}") != cases:
            raise ValueError(f"{context}.{field} must be {cases}")
    bootstrap = _require_mapping(payload.get("bootstrap"), f"{context}.bootstrap")
    pseudo = _require_mapping(
        payload.get("pseudo_test_one_per_block_count"), f"{context}.pseudo_test"
    )
    if _require_int(bootstrap.get("samples"), f"{context}.bootstrap.samples") != BOOTSTRAP_SAMPLES:
        raise ValueError(f"{context} must use {BOOTSTRAP_SAMPLES} bootstrap samples")
    if _require_int(bootstrap.get("seed"), f"{context}.bootstrap.seed") != BOOTSTRAP_SEED:
        raise ValueError(f"{context} uses the wrong bootstrap seed")
    if bootstrap.get("cluster_unit") != "source_file":
        raise ValueError(f"{context} must bootstrap source_file clusters")
    if _require_int(pseudo.get("samples"), f"{context}.pseudo.samples") != BOOTSTRAP_SAMPLES:
        raise ValueError(f"{context} uses the wrong pseudo-test sample count")
    if _require_int(pseudo.get("seed"), f"{context}.pseudo.seed") != PSEUDO_SEED:
        raise ValueError(f"{context} uses the wrong pseudo-test seed")
    if pseudo.get("block_counts") != list(EXPECTED_BLOCK_COUNTS):
        raise ValueError(f"{context} has the wrong pseudo-test block counts")
    fold_rows = _require_list(payload.get("folds"), f"{context}.folds")
    observed_folds = [
        _require_int(row.get("fold"), f"{context}.folds[{index}].fold")
        for index, row in enumerate(fold_rows)
        if isinstance(row, dict)
    ]
    if len(observed_folds) != len(fold_rows) or observed_folds != list(folds):
        raise ValueError(f"{context} folds are {observed_folds}; expected {list(folds)}")


def _normalize_recomputed_paths(
    recomputed: dict[str, Any], stored: dict[str, Any]
) -> None:
    for role in ("baseline", "candidate"):
        for rebuilt, expected in zip(
            recomputed["input_result_artifacts"][role],
            stored["input_result_artifacts"][role],
        ):
            rebuilt["path"] = expected["path"]
    recomputed["source_manifest_artifact"]["path"] = stored[
        "source_manifest_artifact"
    ]["path"]


def _load_and_recompute_comparison(
    path: Path,
    *,
    panel: str,
    candidate_mode: str,
    folds: tuple[int, ...],
    artifact_root: Path,
    campaign: dict[str, Any],
    holdout_cache: dict[Path, dict[str, Any]],
) -> dict[str, Any]:
    raw, stored = _load_object(path, "comparison artifact")
    context = f"comparison {path}"
    _validate_comparison_shape(stored, folds=folds, context=context)
    inputs = _require_exact_keys(
        stored.get("input_result_artifacts"), {"baseline", "candidate"}, f"{context}.inputs"
    )
    baseline_descriptors = _require_list(inputs["baseline"], f"{context}.baseline")
    candidate_descriptors = _require_list(inputs["candidate"], f"{context}.candidate")
    if len(baseline_descriptors) != len(folds) or len(candidate_descriptors) != len(folds):
        raise ValueError(f"{context} must bind one baseline/candidate artifact per fold")

    manifest_descriptor, manifest_path = _validate_descriptor(
        stored.get("source_manifest_artifact"),
        artifact_root=artifact_root,
        owner_path=path,
        context=f"{context}.source_manifest_artifact",
    )
    expected_manifest = campaign["manifests"][panel]
    if manifest_path.read_bytes() != expected_manifest["raw"]:
        raise ValueError(f"{context} does not bind the frozen {panel} manifest")
    if manifest_descriptor["sha256"] != expected_manifest["sha256"]:
        raise ValueError(f"{context} has the wrong {panel} manifest digest")

    baseline_paths: list[Path] = []
    candidate_paths: list[Path] = []
    baseline_records = []
    candidate_records = []
    for index, fold in enumerate(folds):
        for role, descriptors, expected_mode, paths, records in (
            ("baseline", baseline_descriptors, "off", baseline_paths, baseline_records),
            ("candidate", candidate_descriptors, candidate_mode, candidate_paths, candidate_records),
        ):
            descriptor, holdout_path = _validate_descriptor(
                descriptors[index],
                artifact_root=artifact_root,
                owner_path=path,
                context=f"{context}.{role}[{index}]",
            )
            paths.append(holdout_path)
            if holdout_path not in holdout_cache:
                holdout_cache[holdout_path] = _validate_holdout(
                    holdout_path,
                    panel=panel,
                    mode=expected_mode,
                    fold=fold,
                    manifest_raw=expected_manifest["raw"],
                    manifest_payload=expected_manifest["payload"],
                    campaign=campaign,
                )
            record = holdout_cache[holdout_path]
            config = record["payload"]["config"]
            if config["learned_mode"] != expected_mode or config["fold"] != fold:
                raise ValueError(f"{context}.{role}[{index}] has the wrong role")
            if descriptor["sha256"] != _sha256_bytes(record["raw"]):
                raise ValueError(f"{context}.{role}[{index}] hash changed while loading")
            records.append(record)

    source_map = fold_compare._source_map(manifest_path)
    recomputed = fold_compare.compare_result_pairs(
        baseline_paths,
        candidate_paths,
        source_by_sample=source_map,
        source_manifest_path=manifest_path,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    _normalize_recomputed_paths(recomputed, stored)
    difference = _first_difference(stored, recomputed)
    if difference:
        raise ValueError(f"{context} does not match recomputation: {difference}")

    return {
        "path": path,
        "raw": raw,
        "payload": stored,
        "artifact": _artifact(path),
        "manifest_path": manifest_path,
        "manifest_descriptor": manifest_descriptor,
        "baseline_paths": baseline_paths,
        "candidate_paths": candidate_paths,
        "baseline_records": baseline_records,
        "candidate_records": candidate_records,
    }


def _check(
    name: str,
    observed: Any,
    operator: str,
    threshold: Any,
    passed: bool,
) -> dict[str, Any]:
    return {
        "name": name,
        "observed": observed,
        "operator": operator,
        "threshold": threshold,
        "passed": bool(passed),
    }


def _panel_checks(payload: dict[str, Any], panel: str, stage: str) -> list[dict[str, Any]]:
    delta = _require_number(payload["delta_candidate_minus_baseline"], "comparison delta")
    bootstrap = payload["bootstrap"]
    pseudo = payload["pseudo_test_one_per_block_count"]
    tail = payload["tail_risk"]
    bootstrap_upper = _require_number(bootstrap["delta_ci95"][1], "bootstrap CI upper")
    pseudo_upper = _require_number(pseudo["delta_ci95"][1], "pseudo CI upper")
    probability = _require_number(
        bootstrap["probability_candidate_improves"], "bootstrap probability", minimum=0.0
    )
    worst_pseudo = _require_number(pseudo["worst_sampled_delta"], "worst pseudo delta")
    worst_case = _require_number(
        tail["worst_case_score_contribution"], "worst case contribution"
    )
    cvar = _require_number(
        tail["regression_cvar_5pct_score_contribution"], "regression CVaR"
    )
    checks = [
        _check(
            "full_feasibility",
            payload["candidate_feasible"],
            "==",
            payload["cases"],
            payload["candidate_feasible"] == payload["cases"],
        )
    ]
    threshold = THRESHOLDS[stage][panel]
    if stage == "development" and panel == "clean":
        checks.extend(
            [
                _check("pooled_delta", delta, "<=", threshold["maximum_pooled_delta"], delta <= threshold["maximum_pooled_delta"]),
                _check("bootstrap_probability_improves", probability, ">=", threshold["minimum_bootstrap_probability_improves"], probability >= threshold["minimum_bootstrap_probability_improves"]),
                _check("bootstrap_ci95_upper", bootstrap_upper, "<=", threshold["maximum_bootstrap_ci95_upper"], bootstrap_upper <= threshold["maximum_bootstrap_ci95_upper"]),
                _check("pseudo_ci95_upper", pseudo_upper, "<=", threshold["maximum_pseudo_ci95_upper"], pseudo_upper <= threshold["maximum_pseudo_ci95_upper"]),
                _check("maximum_fold_delta", max(row["delta_candidate_minus_baseline"] for row in payload["folds"]), "<=", threshold["maximum_fold_delta"], max(row["delta_candidate_minus_baseline"] for row in payload["folds"]) <= threshold["maximum_fold_delta"]),
            ]
        )
    elif stage == "development" and panel == "raw":
        checks.extend(
            [
                _check("pooled_delta", delta, "<=", threshold["maximum_pooled_delta"], delta <= threshold["maximum_pooled_delta"]),
                _check("bootstrap_ci95_upper", bootstrap_upper, "<=", threshold["maximum_bootstrap_ci95_upper"], bootstrap_upper <= threshold["maximum_bootstrap_ci95_upper"]),
                _check("pseudo_ci95_upper", pseudo_upper, "<=", threshold["maximum_pseudo_ci95_upper"], pseudo_upper <= threshold["maximum_pseudo_ci95_upper"]),
                _check("maximum_fold_delta", max(row["delta_candidate_minus_baseline"] for row in payload["folds"]), "<=", threshold["maximum_fold_delta"], max(row["delta_candidate_minus_baseline"] for row in payload["folds"]) <= threshold["maximum_fold_delta"]),
            ]
        )
    elif stage == "calibration" and panel == "clean":
        checks.extend(
            [
                _check("pooled_delta", delta, "<", threshold["maximum_pooled_delta_exclusive"], delta < threshold["maximum_pooled_delta_exclusive"]),
                _check("bootstrap_probability_improves", probability, ">=", threshold["minimum_bootstrap_probability_improves"], probability >= threshold["minimum_bootstrap_probability_improves"]),
                _check("pseudo_ci95_upper", pseudo_upper, "<=", threshold["maximum_pseudo_ci95_upper"], pseudo_upper <= threshold["maximum_pseudo_ci95_upper"]),
            ]
        )
    else:
        checks.extend(
            [
                _check("pooled_delta", delta, "<=", threshold["maximum_pooled_delta"], delta <= threshold["maximum_pooled_delta"]),
                _check("bootstrap_ci95_upper", bootstrap_upper, "<=", threshold["maximum_bootstrap_ci95_upper"], bootstrap_upper <= threshold["maximum_bootstrap_ci95_upper"]),
                _check("pseudo_ci95_upper", pseudo_upper, "<=", threshold["maximum_pseudo_ci95_upper"], pseudo_upper <= threshold["maximum_pseudo_ci95_upper"]),
            ]
        )
    checks.extend(
        [
            _check("worst_case_score_contribution", worst_case, "<=", threshold["maximum_worst_case_score_contribution"], worst_case <= threshold["maximum_worst_case_score_contribution"]),
            _check("regression_cvar_5pct", cvar, "<=", threshold["maximum_regression_cvar_5pct"], cvar <= threshold["maximum_regression_cvar_5pct"]),
            _check("worst_sampled_pseudo_delta", worst_pseudo, "<=", threshold["maximum_worst_sampled_pseudo_delta"], worst_pseudo <= threshold["maximum_worst_sampled_pseudo_delta"]),
        ]
    )
    return checks


def _gate_result(payload: dict[str, Any], panel: str, stage: str) -> dict[str, Any]:
    checks = _panel_checks(payload, panel, stage)
    reasons = [check["name"] for check in checks if not check["passed"]]
    return {
        "passed": not reasons,
        "reasons": reasons,
        "checks": checks,
        "metrics": {
            "baseline_score": payload["baseline_score"],
            "candidate_score": payload["candidate_score"],
            "delta_candidate_minus_baseline": payload["delta_candidate_minus_baseline"],
            "wins": payload["wins"],
            "losses": payload["losses"],
            "ties": payload["ties"],
            "bootstrap": copy.deepcopy(payload["bootstrap"]),
            "pseudo_test_one_per_block_count": copy.deepcopy(
                payload["pseudo_test_one_per_block_count"]
            ),
            "tail_risk": copy.deepcopy(payload["tail_risk"]),
            "folds": copy.deepcopy(payload["folds"]),
        },
    }


def _direct_comparison(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    return fold_compare.compare_result_pairs(
        baseline["candidate_paths"],
        candidate["candidate_paths"],
        source_by_sample=fold_compare._source_map(candidate["manifest_path"]),
        source_manifest_path=candidate["manifest_path"],
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )


def _direct_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline_score": payload["baseline_score"],
        "candidate_score": payload["candidate_score"],
        "delta_candidate_minus_baseline": payload["delta_candidate_minus_baseline"],
        "bootstrap": copy.deepcopy(payload["bootstrap"]),
        "pseudo_test_one_per_block_count": copy.deepcopy(
            payload["pseudo_test_one_per_block_count"]
        ),
        "tail_risk": copy.deepcopy(payload["tail_risk"]),
    }


def _select_finalist(
    comparisons: dict[str, dict[str, dict[str, Any]]],
    gates: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    passing = [mode for mode in CHALLENGERS if gates[mode]["passed"]]
    if not passing:
        return "off", {
            "passing_challengers": [],
            "ranking": [],
            "direct_comparisons": {},
            "reason": "no challenger passed both development panels",
        }
    ranking = sorted(
        passing,
        key=lambda mode: (
            comparisons[mode]["clean"]["payload"]["candidate_score"],
            COMPLEXITY_ORDER[mode],
        ),
    )
    if len(ranking) == 1:
        return ranking[0], {
            "passing_challengers": passing,
            "ranking": ranking,
            "direct_comparisons": {},
            "reason": "only challenger passing both development panels",
        }

    top = ranking[0]
    tied = {top}
    direct_records: dict[str, Any] = {}
    threshold = THRESHOLDS["development"]["direct_ranking"]
    for other in ranking[1:]:
        clean = _direct_comparison(comparisons[other]["clean"], comparisons[top]["clean"])
        raw = _direct_comparison(comparisons[other]["raw"], comparisons[top]["raw"])
        clean_delta = clean["delta_candidate_minus_baseline"]
        clean_probability = clean["bootstrap"]["probability_candidate_improves"]
        raw_delta = raw["delta_candidate_minus_baseline"]
        raw_upper = raw["bootstrap"]["delta_ci95"][1]
        decisive = (
            clean_delta <= -threshold["minimum_clean_improvement"]
            and clean_probability
            >= threshold["minimum_clean_bootstrap_probability_improves"]
            and raw_delta <= threshold["maximum_raw_delta"]
            and raw_upper <= threshold["maximum_raw_bootstrap_ci95_upper"]
        )
        direct_records[f"{top}_vs_{other}"] = {
            "decisive": decisive,
            "checks": {
                "clean_improvement": -clean_delta,
                "clean_bootstrap_probability_improves": clean_probability,
                "raw_delta": raw_delta,
                "raw_bootstrap_ci95_upper": raw_upper,
            },
            "clean": _direct_summary(clean),
            "raw": _direct_summary(raw),
        }
        if not decisive:
            tied.add(other)
    selected = min(tied, key=lambda mode: COMPLEXITY_ORDER[mode])
    reason = (
        "lowest clean score decisively beat every other passing challenger"
        if tied == {top}
        else "statistically tied challengers resolved by predeclared complexity order"
    )
    return selected, {
        "passing_challengers": passing,
        "ranking": ranking,
        "statistically_tied": sorted(tied, key=lambda mode: COMPLEXITY_ORDER[mode]),
        "direct_comparisons": direct_records,
        "reason": reason,
    }


def _validate_matrix_contracts(
    comparisons: dict[str, dict[str, dict[str, Any]]],
    *,
    folds: tuple[int, ...],
    expected_unique_artifacts: int,
    modes: tuple[str, ...] = CHALLENGERS,
) -> None:
    unique_paths: set[Path] = set()
    for panel in PANELS:
        canonical_baseline: list[Path] | None = None
        golden_by_fold: list[dict[str, Any]] | None = None
        identities_by_fold: list[list[tuple[Any, ...]]] | None = None
        for mode in modes:
            record = comparisons[mode][panel]
            if canonical_baseline is None:
                canonical_baseline = record["baseline_paths"]
                identities_by_fold = [
                    item["identity_contract"] for item in record["baseline_records"]
                ]
                golden_by_fold = [
                    item["golden_contract"] for item in record["baseline_records"]
                ]
            elif record["baseline_paths"] != canonical_baseline:
                raise ValueError(f"{panel} comparisons do not reuse one canonical off baseline")
            for path in record["baseline_paths"] + record["candidate_paths"]:
                unique_paths.add(path.resolve())
            mode_identities = [item["identity_contract"] for item in record["candidate_records"]]
            mode_golden = [item["golden_contract"] for item in record["candidate_records"]]
            if mode_identities != identities_by_fold or mode_golden != golden_by_fold:
                raise ValueError(f"{panel} modes do not bind identical cases/golden summaries")
        if canonical_baseline is None or len(canonical_baseline) != len(folds):
            raise ValueError(f"{panel} comparison matrix is incomplete")
        for mode in modes:
            if set(comparisons[mode][panel]["candidate_paths"]) & set(canonical_baseline):
                raise ValueError(f"{panel} candidate reuses an off artifact path")
    if len(unique_paths) != expected_unique_artifacts:
        raise ValueError(
            f"tournament binds {len(unique_paths)} unique holdouts; "
            f"expected {expected_unique_artifacts}"
        )


def select_policy_tournament(
    *,
    development_paths: dict[str, dict[str, Path]],
    expected_commit: str,
    artifact_root: Path,
    calibration_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    campaign = _campaign_contract(expected_commit)
    holdout_cache: dict[Path, dict[str, Any]] = {}
    comparisons: dict[str, dict[str, dict[str, Any]]] = {
        mode: {} for mode in CHALLENGERS
    }
    seen_comparison_paths: set[Path] = set()
    for mode in CHALLENGERS:
        if set(development_paths.get(mode, {})) != set(PANELS):
            raise ValueError(f"development comparison matrix is incomplete for {mode}")
        for panel in PANELS:
            path = development_paths[mode][panel].resolve()
            if path in seen_comparison_paths:
                raise ValueError(f"comparison artifact is reused: {path}")
            seen_comparison_paths.add(path)
            comparisons[mode][panel] = _load_and_recompute_comparison(
                path,
                panel=panel,
                candidate_mode=mode,
                folds=DEV_FOLDS,
                artifact_root=artifact_root,
                campaign=campaign,
                holdout_cache=holdout_cache,
            )
    _validate_matrix_contracts(
        comparisons, folds=DEV_FOLDS, expected_unique_artifacts=24
    )

    gates: dict[str, dict[str, Any]] = {}
    for mode in CHALLENGERS:
        panel_results = {
            panel: _gate_result(comparisons[mode][panel]["payload"], panel, "development")
            for panel in PANELS
        }
        reasons = [
            f"{panel}.{reason}"
            for panel in PANELS
            for reason in panel_results[panel]["reasons"]
        ]
        gates[mode] = {
            "passed": not reasons,
            "reasons": reasons,
            "panels": panel_results,
        }
    dev_finalist, selection = _select_finalist(comparisons, gates)

    holdout_artifacts: dict[str, dict[str, dict[str, Any]]] = {
        panel: {mode: {} for mode in MODES} for panel in PANELS
    }
    for panel in PANELS:
        reference = comparisons[CHALLENGERS[0]][panel]
        for fold, record in zip(DEV_FOLDS, reference["baseline_records"]):
            holdout_artifacts[panel]["off"][str(fold)] = record["artifact"]
        for mode in CHALLENGERS:
            for fold, record in zip(DEV_FOLDS, comparisons[mode][panel]["candidate_records"]):
                holdout_artifacts[panel][mode][str(fold)] = record["artifact"]

    ledger: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "selector": {
            "path": "scripts/select_policy_tournament.py",
            "sha256": _sha256_file(Path(__file__)),
        },
        "comparison_harness": {
            "path": "scripts/compare_fold_results.py",
            "sha256": campaign["comparison_harness_sha256"],
        },
        "holdout_harness": {
            "path": "scripts/evaluate_training_holdout.py",
            "sha256": campaign["evaluation_harness_sha256"],
        },
        "expected_solver_commit": campaign["expected_commit"],
        "official_sources": campaign["official_sources"],
        "thresholds": copy.deepcopy(THRESHOLDS),
        "bootstrap_contract": {
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "cluster_unit": "source_file",
            "pseudo_seed": PSEUDO_SEED,
        },
        "manifest_artifacts": {
            panel: {
                "path": campaign["manifests"][panel]["relative_path"],
                "sha256": campaign["manifests"][panel]["sha256"],
                "size_bytes": len(campaign["manifests"][panel]["raw"]),
            }
            for panel in PANELS
        },
        "development": {
            "folds": list(DEV_FOLDS),
            "comparison_artifacts": {
                mode: {
                    panel: comparisons[mode][panel]["artifact"] for panel in PANELS
                }
                for mode in CHALLENGERS
            },
            "holdout_artifacts": holdout_artifacts,
            "challengers": gates,
            "selection": selection,
        },
        "dev_finalist": dev_finalist,
        "calibration": None,
        "final_mode": "off" if dev_finalist == "off" else None,
        "status": "off_selected" if dev_finalist == "off" else "requires_calibration",
    }

    if calibration_paths is not None:
        if dev_finalist == "off":
            raise ValueError("fold-3 evidence is not permitted when development selected off")
        if set(calibration_paths) != set(PANELS):
            raise ValueError("both clean and raw fold-3 comparisons are required")
        calibration: dict[str, dict[str, Any]] = {}
        calibration_seen: set[Path] = set()
        for panel in PANELS:
            path = calibration_paths[panel].resolve()
            if path in calibration_seen or path in seen_comparison_paths:
                raise ValueError(f"calibration comparison artifact is reused: {path}")
            calibration_seen.add(path)
            calibration[panel] = _load_and_recompute_comparison(
                path,
                panel=panel,
                candidate_mode=dev_finalist,
                folds=(CALIBRATION_FOLD,),
                artifact_root=artifact_root,
                campaign=campaign,
                holdout_cache=holdout_cache,
            )
            if (
                calibration[panel]["payload"]["evaluation_contract"]
                != comparisons[dev_finalist][panel]["payload"]["evaluation_contract"]
            ):
                raise ValueError(f"{panel} fold-3 evaluation contract differs from development")
        _validate_matrix_contracts(
            {dev_finalist: calibration},
            folds=(CALIBRATION_FOLD,),
            expected_unique_artifacts=4,
            modes=(dev_finalist,),
        )
        panel_results = {
            panel: _gate_result(calibration[panel]["payload"], panel, "calibration")
            for panel in PANELS
        }
        reasons = [
            f"{panel}.{reason}"
            for panel in PANELS
            for reason in panel_results[panel]["reasons"]
        ]
        passed = not reasons
        ledger["calibration"] = {
            "fold": CALIBRATION_FOLD,
            "comparison_artifacts": {
                panel: calibration[panel]["artifact"] for panel in PANELS
            },
            "holdout_artifacts": {
                panel: {
                    "off": calibration[panel]["baseline_records"][0]["artifact"],
                    dev_finalist: calibration[panel]["candidate_records"][0]["artifact"],
                }
                for panel in PANELS
            },
            "passed": passed,
            "reasons": reasons,
            "panels": panel_results,
            "fallback_mode": "off",
        }
        ledger["final_mode"] = dev_finalist if passed else "off"
        ledger["status"] = (
            "calibration_passed" if passed else "calibration_failed_fallback_off"
        )

    return ledger


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o644)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"output already exists: {path}") from exc
        temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def _development_paths_from_args(args: argparse.Namespace) -> dict[str, dict[str, Path]]:
    return {
        "replacement": {
            "clean": args.clean_replacement,
            "raw": args.raw_replacement,
        },
        "additive": {
            "clean": args.clean_additive,
            "raw": args.raw_additive,
        },
        "additive_first_pass": {
            "clean": args.clean_additive_first_pass,
            "raw": args.raw_additive_first_pass,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-solver-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, default=ROOT)
    for panel in PANELS:
        for mode in CHALLENGERS:
            parser.add_argument(
                f"--{panel}-{mode.replace('_', '-')}",
                dest=f"{panel}_{mode}",
                type=Path,
                required=True,
            )
    parser.add_argument("--fold3-clean", type=Path)
    parser.add_argument("--fold3-raw", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.fold3_clean is None) != (args.fold3_raw is None):
        parser.error("--fold3-clean and --fold3-raw must be supplied together")

    try:
        ledger = select_policy_tournament(
            development_paths=_development_paths_from_args(args),
            expected_commit=args.expected_solver_commit,
            artifact_root=args.artifact_root.resolve(),
            calibration_paths=(
                {"clean": args.fold3_clean, "raw": args.fold3_raw}
                if args.fold3_clean is not None
                else None
            ),
        )
        _write_json_exclusive(args.output, ledger)
    except Exception as exc:
        print(f"Policy tournament selection: FAIL\n  {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(
        json.dumps(
            {
                "status": ledger["status"],
                "dev_finalist": ledger["dev_finalist"],
                "final_mode": ledger["final_mode"],
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
