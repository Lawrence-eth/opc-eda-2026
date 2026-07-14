#!/usr/bin/env python3
"""Compose the derived slot map and public rejection audit into final policy.

This is a provenance gate, not an evaluator and not a policy tuner.  It accepts
only the frozen schema-2 derivation artifact and the schema-1 public fidelity
audit produced for those exact bytes.  Public evidence can remove a derived
mapping, but it can never add a size or change a width factor.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
HEAVY_SIZES = tuple(range(100, 121))
SLOTS = (0.8, 0.9, 1.0, 1.1, 1.2)
SLOT_KEYS = tuple(str(slot) for slot in SLOTS)
DERIVATION_MODE = "legacy_v32_final_output_preserving_slot_calibration"
PUBLIC_AUDIT_MODE = "public_v32_final_output_slot_removal_rejection"
PUBLIC_POLICY = "rejection_only_no_slot_retuning"
EVIDENCE_STATUS = "final_rejection_only_public_audit_complete"
DERIVATION_HARNESS = ROOT / "scripts" / "derive_redundant_slot_map.py"
SLOT_AUDIT_HARNESS = ROOT / "scripts" / "audit_standard_slot_usage.py"
PUBLIC_AUDIT_HARNESS = ROOT / "scripts" / "audit_public_slot_fidelity.py"
FINALIZER_PATH = ROOT / "scripts" / "finalize_learned_policy.py"


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return resolved.name


def _require_exact_keys(value: Any, expected: set[str], context: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        observed = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(
            f"{context} must have exactly {sorted(expected)}; observed {observed}"
        )
    return value


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
        raise ValueError(f"{context} must be a full lowercase git commit")
    return value


def _require_int(value: Any, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{context} must be an integer >= {minimum}")
    return value


def _require_bool(value: Any, context: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{context} must be boolean")
    return value


def _require_slot(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a numeric standard slot")
    slot = float(value)
    if not math.isfinite(slot) or slot not in SLOTS:
        raise ValueError(f"{context} must be a standard paid width factor")
    return slot


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _load_object(path: Path, context: str) -> tuple[bytes, dict]:
    try:
        raw = path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"cannot load {context} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must be a JSON object")
    return raw, payload


def _size_map(value: Any, context: str, *, allow_empty: bool = True) -> dict[str, float]:
    if not isinstance(value, dict) or (not allow_empty and not value):
        qualifier = "nonempty " if not allow_empty else ""
        raise ValueError(f"{context} must be a {qualifier}object")
    result = {}
    for raw_size, raw_slot in value.items():
        if (
            not isinstance(raw_size, str)
            or not raw_size.isdigit()
            or str(int(raw_size)) != raw_size
        ):
            raise ValueError(f"{context} contains a non-canonical block count")
        size = int(raw_size)
        if size not in HEAVY_SIZES:
            raise ValueError(f"{context} contains non-heavy size {size}")
        result[raw_size] = _require_slot(raw_slot, f"{context}[{raw_size}]")
    return result


def _size_list(value: Any, context: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    result = [
        _require_int(item, f"{context}[{index}]", minimum=100)
        for index, item in enumerate(value)
    ]
    if any(item not in HEAVY_SIZES for item in result):
        raise ValueError(f"{context} contains a non-heavy size")
    if result != sorted(set(result)):
        raise ValueError(f"{context} must be sorted and unique")
    return result


def _count_matrix(value: Any, context: str, *, maximum: int) -> dict[str, dict[str, int]]:
    _require_exact_keys(value, {str(size) for size in HEAVY_SIZES}, context)
    result = {}
    for size in HEAVY_SIZES:
        row = _require_exact_keys(value[str(size)], set(SLOT_KEYS), f"{context}[{size}]")
        result[str(size)] = {}
        for slot in SLOT_KEYS:
            count = _require_int(row[slot], f"{context}[{size}][{slot}]")
            if count > maximum:
                raise ValueError(f"{context}[{size}][{slot}] exceeds {maximum}")
            result[str(size)][slot] = count
    return result


def _validate_descriptor(value: Any, context: str, allowed_folds: set[int]) -> tuple[str, int]:
    descriptor = _require_exact_keys(
        value,
        {
            "path",
            "sha256",
            "fold",
            "domain",
            "manifest_sha256",
            "case_count",
            "source_file_count",
        },
        context,
    )
    if not isinstance(descriptor["path"], str) or not descriptor["path"]:
        raise ValueError(f"{context}.path must be a nonempty string")
    _require_sha256(descriptor["sha256"], f"{context}.sha256")
    _require_sha256(descriptor["manifest_sha256"], f"{context}.manifest_sha256")
    fold = _require_int(descriptor["fold"], f"{context}.fold")
    if fold not in allowed_folds:
        raise ValueError(f"{context} has unexpected fold {fold}")
    domain = descriptor["domain"]
    if domain not in {"mib_input_compatible", "raw_hash"}:
        raise ValueError(f"{context} has unexpected domain {domain!r}")
    if _require_int(descriptor["case_count"], f"{context}.case_count") != 105:
        raise ValueError(f"{context} must bind exactly 105 cases")
    _require_int(descriptor["source_file_count"], f"{context}.source_file_count", minimum=1)
    return domain, fold


def _expected_components() -> tuple[str, ...]:
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    try:
        from solver_components import (  # pylint: disable=import-outside-toplevel
            LIVE_SOLVER_COMPONENTS,
            validate_live_solver_components,
        )
    except ImportError as error:
        raise ValueError("authoritative live solver component registry is unavailable") from error
    return validate_live_solver_components(LIVE_SOLVER_COMPONENTS)


def _validate_solver_binding(value: Any) -> dict:
    binding = _require_exact_keys(
        value, {"commit", "components", "audit_harness_sha256"}, "solver_binding"
    )
    _require_commit(binding["commit"], "solver_binding.commit")
    components = binding["components"]
    expected = _expected_components()
    if not isinstance(components, dict) or set(components) != set(expected):
        raise ValueError("solver_binding has an incomplete live component set")
    for name in expected:
        digest = _require_sha256(components[name], f"solver_binding.components[{name}]")
        component_path = ROOT / "contest_solution" / name
        if not component_path.is_file() or digest != _sha256_file(component_path):
            raise ValueError(f"solver binding does not match live component bytes: {name}")
    harness_digest = _require_sha256(
        binding["audit_harness_sha256"], "solver_binding.audit_harness_sha256"
    )
    if harness_digest != _sha256_file(SLOT_AUDIT_HARNESS):
        raise ValueError("solver binding does not match the slot-audit harness bytes")
    return copy.deepcopy(binding)


def _validate_derivation(payload: dict) -> dict[str, Any]:
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "mode",
            "provenance",
            "contract",
            "solver_binding",
            "development_artifacts",
            "confirmation_artifacts",
            "total_selected_counts",
            "tie_preference",
            "selected_counts_by_size",
            "nonpreserved_counts_by_size",
            "derivation_case_counts_by_size",
            "derivation_unique_source_counts_by_size",
            "derived_replacement_wf_by_size",
            "development_abstain_sizes",
            "confirmation_rejections",
            "confirmation_rejected_sizes",
            "replacement_wf_by_size",
            "abstain_sizes",
        },
        "derivation artifact",
    )
    if payload["schema_version"] != 2 or payload["mode"] != DERIVATION_MODE:
        raise ValueError("derivation artifact must use the frozen schema-2 contract")

    provenance = _require_exact_keys(
        payload["provenance"], {"derivation_harness"}, "derivation provenance"
    )
    harness = _require_exact_keys(
        provenance["derivation_harness"], {"path", "sha256"}, "derivation harness"
    )
    if harness["path"] != "scripts/derive_redundant_slot_map.py":
        raise ValueError("derivation artifact names an unexpected harness")
    if _require_sha256(harness["sha256"], "derivation harness SHA-256") != _sha256_file(
        DERIVATION_HARNESS
    ):
        raise ValueError("derivation artifact does not match the current harness bytes")

    contract = _require_exact_keys(
        payload["contract"],
        {
            "derivation_folds",
            "derivation_domains",
            "confirmation_fold",
            "confirmation_policy",
            "uses_learned_outputs",
            "uses_golden_costs",
            "uses_public_cases",
            "invariant",
            "comparison_stage",
            "removal_method",
            "tie_rule",
            "required_development_support_per_size",
        },
        "derivation contract",
    )
    expected_contract = {
        "derivation_folds": [0, 1, 2],
        "derivation_domains": ["mib_input_compatible", "raw_hash"],
        "confirmation_fold": 3,
        "confirmation_policy": PUBLIC_POLICY,
        "uses_learned_outputs": False,
        "uses_golden_costs": False,
        "uses_public_cases": False,
        "comparison_stage": "complete_deployed_solver_output",
        "removal_method": "full solver rerun for every paid standard slot",
        "tie_rule": "ascending(total_selected_count, numeric_width_factor)",
        "required_development_support_per_size": 30,
    }
    for field, expected in expected_contract.items():
        if contract[field] != expected:
            raise ValueError(f"derivation contract has unexpected {field}")
    if not isinstance(contract["invariant"], str) or "packed-byte final positions" not in contract[
        "invariant"
    ]:
        raise ValueError("derivation contract has an unexpected output invariant")

    binding = _validate_solver_binding(payload["solver_binding"])

    development = payload["development_artifacts"]
    confirmation = payload["confirmation_artifacts"]
    if not isinstance(development, list) or len(development) != 6:
        raise ValueError("derivation artifact must bind six development panels")
    if not isinstance(confirmation, list) or len(confirmation) != 2:
        raise ValueError("derivation artifact must bind two confirmation panels")
    development_roles = {
        _validate_descriptor(row, f"development_artifacts[{index}]", {0, 1, 2})
        for index, row in enumerate(development)
    }
    confirmation_roles = {
        _validate_descriptor(row, f"confirmation_artifacts[{index}]", {3})
        for index, row in enumerate(confirmation)
    }
    if development_roles != {
        (domain, fold)
        for domain in ("mib_input_compatible", "raw_hash")
        for fold in (0, 1, 2)
    }:
        raise ValueError("development panel roles are incomplete or duplicated")
    if confirmation_roles != {
        ("mib_input_compatible", 3),
        ("raw_hash", 3),
    }:
        raise ValueError("confirmation panel roles are incomplete or duplicated")

    support = _require_exact_keys(
        payload["derivation_case_counts_by_size"],
        {str(size) for size in HEAVY_SIZES},
        "derivation_case_counts_by_size",
    )
    sources = _require_exact_keys(
        payload["derivation_unique_source_counts_by_size"],
        {str(size) for size in HEAVY_SIZES},
        "derivation_unique_source_counts_by_size",
    )
    for size in HEAVY_SIZES:
        if _require_int(support[str(size)], f"derivation support for {size}") != 30:
            raise ValueError(f"derivation support for {size} is not 30")
        source_count = _require_int(sources[str(size)], f"derivation sources for {size}", minimum=1)
        if source_count > 30:
            raise ValueError(f"derivation source count for {size} exceeds support")

    selected = _count_matrix(
        payload["selected_counts_by_size"], "selected_counts_by_size", maximum=30
    )
    nonpreserved = _count_matrix(
        payload["nonpreserved_counts_by_size"],
        "nonpreserved_counts_by_size",
        maximum=30,
    )
    totals = _require_exact_keys(
        payload["total_selected_counts"], set(SLOT_KEYS), "total_selected_counts"
    )
    normalized_totals = {}
    for slot in SLOT_KEYS:
        normalized_totals[slot] = _require_int(totals[slot], f"total_selected_counts[{slot}]")
        if normalized_totals[slot] != sum(selected[str(size)][slot] for size in HEAVY_SIZES):
            raise ValueError(f"selected count total mismatch for slot {slot}")
    preference_raw = payload["tie_preference"]
    if not isinstance(preference_raw, list):
        raise ValueError("tie_preference must be a list")
    preference = [_require_slot(value, "tie_preference") for value in preference_raw]
    expected_preference = sorted(SLOTS, key=lambda slot: (normalized_totals[str(slot)], slot))
    if preference != expected_preference:
        raise ValueError("tie_preference does not follow the frozen tie rule")

    derived = _size_map(payload["derived_replacement_wf_by_size"], "derived map")
    expected_derived = {}
    for size in HEAVY_SIZES:
        chosen = next(
            (slot for slot in preference if nonpreserved[str(size)][str(slot)] == 0),
            None,
        )
        if chosen is not None:
            expected_derived[str(size)] = chosen
    if derived != expected_derived:
        raise ValueError("derived map does not follow the recorded counts and tie rule")
    development_abstain = _size_list(
        payload["development_abstain_sizes"], "development_abstain_sizes"
    )
    if development_abstain != [size for size in HEAVY_SIZES if str(size) not in derived]:
        raise ValueError("development abstentions are not the complement of the derived map")

    confirmation_rejected = _size_list(
        payload["confirmation_rejected_sizes"], "confirmation_rejected_sizes"
    )
    rejection_rows = payload["confirmation_rejections"]
    if not isinstance(rejection_rows, list):
        raise ValueError("confirmation_rejections must be a list")
    observed_rejections = []
    for index, row in enumerate(rejection_rows):
        row = _require_exact_keys(
            row,
            {"block_count", "derived_width_factor", "failures"},
            f"confirmation_rejections[{index}]",
        )
        size = _require_int(row["block_count"], f"confirmation rejection {index} size", minimum=100)
        if str(size) not in derived:
            raise ValueError("confirmation rejected a size absent from the derived map")
        if _require_slot(row["derived_width_factor"], "confirmation rejected slot") != derived[str(size)]:
            raise ValueError("confirmation rejection retunes the derived slot")
        failures = row["failures"]
        if not isinstance(failures, list) or not failures:
            raise ValueError("confirmation rejection must contain at least one failure")
        for failure_index, failure in enumerate(failures):
            failure = _require_exact_keys(
                failure,
                {"domain", "case_id", "final_positions_equal", "visible_metrics_equal"},
                f"confirmation rejection {index} failure {failure_index}",
            )
            if failure["domain"] not in {"mib_input_compatible", "raw_hash"}:
                raise ValueError("confirmation rejection has an unknown domain")
            if not isinstance(failure["case_id"], str) or not failure["case_id"]:
                raise ValueError("confirmation rejection has no case identity")
            positions_equal = _require_bool(failure["final_positions_equal"], "confirmation position verdict")
            metrics_equal = _require_bool(failure["visible_metrics_equal"], "confirmation metric verdict")
            if positions_equal and metrics_equal:
                raise ValueError("confirmation failure records a preserved output")
        observed_rejections.append(size)
    if observed_rejections != confirmation_rejected:
        raise ValueError("confirmation rejection rows and rejected-size summary disagree")

    replacement = _size_map(
        payload["replacement_wf_by_size"], "confirmed replacement map", allow_empty=False
    )
    expected_replacement = {
        size: slot for size, slot in derived.items() if int(size) not in confirmation_rejected
    }
    if replacement != expected_replacement:
        raise ValueError("confirmation may only remove, never add or retune, derived mappings")
    abstain = _size_list(payload["abstain_sizes"], "abstain_sizes")
    if abstain != [size for size in HEAVY_SIZES if str(size) not in replacement]:
        raise ValueError("confirmed abstentions are not the complement of the replacement map")

    return {
        "binding": binding,
        "derived": derived,
        "development_abstain": development_abstain,
        "confirmation_rejected": confirmation_rejected,
        "replacement": replacement,
        "abstain": abstain,
    }


def _visible_metrics(value: Any, context: str) -> dict[str, Any]:
    metrics = _require_exact_keys(
        value, {"feasible", "hpwl", "area", "soft_violations"}, context
    )
    feasible = _require_bool(metrics["feasible"], f"{context}.feasible")
    soft = _require_int(metrics["soft_violations"], f"{context}.soft_violations")
    normalized = {"feasible": feasible, "soft_violations": soft}
    for field in ("hpwl", "area"):
        raw = metrics[field]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"{context}.{field} must be numeric")
        number = float(raw)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"{context}.{field} must be finite and nonnegative")
        normalized[field] = number
    return normalized


def _metrics_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left["feasible"] == right["feasible"]
        and left["soft_violations"] == right["soft_violations"]
        and math.isclose(left["hpwl"], right["hpwl"], rel_tol=0.0, abs_tol=1e-9)
        and math.isclose(left["area"], right["area"], rel_tol=0.0, abs_tol=1e-9)
    )


def _validate_public_audit(payload: dict, slot_map_sha256: str, derivation: dict) -> dict:
    _require_exact_keys(
        payload,
        {"schema_version", "mode", "config", "provenance", "summary", "cases"},
        "public audit",
    )
    if payload["schema_version"] != 1 or payload["mode"] != PUBLIC_AUDIT_MODE:
        raise ValueError("public audit must use the frozen schema-1 rejection contract")
    config = _require_exact_keys(
        payload["config"],
        {"solver_dir", "slot_map_sha256", "uses_golden_costs", "policy"},
        "public audit config",
    )
    if not isinstance(config["solver_dir"], str) or not config["solver_dir"]:
        raise ValueError("public audit solver_dir must be a nonempty string")
    if _require_sha256(config["slot_map_sha256"], "public audit slot-map SHA-256") != slot_map_sha256:
        raise ValueError("public audit is not bound to the exact derivation input bytes")
    if config["uses_golden_costs"] is not False or config["policy"] != PUBLIC_POLICY:
        raise ValueError("public audit is not rejection-only and golden-cost-free")

    provenance = _require_exact_keys(
        payload["provenance"],
        {"harness_sha256", "solver_components", "solver_git"},
        "public audit provenance",
    )
    harness_sha = _require_sha256(provenance["harness_sha256"], "public audit harness")
    if harness_sha != _sha256_file(PUBLIC_AUDIT_HARNESS):
        raise ValueError("public audit does not match the current harness bytes")
    if provenance["solver_components"] != derivation["binding"]["components"]:
        raise ValueError("public audit and derivation have different solver components")
    for name, digest in provenance["solver_components"].items():
        _require_sha256(digest, f"public audit solver component {name}")
    solver_git = _require_exact_keys(
        provenance["solver_git"],
        {"commit", "dirty", "tracked_dirty", "has_untracked"},
        "public audit solver git",
    )
    if _require_commit(solver_git["commit"], "public audit solver commit") != derivation[
        "binding"
    ]["commit"]:
        raise ValueError("public audit and derivation have different solver commits")
    if any(solver_git[field] is not False for field in ("dirty", "tracked_dirty", "has_untracked")):
        raise ValueError("public audit solver provenance is not clean")

    cases = payload["cases"]
    expected_map = derivation["replacement"]
    if not isinstance(cases, list) or len(cases) != len(expected_map):
        raise ValueError("public audit must contain one case for every confirmed mapping")
    observed_sizes = set()
    observed_test_ids = set()
    rejected = []
    for index, row in enumerate(cases):
        row = _require_exact_keys(
            row,
            {
                "test_id",
                "block_count",
                "removed_width_factor",
                "control_positions_sha256",
                "removed_positions_sha256",
                "final_positions_equal",
                "control_visible_metrics",
                "removed_visible_metrics",
                "visible_metrics_equal",
                "final_preserved",
            },
            f"public audit case {index}",
        )
        test_id = _require_int(row["test_id"], f"public audit case {index} test_id")
        size = _require_int(row["block_count"], f"public audit case {index} block_count", minimum=100)
        if test_id in observed_test_ids or size in observed_sizes:
            raise ValueError("public audit contains a duplicate test ID or block count")
        observed_test_ids.add(test_id)
        observed_sizes.add(size)
        if str(size) not in expected_map:
            raise ValueError("public audit contains a case outside the confirmed map")
        if _require_slot(row["removed_width_factor"], "public removed width factor") != expected_map[str(size)]:
            raise ValueError("public audit retuned a confirmed width factor")
        control_sha = _require_sha256(row["control_positions_sha256"], "public control positions")
        removed_sha = _require_sha256(row["removed_positions_sha256"], "public removed positions")
        positions_equal = control_sha == removed_sha
        if row["final_positions_equal"] is not positions_equal:
            raise ValueError("public audit position verdict does not match its hashes")
        control_metrics = _visible_metrics(row["control_visible_metrics"], "public control metrics")
        removed_metrics = _visible_metrics(row["removed_visible_metrics"], "public removed metrics")
        metrics_equal = _metrics_equal(control_metrics, removed_metrics)
        if row["visible_metrics_equal"] is not metrics_equal:
            raise ValueError("public audit metric verdict does not match its metrics")
        preserved = positions_equal and metrics_equal
        if row["final_preserved"] is not preserved:
            raise ValueError("public audit final verdict is inconsistent")
        if not preserved:
            rejected.append(size)
    if observed_sizes != {int(size) for size in expected_map}:
        raise ValueError("public audit mapped sizes do not match the confirmed map")
    rejected.sort()

    summary = _require_exact_keys(
        payload["summary"],
        {"mapped_cases", "preserved_cases", "rejected_sizes"},
        "public audit summary",
    )
    if _require_int(summary["mapped_cases"], "public mapped case count") != len(cases):
        raise ValueError("public mapped case count is inconsistent")
    if _require_int(summary["preserved_cases"], "public preserved case count") != len(cases) - len(rejected):
        raise ValueError("public preserved case count is inconsistent")
    if _size_list(summary["rejected_sizes"], "public rejected sizes") != rejected:
        raise ValueError("public rejected-size summary is inconsistent")
    return {
        "harness_sha256": harness_sha,
        "mapped_cases": len(cases),
        "mapped_sizes": sorted(observed_sizes),
        "preserved_cases": len(cases) - len(rejected),
        "rejected_sizes": rejected,
    }


def finalize_policy(slot_map_path: Path, public_audit_path: Path) -> dict:
    """Validate both artifacts and return a deterministic final policy object."""

    slot_map_path = Path(slot_map_path)
    public_audit_path = Path(public_audit_path)
    slot_raw, slot_payload = _load_object(slot_map_path, "derivation artifact")
    audit_raw, audit_payload = _load_object(public_audit_path, "public audit")
    slot_sha = _sha256_bytes(slot_raw)
    audit_sha = _sha256_bytes(audit_raw)
    derivation = _validate_derivation(slot_payload)
    public = _validate_public_audit(audit_payload, slot_sha, derivation)

    public_rejected = public["rejected_sizes"]
    final_replacement = {
        size: slot
        for size, slot in derivation["replacement"].items()
        if int(size) not in public_rejected
    }
    # This assertion is deliberately redundant: it documents the only allowed
    # transformation and protects future edits to the composition below.
    if any(
        size not in derivation["replacement"]
        or slot != derivation["replacement"][size]
        for size, slot in final_replacement.items()
    ):
        raise AssertionError("finalizer attempted to add or retune a mapping")
    final_abstain = [size for size in HEAVY_SIZES if str(size) not in final_replacement]

    result = {
        "schema_version": 2,
        "mode": DERIVATION_MODE,
        "evidence_status": EVIDENCE_STATUS,
        "provenance": {
            "derivation_harness": copy.deepcopy(slot_payload["provenance"]["derivation_harness"]),
            "public_audit_harness": {
                "path": "scripts/audit_public_slot_fidelity.py",
                "sha256": public["harness_sha256"],
            },
            "finalizer": {
                "path": "scripts/finalize_learned_policy.py",
                "sha256": _sha256_file(FINALIZER_PATH),
            },
            "input_artifacts": {
                "derivation": {
                    "path": _portable_path(slot_map_path),
                    "sha256": slot_sha,
                },
                "public_audit": {
                    "path": _portable_path(public_audit_path),
                    "sha256": audit_sha,
                },
            },
        },
        "contract": {
            **copy.deepcopy(slot_payload["contract"]),
            "public_validation": {
                "uses_public_cases": True,
                "uses_public_golden_costs": False,
                "policy": PUBLIC_POLICY,
                "required_cases_per_mapped_size": 1,
            },
        },
        "solver_binding": copy.deepcopy(slot_payload["solver_binding"]),
        "development_artifacts": copy.deepcopy(slot_payload["development_artifacts"]),
        "confirmation_artifacts": copy.deepcopy(slot_payload["confirmation_artifacts"]),
        "total_selected_counts": copy.deepcopy(slot_payload["total_selected_counts"]),
        "tie_preference": copy.deepcopy(slot_payload["tie_preference"]),
        "selected_counts_by_size": copy.deepcopy(slot_payload["selected_counts_by_size"]),
        "nonpreserved_counts_by_size": copy.deepcopy(slot_payload["nonpreserved_counts_by_size"]),
        "derivation_case_counts_by_size": copy.deepcopy(slot_payload["derivation_case_counts_by_size"]),
        "derivation_unique_source_counts_by_size": copy.deepcopy(
            slot_payload["derivation_unique_source_counts_by_size"]
        ),
        "derived_replacement_wf_by_size": copy.deepcopy(derivation["derived"]),
        "development_abstain_sizes": copy.deepcopy(derivation["development_abstain"]),
        "confirmation_rejections": copy.deepcopy(slot_payload["confirmation_rejections"]),
        "confirmation_rejected_sizes": copy.deepcopy(derivation["confirmation_rejected"]),
        "public_audit_summary": {
            "mapped_cases": public["mapped_cases"],
            "mapped_sizes": public["mapped_sizes"],
            "preserved_cases": public["preserved_cases"],
        },
        "public_rejected_sizes": copy.deepcopy(public_rejected),
        "replacement_wf_by_size": final_replacement,
        "abstain_sizes": final_abstain,
    }
    # A final internal set-algebra check keeps abstention semantics explicit.
    if set(result["public_rejected_sizes"]) & set(derivation["abstain"]):
        raise AssertionError("public audit rejected a previously abstained size")
    if result["abstain_sizes"] != sorted(
        set(derivation["abstain"]) | set(result["public_rejected_sizes"])
    ):
        raise AssertionError("final abstentions are not rejection-only")
    return result


def _write_json_atomic(path: Path, payload: dict, *, overwrite: bool) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o644)
        if overwrite:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise FileExistsError(
                    f"output already exists (pass --overwrite to replace it): {path}"
                ) from error
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot-map", type=Path, required=True)
    parser.add_argument("--public-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing output atomically"
    )
    args = parser.parse_args()
    try:
        if args.output.exists() and not args.overwrite:
            raise FileExistsError(
                f"output already exists (pass --overwrite to replace it): {args.output}"
            )
        result = finalize_policy(args.slot_map, args.public_audit)
        _write_json_atomic(args.output, result, overwrite=args.overwrite)
    except Exception as error:  # A CLI provenance gate must fail with no output.
        print(f"Learned-policy finalization: FAIL\n  {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps({
        "evidence_status": result["evidence_status"],
        "public_rejected_sizes": result["public_rejected_sizes"],
        "replacement_wf_by_size": result["replacement_wf_by_size"],
        "abstain_sizes": result["abstain_sizes"],
    }, indent=2))


if __name__ == "__main__":
    main()
