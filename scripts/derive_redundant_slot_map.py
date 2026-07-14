#!/usr/bin/env python3
"""Derive and confirm the final-output-preserving learned replacement map.

Exactly six development panels are required: clean/raw domains crossed with
folds 0, 1, and 2.  Two fold-3 panels are then used only to reject a derived
slot; confirmation can never retune a size to another width factor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


SLOTS = (0.8, 0.9, 1.0, 1.1, 1.2)
DERIVATION_FOLDS = (0, 1, 2)
CONFIRMATION_FOLD = 3
EXPECTED_MANIFESTS = {
    "mib_input_compatible": {
        "require_mib_input_compatible": True,
        "sha256": "48ecda41bb642caa67d2e617ff9e467816a0392d6a68a0a91c38cf2e5f847895",
    },
    "raw_hash": {
        "require_mib_input_compatible": False,
        "sha256": "9b4ff6a36e1945718411a83045f598228c2b301fdfa22340e33c297da9ac41ec",
    },
}
EXPECTED_GENERATION = {
    "min_blocks": 100,
    "max_blocks": 120,
    "num_folds": 5,
    "per_size": 5,
    "seed": 20260710,
}


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_sha256(value, context):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _require_commit(value, context):
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a full lowercase git commit")
    return value


def _visible_metrics(value, context):
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    feasible = value.get("feasible")
    soft = value.get("soft_violations")
    if type(feasible) is not bool:
        raise ValueError(f"{context}.feasible must be boolean")
    if isinstance(soft, bool) or not isinstance(soft, int) or soft < 0:
        raise ValueError(f"{context}.soft_violations must be nonnegative integer")
    numbers = {}
    for field in ("hpwl", "area"):
        try:
            number = float(value[field])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{context}.{field} must be numeric") from error
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"{context}.{field} must be finite and nonnegative")
        numbers[field] = number
    return {
        "feasible": feasible,
        "soft_violations": soft,
        **numbers,
    }


def _visible_metrics_equal(left, right):
    return (
        left["feasible"] == right["feasible"]
        and left["soft_violations"] == right["soft_violations"]
        and math.isclose(left["hpwl"], right["hpwl"], rel_tol=0.0, abs_tol=1e-9)
        and math.isclose(left["area"], right["area"], rel_tol=0.0, abs_tol=1e-9)
    )


def _domain_from_manifest(manifest, path):
    metadata = manifest.get("fold_metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{path} missing fold metadata")
    compatibility = metadata.get("require_mib_input_compatible")
    matches = [
        domain
        for domain, expected in EXPECTED_MANIFESTS.items()
        if compatibility is expected["require_mib_input_compatible"]
    ]
    if len(matches) != 1:
        raise ValueError(f"{path} has an unexpected domain policy")
    domain = matches[0]
    expected_sha = EXPECTED_MANIFESTS[domain]["sha256"]
    if manifest.get("sha256") != expected_sha:
        raise ValueError(f"{path} has an unexpected {domain} manifest hash")
    return domain


def _solver_binding(payload, path, expected_components):
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{path} missing provenance")
    git = provenance.get("solver_git")
    if not isinstance(git, dict):
        raise ValueError(f"{path} missing solver git provenance")
    if any(git.get(field) is not False for field in (
        "dirty", "tracked_dirty", "has_untracked"
    )):
        raise ValueError(f"{path} solver provenance is not clean")
    components = provenance.get("solver_components")
    if not isinstance(components, dict) or set(components) != set(expected_components):
        raise ValueError(f"{path} has an incomplete solver component binding")
    for name, digest in components.items():
        _require_sha256(digest, f"{path} solver component {name}")
    return {
        "commit": _require_commit(git.get("commit"), f"{path} solver commit"),
        "components": components,
        "audit_harness_sha256": _require_sha256(
            provenance.get("harness_sha256"), f"{path} audit harness"
        ),
    }


def _validate_panel(path, allowed_folds, expected_components):
    payload = json.loads(path.read_bytes())
    if payload.get("schema_version") != 2 or payload.get("mode") != (
        "legacy_v32_final_output_slot_removal_fidelity"
    ):
        raise ValueError(f"not a schema-2 final-fidelity audit: {path}")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{path} missing config")
    if config.get("learned_enabled") is not False:
        raise ValueError(f"learned solver was enabled in {path}")
    if config.get("golden_cost_computed") is not False:
        raise ValueError(f"golden cost was computed in {path}")
    if config.get("comparison_stage") != "complete_deployed_solver_output":
        raise ValueError(f"{path} does not compare complete solver output")
    if config.get("removal_method") != (
        "full solver rerun for every paid standard slot"
    ):
        raise ValueError(f"{path} did not rerun every slot")
    fold = config.get("fold")
    if isinstance(fold, bool) or not isinstance(fold, int) or fold not in allowed_folds:
        raise ValueError(f"{path} has unexpected fold {fold}")
    manifest = config.get("manifest")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 3:
        raise ValueError(f"{path} missing schema-3 manifest provenance")
    domain = _domain_from_manifest(manifest, path)
    metadata = manifest["fold_metadata"]
    if manifest.get("fold") != fold or metadata.get("fold") != fold:
        raise ValueError(f"{path} fold metadata mismatch")
    if metadata.get("case_count") != 105 or metadata.get("source_file_count") != 105:
        raise ValueError(f"{path} has unexpected panel size metadata")
    generation = manifest.get("generation")
    if not isinstance(generation, dict) or any(
        generation.get(field) != value
        for field, value in EXPECTED_GENERATION.items()
    ):
        raise ValueError(f"{path} has unexpected generation metadata")

    rows = payload.get("cases")
    if not isinstance(rows, list) or len(rows) != 105:
        raise ValueError(f"expected 105 cases in {path}")
    counts = Counter()
    seen = set()
    sources = set()
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path} case {position} is not an object")
        n = row.get("block_count")
        if isinstance(n, bool) or not isinstance(n, int) or not 100 <= n <= 120:
            raise ValueError(f"{path} case {position} has invalid block_count")
        counts[n] += 1
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError(f"{path} has missing/duplicate case identity")
        seen.add(case_id)
        if "#" not in case_id:
            raise ValueError(f"{path} case {position} has malformed case identity")
        sources.add(case_id.rsplit("#", 1)[0])
        baseline_sha256 = _require_sha256(
            row.get("baseline_final_positions_sha256"),
            f"{path} case {position} baseline positions",
        )
        baseline_metrics = _visible_metrics(
            row.get("baseline_visible_metrics"),
            f"{path} case {position} baseline metrics",
        )
        removals = row.get("removals")
        if not isinstance(removals, dict) or set(removals) != {
            str(slot) for slot in SLOTS
        }:
            raise ValueError(f"{path} case {position} has incomplete removals")
        for slot in SLOTS:
            removal = removals[str(slot)]
            if removal.get("execution") != "full_solver_rerun_with_slot_removed":
                raise ValueError(f"{path} case {position} did not rerun slot {slot}")
            removal_sha256 = _require_sha256(
                removal.get("final_positions_sha256"),
                f"{path} case {position} removal {slot}",
            )
            removal_metrics = _visible_metrics(
                removal.get("visible_metrics"),
                f"{path} case {position} removal {slot} metrics",
            )
            positions_equal = removal_sha256 == baseline_sha256
            metrics_equal = _visible_metrics_equal(
                removal_metrics, baseline_metrics
            )
            if removal.get("final_positions_equal") is not positions_equal:
                raise ValueError(f"{path} case {position} position verdict mismatch")
            if removal.get("visible_metrics_equal") is not metrics_equal:
                raise ValueError(f"{path} case {position} metric verdict mismatch")
            if removal.get("final_preserved") is not (
                positions_equal and metrics_equal
            ):
                raise ValueError(f"{path} case {position} final verdict mismatch")
    if any(counts[n] != 5 for n in range(100, 121)):
        raise ValueError(f"{path} does not have five cases per block count")

    descriptor = {
        "path": path.name,
        "sha256": _sha256(path),
        "fold": fold,
        "domain": domain,
        "manifest_sha256": manifest["sha256"],
        "case_count": len(rows),
    }
    return {
        "payload": payload,
        "rows": rows,
        "role": (domain, fold),
        "sources": sources,
        "descriptor": descriptor,
        "binding": _solver_binding(payload, path, expected_components),
    }


def derive_map(audit_paths, confirmation_paths, expected_components=None):
    audit_paths = [Path(path) for path in audit_paths]
    confirmation_paths = [Path(path) for path in confirmation_paths]
    if expected_components is None:
        try:
            from solver_components import LIVE_SOLVER_COMPONENTS
        except ImportError as error:
            raise ValueError(
                "authoritative LIVE_SOLVER_COMPONENTS registry is unavailable"
            ) from error
        expected_components = LIVE_SOLVER_COMPONENTS
    expected_components = tuple(expected_components)
    if not expected_components or len(expected_components) != len(set(expected_components)):
        raise ValueError("expected component registry is empty or duplicated")
    if len(audit_paths) != 6:
        raise ValueError("exactly six development audit panels are required")
    if len(confirmation_paths) != 2:
        raise ValueError("exactly two fold-3 confirmation panels are required")
    development = [
        _validate_panel(path, set(DERIVATION_FOLDS), expected_components)
        for path in audit_paths
    ]
    confirmation = [
        _validate_panel(path, {CONFIRMATION_FOLD}, expected_components)
        for path in confirmation_paths
    ]
    expected_development = {
        (domain, fold)
        for domain in EXPECTED_MANIFESTS
        for fold in DERIVATION_FOLDS
    }
    expected_confirmation = {
        (domain, CONFIRMATION_FOLD) for domain in EXPECTED_MANIFESTS
    }
    if {panel["role"] for panel in development} != expected_development or len({
        panel["role"] for panel in development
    }) != len(development):
        raise ValueError("development panels must be the unique clean/raw x fold0-2 matrix")
    if {panel["role"] for panel in confirmation} != expected_confirmation or len({
        panel["role"] for panel in confirmation
    }) != len(confirmation):
        raise ValueError("confirmation panels must be unique clean/raw fold3")
    binding = development[0]["binding"]
    if any(panel["binding"] != binding for panel in development + confirmation):
        raise ValueError("audit panels have different solver/source bindings")
    for domain in EXPECTED_MANIFESTS:
        domain_panels = [
            panel for panel in development if panel["role"][0] == domain
        ]
        for index, left in enumerate(domain_panels):
            for right in domain_panels[index + 1:]:
                if left["sources"] & right["sources"]:
                    raise ValueError(
                        f"{domain} development folds are not source-disjoint"
                    )
    development_sources = set().union(
        *(panel["sources"] for panel in development)
    )
    if any(panel["sources"] & development_sources for panel in confirmation):
        raise ValueError("fold3 confirmation sources overlap development")

    selected_by_size = defaultdict(Counter)
    nonpreserved_by_size = defaultdict(Counter)
    support_by_size = Counter()
    sources_by_size = defaultdict(set)
    total_selected = Counter()
    for panel in development:
        for row in panel["rows"]:
            n = row["block_count"]
            support_by_size[n] += 1
            sources_by_size[n].add(row["case_id"].rsplit("#", 1)[0])
            selected = row.get("selected_standard_wf")
            if selected is not None:
                if isinstance(selected, bool) or not isinstance(selected, (int, float)):
                    raise ValueError(
                        "development audit has non-numeric selected slot"
                    )
                selected = float(selected)
                if not math.isfinite(selected) or selected not in SLOTS:
                    raise ValueError("development audit contains an unknown selected slot")
                selected_by_size[n][selected] += 1
                total_selected[selected] += 1
            for slot in SLOTS:
                if row["removals"][str(slot)]["final_preserved"] is not True:
                    nonpreserved_by_size[n][slot] += 1
    if any(support_by_size[n] != 30 for n in range(100, 121)):
        raise ValueError("every block count must have exactly 30 development cases")

    preference = sorted(SLOTS, key=lambda slot: (total_selected[slot], slot))
    derived = {}
    development_abstentions = []
    for n in range(100, 121):
        chosen = next(
            (slot for slot in preference if nonpreserved_by_size[n][slot] == 0),
            None,
        )
        if chosen is None:
            development_abstentions.append(n)
        else:
            derived[str(n)] = chosen

    confirmation_rejections = []
    rejected_sizes = []
    for n_text, slot in derived.items():
        n = int(n_text)
        failures = []
        for panel in confirmation:
            for row in panel["rows"]:
                if row["block_count"] != n:
                    continue
                removal = row["removals"][str(slot)]
                if removal["final_preserved"] is not True:
                    failures.append({
                        "domain": panel["role"][0],
                        "case_id": row["case_id"],
                        "final_positions_equal": removal["final_positions_equal"],
                        "visible_metrics_equal": removal["visible_metrics_equal"],
                    })
        if failures:
            rejected_sizes.append(n)
            confirmation_rejections.append({
                "block_count": n,
                "derived_width_factor": slot,
                "failures": failures,
            })
    replacement = {
        n: slot for n, slot in derived.items() if int(n) not in rejected_sizes
    }
    abstentions = sorted(development_abstentions + rejected_sizes)

    return {
        "schema_version": 2,
        "mode": "legacy_v32_final_output_preserving_slot_calibration",
        "contract": {
            "derivation_folds": list(DERIVATION_FOLDS),
            "derivation_domains": list(EXPECTED_MANIFESTS),
            "confirmation_fold": CONFIRMATION_FOLD,
            "confirmation_policy": "rejection_only_no_slot_retuning",
            "uses_learned_outputs": False,
            "uses_golden_costs": False,
            "uses_public_cases": False,
            "invariant": (
                "observed removal preserves packed-byte final positions and "
                "inference-visible metrics on every derivation case"
            ),
            "comparison_stage": "complete_deployed_solver_output",
            "removal_method": "full solver rerun for every paid standard slot",
            "tie_rule": "ascending(total_selected_count, numeric_width_factor)",
            "required_development_support_per_size": 30,
        },
        "solver_binding": binding,
        "development_artifacts": [panel["descriptor"] for panel in development],
        "confirmation_artifacts": [panel["descriptor"] for panel in confirmation],
        "total_selected_counts": {str(slot): total_selected[slot] for slot in SLOTS},
        "tie_preference": preference,
        "selected_counts_by_size": {
            str(n): {str(slot): selected_by_size[n][slot] for slot in SLOTS}
            for n in range(100, 121)
        },
        "nonpreserved_counts_by_size": {
            str(n): {str(slot): nonpreserved_by_size[n][slot] for slot in SLOTS}
            for n in range(100, 121)
        },
        "derivation_case_counts_by_size": {
            str(n): support_by_size[n] for n in range(100, 121)
        },
        "derivation_unique_source_counts_by_size": {
            str(n): len(sources_by_size[n]) for n in range(100, 121)
        },
        "derived_replacement_wf_by_size": derived,
        "development_abstain_sizes": development_abstentions,
        "confirmation_rejections": confirmation_rejections,
        "confirmation_rejected_sizes": rejected_sizes,
        "replacement_wf_by_size": replacement,
        "abstain_sizes": abstentions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audits", type=Path, nargs=6)
    parser.add_argument(
        "--confirmation-audits", type=Path, nargs=2, required=True
    )
    parser.add_argument(
        "--expected-component",
        action="append",
        help="explicit test/research component registry override",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = derive_map(
        args.audits,
        args.confirmation_audits,
        expected_components=args.expected_component,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "replacement_wf_by_size": result["replacement_wf_by_size"],
        "development_abstain_sizes": result["development_abstain_sizes"],
        "confirmation_rejected_sizes": result["confirmation_rejected_sizes"],
        "abstain_sizes": result["abstain_sizes"],
    }, indent=2))


if __name__ == "__main__":
    main()
