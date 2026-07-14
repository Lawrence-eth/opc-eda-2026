import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DERIVE = _load(
    "derive_redundant_slot_map",
    ROOT / "scripts" / "derive_redundant_slot_map.py",
)
PUBLIC = _load(
    "audit_public_slot_fidelity",
    ROOT / "scripts" / "audit_public_slot_fidelity.py",
)
COMPONENTS = ("my_optimizer.py", "dissect.py")
COMMIT = "a" * 40


def _digest(label):
    return hashlib.sha256(label.encode()).hexdigest()


def _metrics(hpwl=100.0):
    return {
        "feasible": True,
        "hpwl": hpwl,
        "area": 100.0,
        "soft_violations": 1,
    }


def _panel(domain, fold):
    compatible = domain == "mib_input_compatible"
    rows = []
    phase = "dev" if fold < 3 else "confirm"
    for n in range(100, 121):
        for index in range(5):
            # Clean/raw panels intentionally share source groups; folds do not.
            source = f"{phase}_fold{fold}_n{n}_source{index}.th"
            baseline_sha = _digest(f"{domain}-{fold}-{n}-{index}-baseline")
            removals = {}
            for slot in DERIVE.SLOTS:
                removals[str(slot)] = {
                    "execution": "full_solver_rerun_with_slot_removed",
                    "final_positions_sha256": baseline_sha,
                    "final_positions_equal": True,
                    "visible_metrics": _metrics(),
                    "visible_metrics_equal": True,
                    "final_preserved": True,
                }
            rows.append({
                "case_id": f"{source}#0",
                "block_count": n,
                "selected_standard_wf": None,
                "baseline_final_positions_sha256": baseline_sha,
                "baseline_visible_metrics": _metrics(),
                "removals": removals,
            })
    manifest_sha = DERIVE.EXPECTED_MANIFESTS[domain]["sha256"]
    return {
        "schema_version": 2,
        "mode": "legacy_v32_final_output_slot_removal_fidelity",
        "config": {
            "fold": fold,
            "learned_enabled": False,
            "golden_cost_computed": False,
            "comparison_stage": "complete_deployed_solver_output",
            "removal_method": "full solver rerun for every paid standard slot",
            "manifest": {
                "schema_version": 3,
                "sha256": manifest_sha,
                "fold": fold,
                "fold_metadata": {
                    "fold": fold,
                    "require_mib_input_compatible": compatible,
                    "case_count": 105,
                    "source_file_count": 105,
                },
                "generation": dict(DERIVE.EXPECTED_GENERATION),
            },
        },
        "provenance": {
            "harness_sha256": _digest("harness"),
            "solver_components": {
                name: _digest(name) for name in COMPONENTS
            },
            "solver_git": {
                "commit": COMMIT,
                "dirty": False,
                "tracked_dirty": False,
                "has_untracked": False,
            },
        },
        "cases": rows,
        "preserved_counts_by_size": {},
    }


def _matrix(tmp_path):
    development = []
    confirmation = []
    for domain in DERIVE.EXPECTED_MANIFESTS:
        for fold in DERIVE.DERIVATION_FOLDS:
            path = tmp_path / f"{domain}_{fold}.json"
            path.write_text(json.dumps(_panel(domain, fold)))
            development.append(path)
        path = tmp_path / f"{domain}_3.json"
        path.write_text(json.dumps(_panel(domain, 3)))
        confirmation.append(path)
    return development, confirmation


def _rewrite(path, mutate):
    payload = json.loads(path.read_text())
    mutate(payload)
    path.write_text(json.dumps(payload))


def test_derivation_accepts_only_complete_clean_bound_matrix(tmp_path):
    development, confirmation = _matrix(tmp_path)
    result = DERIVE.derive_map(
        development, confirmation, expected_components=COMPONENTS
    )
    assert result["schema_version"] == 2
    assert result["solver_binding"]["commit"] == COMMIT
    assert set(result["derivation_case_counts_by_size"].values()) == {30}
    assert set(result["derivation_unique_source_counts_by_size"].values()) == {15}
    assert result["development_abstain_sizes"] == []
    assert result["confirmation_rejected_sizes"] == []


def test_derivation_rejects_missing_or_duplicate_matrix_roles(tmp_path):
    development, confirmation = _matrix(tmp_path)
    with pytest.raises(ValueError, match="exactly six"):
        DERIVE.derive_map(
            development[:-1], confirmation, expected_components=COMPONENTS
        )
    with pytest.raises(ValueError, match="unique clean/raw"):
        DERIVE.derive_map(
            development[:-1] + [development[0]],
            confirmation,
            expected_components=COMPONENTS,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["provenance"]["solver_git"].__setitem__(
                "dirty", True
            ),
            "not clean",
        ),
        (
            lambda payload: payload["provenance"]["solver_components"].__setitem__(
                "my_optimizer.py", _digest("changed")
            ),
            "different solver/source",
        ),
        (
            lambda payload: payload["cases"][0].__setitem__(
                "block_count", 101
            ),
            "five cases per block count",
        ),
        (
            lambda payload: payload["cases"][0]["removals"]["1.0"].__setitem__(
                "final_positions_sha256", _digest("contradiction")
            ),
            "position verdict mismatch",
        ),
        (
            lambda payload: payload["cases"][0]["removals"]["1.0"][
                "visible_metrics"
            ].__setitem__("hpwl", 101.0),
            "metric verdict mismatch",
        ),
        (
            lambda payload: payload["cases"][0].__setitem__(
                "selected_standard_wf", True
            ),
            "non-numeric selected slot",
        ),
    ],
)
def test_derivation_recomputes_evidence_and_binding(
    tmp_path, mutation, message
):
    development, confirmation = _matrix(tmp_path)
    _rewrite(development[0], mutation)
    with pytest.raises(ValueError, match=message):
        DERIVE.derive_map(
            development, confirmation, expected_components=COMPONENTS
        )


def test_derivation_rejects_source_overlap(tmp_path):
    development, confirmation = _matrix(tmp_path)
    first_source = json.loads(development[0].read_text())["cases"][0][
        "case_id"
    ]
    _rewrite(
        development[1],
        lambda payload: payload["cases"][0].__setitem__(
            "case_id", first_source
        ),
    )
    with pytest.raises(ValueError, match="not source-disjoint"):
        DERIVE.derive_map(
            development, confirmation, expected_components=COMPONENTS
        )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"100": True},
        {"0100": 1.0},
        {"99": 1.0},
        {"100": 1.3},
        {
            "schema_version": 1,
            "mode": "legacy_v32_final_output_preserving_slot_calibration",
            "replacement_wf_by_size": {"100": 1.0},
        },
    ],
)
def test_public_slot_map_parser_fails_closed(payload):
    with pytest.raises(ValueError):
        PUBLIC.parse_slot_map(copy.deepcopy(payload))


def test_public_slot_map_parser_accepts_frozen_schema2_artifact():
    payload = {
        "schema_version": 2,
        "mode": "legacy_v32_final_output_preserving_slot_calibration",
        "replacement_wf_by_size": {"100": 1.0, "120": 0.9},
    }
    assert PUBLIC.parse_slot_map(payload) == {100: 1.0, 120: 0.9}
