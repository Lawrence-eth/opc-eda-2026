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


def _canonical_manifest(domain, fold):
    expected = DERIVE.EXPECTED_MANIFESTS[domain]
    path = ROOT / expected["path"]
    payload = json.loads(path.read_text())
    manifest = next(row for row in payload["manifests"] if row["fold"] == fold)
    return payload, manifest


def _panel(domain, fold):
    canonical, canonical_fold = _canonical_manifest(domain, fold)
    rows = []
    for index, case in enumerate(canonical_fold["cases"]):
        n = case["block_count"]
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
            "case_id": case["case_id"],
            "sample_index": case["sample_index"],
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
            "fold_manifest": DERIVE.EXPECTED_MANIFESTS[domain]["path"],
            "learned_enabled": False,
            "golden_cost_computed": False,
            "comparison_stage": "complete_deployed_solver_output",
            "removal_method": "full solver rerun for every paid standard slot",
            "manifest": {
                "schema_version": 3,
                "sha256": manifest_sha,
                "fold": fold,
                "fold_metadata": {
                    key: value
                    for key, value in canonical_fold.items()
                    if key != "cases"
                },
                "generation": canonical["generation"],
                "dataset": canonical["dataset"],
                "resolved_inventory_sha256": canonical["dataset"][
                    "source_inventory_sha256"
                ],
                "resolved_official_floorset_commit": canonical["dataset"][
                    "official_floorset_commit"
                ],
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
    assert result["provenance"]["derivation_harness"] == {
        "path": "scripts/derive_redundant_slot_map.py",
        "sha256": hashlib.sha256(
            (ROOT / "scripts" / "derive_redundant_slot_map.py").read_bytes()
        ).hexdigest(),
    }
    assert set(result["derivation_case_counts_by_size"].values()) == {30}
    assert set(result["derivation_unique_source_counts_by_size"].values()) == {
        15,
        16,
    }
    assert result["development_abstain_sizes"] == []
    assert result["confirmation_rejected_sizes"] == []


def test_derivation_accepts_frozen_clean_103_and_104_source_counts(tmp_path):
    development, confirmation = _matrix(tmp_path)
    result = DERIVE.derive_map(
        development, confirmation, expected_components=COMPONENTS
    )
    descriptors = {
        (row["domain"], row["fold"]): row["source_file_count"]
        for row in result["development_artifacts"]
    }
    assert descriptors[("mib_input_compatible", 0)] == 105
    assert descriptors[("mib_input_compatible", 1)] == 103
    assert descriptors[("mib_input_compatible", 2)] == 104
    assert descriptors[("raw_hash", 0)] == 105
    assert descriptors[("raw_hash", 1)] == 105
    assert descriptors[("raw_hash", 2)] == 105


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


def test_derivation_rejects_noncanonical_case_identity(tmp_path):
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
    with pytest.raises(ValueError, match="case identity/order"):
        DERIVE.derive_map(
            development, confirmation, expected_components=COMPONENTS
        )


def test_derivation_rejects_stale_frozen_source_count_metadata(tmp_path):
    development, confirmation = _matrix(tmp_path)
    _rewrite(
        development[1],
        lambda payload: payload["config"]["manifest"]["fold_metadata"].__setitem__(
            "source_file_count", 105
        ),
    )
    with pytest.raises(ValueError, match="frozen manifest"):
        DERIVE.derive_map(
            development, confirmation, expected_components=COMPONENTS
        )


def test_derivation_rejects_distinct_source_count_mismatch(tmp_path):
    development, confirmation = _matrix(tmp_path)

    def collapse_one_source(payload):
        first_source = payload["cases"][0]["case_id"].rsplit("#", 1)[0]
        payload["cases"][1]["case_id"] = f"{first_source}#999"

    _rewrite(development[0], collapse_one_source)
    with pytest.raises(ValueError, match="distinct case sources"):
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


def test_public_hard_target_extraction_ignores_stored_metrics():
    polygons = PUBLIC.torch.tensor(
        [
            [[2.0, 3.0], [6.0, 3.0], [6.0, 8.0], [2.0, 8.0]],
            [[-1.0, -1.0], [10.0, 4.0], [13.0, 4.0], [13.0, 6.0]],
        ]
    )
    metrics = object()

    assert PUBLIC._hard_target_positions_from_labels((polygons, metrics), 2) == [
        (2.0, 3.0, 4.0, 5.0),
        (10.0, 4.0, 3.0, 2.0),
    ]


@pytest.mark.parametrize(
    "polygons, message",
    [
        (PUBLIC.torch.tensor([[[-1.0, -1.0], [-1.0, -1.0]]]), "no valid"),
        (PUBLIC.torch.tensor([[[0.0, 0.0], [0.0, 1.0]]]), "nonpositive"),
        (
            PUBLIC.torch.tensor([[[0.0, 0.0], [float("nan"), 1.0]]]),
            "non-finite",
        ),
    ],
)
def test_public_hard_target_extraction_rejects_malformed_geometry(
    polygons, message
):
    with pytest.raises(ValueError, match=message):
        PUBLIC._hard_target_positions_from_labels((polygons, object()), 1)
