import importlib.util
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_candidate_selector", ROOT / "scripts" / "audit_candidate_selector.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


MANIFEST_SHA = "1" * 64
INVENTORY_SHA = "2" * 64
EVALUATOR_SHA = "3" * 64
HARNESS_SHA = "4" * 64
SOLVER_SHA = "5" * 64
COMPONENT_SHA = "6" * 64
OFFICIAL_COMMIT = "7" * 40


def test_candidate_snapshot_is_exact_length_finite_and_detached():
    candidate = [[0.0, 0.0, 2.0, 3.0], [2.0, 0.0, 1.0, 1.0]]
    snapshot = MODULE._snapshot_positions(candidate, 2)
    candidate[0][0] = 99.0
    assert snapshot[0][0] == 0.0
    assert MODULE._snapshot_positions(candidate[:1], 2) is None
    assert MODULE._snapshot_positions([[0.0, 0.0, math.nan, 1.0]], 1) is None
    assert MODULE._snapshot_positions([[0.0, 0.0, 0.0, 1.0]], 1) is None


def test_summary_weights_regret_and_reports_tail_risk():
    rows = [
        {
            "block_count": 100,
            "candidate_count": 3,
            "proxy_cost": 2.0,
            "oracle_cost": 1.5,
            "incumbent_cost": 1.8,
            "proxy_regret": 0.5,
            "proxy_false_accept": True,
            "proxy_missed_win": True,
            "proxy_index": 2,
            "oracle_index": 1,
            "pareto_dominant_indices": [1],
        },
        {
            "block_count": 120,
            "candidate_count": 2,
            "proxy_cost": 1.6,
            "oracle_cost": 1.6,
            "incumbent_cost": 1.7,
            "proxy_regret": 0.0,
            "proxy_false_accept": False,
            "proxy_missed_win": False,
            "proxy_index": 1,
            "oracle_index": 1,
            "pareto_dominant_indices": [],
        },
    ]
    summary = MODULE.summarize(rows)
    assert summary["cases"] == 2
    assert summary["candidate_decisions"] == 5
    assert summary["proxy_false_accepts"] == 1
    assert summary["proxy_matches_oracle"] == 1
    assert 0.0 < summary["weighted_proxy_regret"] < 0.5
    assert summary["worst_proxy_regret"] == 0.5


def _audit_case(source, offset, sample_index, block_count=100):
    return {
        "case_id": f"{source}#{offset}",
        "source_file": source,
        "file_offset": offset,
        "sample_index": sample_index,
        "block_count": block_count,
        "input_sha256": hashlib.sha256(f"input-{sample_index}".encode()).hexdigest(),
        "optimizer_target_sha256": hashlib.sha256(
            f"target-{sample_index}".encode()
        ).hexdigest(),
        "scoring_label_sha256": hashlib.sha256(
            f"label-{sample_index}".encode()
        ).hexdigest(),
        "candidate_count": 2,
        "proxy_cost": 1.5,
        "oracle_cost": 1.4,
        "incumbent_cost": 1.6,
        "proxy_regret": 0.1,
        "proxy_false_accept": False,
        "proxy_missed_win": True,
        "proxy_index": 1,
        "oracle_index": 0,
        "pareto_dominant_indices": [],
    }


def _artifact(fold, rows):
    return {
        "schema_version": 2,
        "config": {
            "fold": fold,
            "mode": MODULE.RAW_MODE,
            "runtime_factor_mode": MODULE.RUNTIME_FACTOR_MODE,
            "oracle_policy": MODULE.ORACLE_POLICY,
            "manifest": {
                "sha256": MANIFEST_SHA,
                "schema_version": 3,
                "fold": fold,
                "fold_metadata": {"fold": fold},
                "generation": {
                    "min_blocks": 100,
                    "max_blocks": 120,
                    "num_folds": 5,
                    "per_size": 5,
                    "seed": 20260710,
                },
                "dataset": {
                    "name": "FloorSet-Lite",
                    "official_floorset_commit": OFFICIAL_COMMIT,
                    "loader": "lite_dataset.FloorplanDatasetLite",
                    "layouts_per_file": 112,
                    "source_file_count": 9000,
                    "source_inventory_sha256": INVENTORY_SHA,
                },
                "resolved_inventory_sha256": INVENTORY_SHA,
                "resolved_official_floorset_commit": OFFICIAL_COMMIT,
            },
        },
        "provenance": {
            "evaluation_harness_sha256": HARNESS_SHA,
            "solver_source_sha256": SOLVER_SHA,
            "solver_component_sha256": {"my_optimizer.py": COMPONENT_SHA},
            "evaluator_sha256": EVALUATOR_SHA,
            "official_floorset_git": {
                "commit": OFFICIAL_COMMIT,
                "tracked_dirty": False,
            },
        },
        "cases": rows,
    }


def _write(path, artifact):
    path.write_text(json.dumps(artifact), encoding="utf-8")


def _set_nested(mapping, path, value):
    for component in path[:-1]:
        mapping = mapping[component]
    mapping[path[-1]] = value


def test_combine_is_provenance_bound_and_records_input_hashes(tmp_path):
    first = tmp_path / "fold0.json"
    second = tmp_path / "fold1.json"
    _write(first, _artifact(0, [_audit_case("a.th", 1, 1)]))
    _write(second, _artifact(1, [_audit_case("b.th", 2, 2, 101)]))

    result = MODULE.combine_artifacts([first, second])

    assert result["schema_version"] == 2
    assert result["summary"]["cases"] == 2
    assert result["evaluation_contract"]["manifest_sha256"] == MANIFEST_SHA
    assert result["evaluation_contract"]["solver_source_sha256"] == SOLVER_SHA
    assert result["config"]["inputs"][0]["sha256"] == hashlib.sha256(
        first.read_bytes()
    ).hexdigest()
    assert result["config"]["inputs"][1]["size_bytes"] == second.stat().st_size


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("config", "manifest", "sha256"), "8" * 64),
        (("config", "manifest", "dataset", "name"), "another-dataset"),
        (("provenance", "evaluator_sha256"), "9" * 64),
        (("provenance", "evaluation_harness_sha256"), "a" * 64),
        (("provenance", "solver_source_sha256"), "b" * 64),
        (
            ("provenance", "solver_component_sha256", "my_optimizer.py"),
            "c" * 64,
        ),
    ],
)
def test_combine_rejects_evaluation_or_solver_contract_mismatch(
    tmp_path, path, value
):
    first = tmp_path / "fold0.json"
    second = tmp_path / "fold1.json"
    _write(first, _artifact(0, [_audit_case("a.th", 1, 1)]))
    changed = deepcopy(_artifact(1, [_audit_case("b.th", 2, 2)]))
    _set_nested(changed, path, value)
    _write(second, changed)

    with pytest.raises(ValueError, match="evaluation contract mismatch"):
        MODULE.combine_artifacts([first, second])


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("schema_version",), 3, "schema_version"),
        (("config", "runtime_factor_mode"), "timed", "runtime_factor_mode"),
        (("config", "oracle_policy"), "different", "oracle_policy"),
        (("provenance", "evaluation_harness_sha256"), None, "required field"),
    ],
)
def test_combine_rejects_missing_or_unsupported_contract(
    tmp_path, path, value, message
):
    artifact = _artifact(0, [_audit_case("a.th", 1, 1)])
    _set_nested(artifact, path, value)
    result_path = tmp_path / "audit.json"
    _write(result_path, artifact)

    with pytest.raises(ValueError, match=message):
        MODULE.combine_artifacts([result_path])
