import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compare_fold_results", ROOT / "scripts" / "compare_fold_results.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


MANIFEST_SHA = "1" * 64
INVENTORY_SHA = "2" * 64
EVALUATOR_SHA = "3" * 64
HARNESS_SHA = "4" * 64
OFFICIAL_COMMIT = "5" * 40


def _case(sample_index, block_count, cost, *, source="same.th", offset=0):
    return {
        "case_id": f"{source}#{offset}",
        "source_file": source,
        "file_offset": offset,
        "sample_index": sample_index,
        "block_count": block_count,
        "input_sha256": hashlib.sha256(f"input-{sample_index}".encode()).hexdigest(),
        "scoring_label_sha256": hashlib.sha256(
            f"label-{sample_index}".encode()
        ).hexdigest(),
        "cost": cost,
        "is_feasible": True,
    }


def _result_data(fold, rows, *, manifest_sha=MANIFEST_SHA):
    return {
        "config": {
            "fold": fold,
            "manifest": {
                "sha256": manifest_sha,
                "schema_version": 2,
                "fold": fold,
                "fold_metadata": {
                    "fold": fold,
                    "case_count": len(rows),
                    "source_file_count": len({row["source_file"] for row in rows}),
                },
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
            },
            "oracle_baseline_selector": False,
            "runtime_factor_mode": "neutral_rf_1",
            "require_golden_mib_clean": None,
        },
        "provenance": {
            "evaluation_harness_sha256": HARNESS_SHA,
            "evaluator_sha256": EVALUATOR_SHA,
            "official_floorset_git": {
                "commit": OFFICIAL_COMMIT,
                "tracked_dirty": False,
                "has_untracked": True,
            },
        },
        "cases": rows,
    }


def _write_data(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_result(path, fold, rows, *, manifest_sha=MANIFEST_SHA):
    data = _result_data(fold, rows, manifest_sha=manifest_sha)
    _write_data(path, data)
    return data


def _set_nested(mapping, path, value):
    for component in path[:-1]:
        mapping = mapping[component]
    mapping[path[-1]] = value


def test_paired_comparison_reports_fold_delta_contract_and_artifact_hashes(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    rows = [
        _case(1, 100, 2.0, offset=0),
        _case(2, 112, 4.0, offset=1),
    ]
    improved = [dict(row, cost=row["cost"] - 0.25) for row in rows]
    _write_result(baseline, 3, rows)
    _write_result(candidate, 3, improved)

    result = MODULE.compare_result_pairs(
        [baseline],
        [candidate],
        source_by_sample={
            "case:same.th#0": "same.th",
            "case:same.th#1": "same.th",
        },
        bootstrap_samples=100,
        seed=7,
    )

    assert result["schema_version"] == 2
    assert abs(result["delta_candidate_minus_baseline"] + 0.25) < 1e-12
    assert result["wins"] == 2 and result["losses"] == 0
    assert result["source_clusters"] == 1
    assert result["bootstrap"]["delta_ci95"] == [-0.25, -0.25]
    assert result["pseudo_test_one_per_block_count"]["delta_ci95"] == [
        -0.25,
        -0.25,
    ]
    assert result["tail_risk"]["worst_case_score_contribution"] < 0.0
    assert result["folds"][0]["fold"] == 3
    assert result["evaluation_contract"]["manifest_sha256"] == MANIFEST_SHA
    assert result["evaluation_contract"]["evaluator_sha256"] == EVALUATOR_SHA
    assert result["input_result_artifacts"]["baseline"][0]["sha256"] == (
        hashlib.sha256(baseline.read_bytes()).hexdigest()
    )
    assert result["input_result_artifacts"]["candidate"][0]["sha256"] == (
        hashlib.sha256(candidate.read_bytes()).hexdigest()
    )


def test_comparison_rejects_mismatched_cases(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_result(baseline, 0, [_case(1, 100, 2.0, source="one.th")])
    _write_result(candidate, 0, [_case(2, 100, 1.0, source="two.th")])

    with pytest.raises(ValueError, match="sample mismatch"):
        MODULE.compare_result_pairs([baseline], [candidate], bootstrap_samples=100)


@pytest.mark.parametrize("field", ["input_sha256", "scoring_label_sha256"])
def test_comparison_rejects_mismatched_case_digests(tmp_path, field):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    row = _case(1, 100, 2.0)
    changed = dict(row, cost=1.0)
    changed[field] = "f" * 64
    _write_result(baseline, 0, [row])
    _write_result(candidate, 0, [changed])

    with pytest.raises(ValueError, match="case contract mismatch"):
        MODULE.compare_result_pairs([baseline], [candidate], bootstrap_samples=100)


@pytest.mark.parametrize(
    "field", ["case_id", "input_sha256", "scoring_label_sha256"]
)
def test_comparison_rejects_missing_case_contract_fields(tmp_path, field):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    row = _case(1, 100, 2.0)
    incomplete = dict(row, cost=1.0)
    del incomplete[field]
    _write_result(baseline, 0, [row])
    _write_result(candidate, 0, [incomplete])

    with pytest.raises(ValueError, match="missing required field"):
        MODULE.compare_result_pairs([baseline], [candidate], bootstrap_samples=100)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("provenance", "evaluator_sha256"), "a" * 64),
        (("provenance", "evaluation_harness_sha256"), "b" * 64),
        (("config", "runtime_factor_mode"), "runtime_adjusted"),
        (("config", "oracle_baseline_selector"), True),
        (("config", "manifest", "sha256"), "c" * 64),
        (("config", "manifest", "generation", "seed"), 123),
    ],
)
def test_comparison_rejects_evaluation_contract_mismatch(tmp_path, path, value):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    row = _case(1, 100, 2.0)
    _write_result(baseline, 0, [row])
    candidate_data = _result_data(0, [dict(row, cost=1.0)])
    _set_nested(candidate_data, path, value)
    _write_data(candidate, candidate_data)

    with pytest.raises(ValueError, match="evaluation contract mismatch"):
        MODULE.compare_result_pairs([baseline], [candidate], bootstrap_samples=100)


def test_comparison_rejects_dataset_provenance_mismatch(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    row = _case(1, 100, 2.0)
    _write_result(baseline, 0, [row])
    candidate_data = _result_data(0, [dict(row, cost=1.0)])
    new_inventory = "d" * 64
    candidate_data["config"]["manifest"]["dataset"][
        "source_inventory_sha256"
    ] = new_inventory
    candidate_data["config"]["manifest"][
        "resolved_inventory_sha256"
    ] = new_inventory
    _write_data(candidate, candidate_data)

    with pytest.raises(ValueError, match="evaluation contract mismatch"):
        MODULE.compare_result_pairs([baseline], [candidate], bootstrap_samples=100)


def test_comparison_rejects_official_commit_mismatch(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    row = _case(1, 100, 2.0)
    _write_result(baseline, 0, [row])
    candidate_data = _result_data(0, [dict(row, cost=1.0)])
    other_commit = "a" * 40
    candidate_data["config"]["manifest"]["dataset"][
        "official_floorset_commit"
    ] = other_commit
    candidate_data["provenance"]["official_floorset_git"][
        "commit"
    ] = other_commit
    _write_data(candidate, candidate_data)

    with pytest.raises(ValueError, match="evaluation contract mismatch"):
        MODULE.compare_result_pairs([baseline], [candidate], bootstrap_samples=100)


def test_comparison_rejects_missing_harness_and_dirty_official_checkout(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    row = _case(1, 100, 2.0)
    baseline_data = _result_data(0, [row])
    del baseline_data["provenance"]["evaluation_harness_sha256"]
    _write_data(baseline, baseline_data)
    _write_result(candidate, 0, [dict(row, cost=1.0)])
    with pytest.raises(ValueError, match="missing required field"):
        MODULE.compare_result_pairs([baseline], [candidate], bootstrap_samples=100)

    baseline_data = _result_data(0, [row])
    baseline_data["provenance"]["official_floorset_git"]["tracked_dirty"] = True
    _write_data(baseline, baseline_data)
    with pytest.raises(ValueError, match="tracked changes"):
        MODULE.compare_result_pairs([baseline], [candidate], bootstrap_samples=100)


def test_comparison_rejects_cross_fold_harness_mismatch(tmp_path):
    paths = [tmp_path / name for name in ("b0.json", "c0.json", "b1.json", "c1.json")]
    b0, c0, b1, c1 = paths
    row0 = _case(1, 100, 2.0, source="zero.th")
    row1 = _case(2, 101, 2.0, source="one.th")
    _write_result(b0, 0, [row0])
    _write_result(c0, 0, [dict(row0, cost=1.9)])
    b1_data = _result_data(1, [row1])
    c1_data = _result_data(1, [dict(row1, cost=1.9)])
    for data in (b1_data, c1_data):
        data["provenance"]["evaluation_harness_sha256"] = "e" * 64
    _write_data(b1, b1_data)
    _write_data(c1, c1_data)

    with pytest.raises(ValueError, match="cross-fold evaluation contract mismatch"):
        MODULE.compare_result_pairs(
            [b0, b1], [c0, c1], bootstrap_samples=100
        )


def test_comparison_hashes_every_result_in_a_multi_fold_comparison(tmp_path):
    baselines = []
    candidates = []
    for fold in (0, 1):
        row = _case(fold + 1, 100 + fold, 2.0, source=f"fold-{fold}.th")
        baseline = tmp_path / f"baseline-{fold}.json"
        candidate = tmp_path / f"candidate-{fold}.json"
        _write_result(baseline, fold, [row])
        _write_result(candidate, fold, [dict(row, cost=1.9)])
        baselines.append(baseline)
        candidates.append(candidate)

    result = MODULE.compare_result_pairs(
        baselines, candidates, bootstrap_samples=100
    )
    assert [entry["sha256"] for entry in result["input_result_artifacts"]["baseline"]] == [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in baselines
    ]
    assert [entry["sha256"] for entry in result["input_result_artifacts"]["candidate"]] == [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in candidates
    ]


def test_comparison_binds_and_verifies_source_manifest_artifact(tmp_path):
    manifest = tmp_path / "manifest.json"
    row = _case(1, 100, 2.0)
    manifest_data = {
        "manifests": [
            {
                "fold": 0,
                "cases": [
                    {
                        "case_id": row["case_id"],
                        "source_file": row["source_file"],
                    }
                ],
            }
        ]
    }
    _write_data(manifest, manifest_data)
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_result(baseline, 0, [row], manifest_sha=manifest_sha)
    _write_result(candidate, 0, [dict(row, cost=1.0)], manifest_sha=manifest_sha)

    result = MODULE.compare_result_pairs(
        [baseline],
        [candidate],
        source_by_sample=MODULE._source_map(manifest),
        source_manifest_path=manifest,
        bootstrap_samples=100,
    )
    assert result["source_manifest_artifact"]["sha256"] == manifest_sha

    manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="source manifest artifact hash"):
        MODULE.compare_result_pairs(
            [baseline],
            [candidate],
            source_by_sample={"case:same.th#0": "same.th"},
            source_manifest_path=manifest,
            bootstrap_samples=100,
        )
