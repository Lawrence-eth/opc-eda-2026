import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "train_order_model", ROOT / "scripts" / "train_order_model.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeDataset:
    def __init__(self, files, sample_factory, layouts_per_file=2):
        self.all_files = [str(path) for path in files]
        self.layouts_per_file = layouts_per_file
        self.sample_factory = sample_factory
        self.calls = []

    def __getitem__(self, index):
        self.calls.append(index)
        return self.sample_factory(index)


def _manifest(path, sources):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "split_unit": "source_file",
                "manifests": [
                    {
                        "fold": 0,
                        "cases": [
                            {"sample_index": i, "source_file": source}
                            for i, source in enumerate(sources)
                        ],
                    }
                ],
            }
        )
    )
    return path


def _sample(index=0):
    shift = float(index % 3)
    return {
        "input": (
            [4.0, 4.0],
            [(0, 1, 1.0)],
            [],
            [],
            [[1, 0, 0, 0, 1], [0, 1, 0, 0, 2]],
        ),
        "label": (
            None,
            [[2.0, 2.0, 0.0, shift], [2.0, 2.0, 2.0, 2.0 + shift]],
            None,
        ),
    }


def test_manifest_sources_are_all_excluded_before_deterministic_file_split(tmp_path):
    data_root = tmp_path / "data"
    files = [data_root / "floorset_lite" / "worker_0" / f"layouts_{i}.th" for i in range(14)]
    excluded_relative = {
        "floorset_lite/worker_0/layouts_2.th",
        "floorset_lite/worker_0/layouts_9.th",
    }
    manifest = _manifest(tmp_path / "folds.json", sorted(excluded_relative))
    holdout = MODULE.load_holdout_sources(manifest)
    dataset = FakeDataset(files, _sample)

    first = MODULE.partition_source_files(
        dataset,
        data_root=data_root,
        excluded_sources=holdout.paths,
        seed=77,
        validation_fraction=0.25,
        max_files=8,
    )
    second = MODULE.partition_source_files(
        dataset,
        data_root=data_root,
        excluded_sources=holdout.paths,
        seed=77,
        validation_fraction=0.25,
        max_files=8,
    )
    assert first == second
    training, validation, discovered = first
    train_paths = {row.relative_path for row in training}
    validation_paths = {row.relative_path for row in validation}
    assert discovered == len(files)
    assert len(training) == 6
    assert len(validation) == 2
    assert train_paths.isdisjoint(validation_paths)
    assert excluded_relative.isdisjoint(train_paths | validation_paths)


def test_multiple_holdout_manifests_exclude_the_source_union(tmp_path):
    first = _manifest(
        tmp_path / "clean.json", ["floorset_lite/worker_0/layouts_2.th"]
    )
    second = _manifest(
        tmp_path / "raw.json",
        [
            "floorset_lite/worker_0/layouts_2.th",
            "floorset_lite/worker_0/layouts_9.th",
        ],
    )
    holdout = MODULE.load_holdout_sources([first, second])
    assert holdout.paths == frozenset(
        {
            "floorset_lite/worker_0/layouts_2.th",
            "floorset_lite/worker_0/layouts_9.th",
        }
    )
    assert len(holdout.manifest_sha256s) == 2


def test_partition_fails_closed_when_a_manifest_source_is_not_in_dataset(tmp_path):
    data_root = tmp_path / "data"
    dataset = FakeDataset([data_root / "floorset_lite/worker_0/layouts_0.th"], _sample)
    with pytest.raises(ValueError, match="holdout sources were not found"):
        MODULE.partition_source_files(
            dataset,
            data_root=data_root,
            excluded_sources=frozenset({"floorset_lite/worker_9/layouts_9.th"}),
            seed=1,
            validation_fraction=0.2,
            max_files=None,
        )


def test_layout_targets_use_shared_feature_path_and_input_visible_constraints(monkeypatch):
    captured = {}

    def fake_features(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return [
            [1.0] + [0.0] * (len(MODULE.FEATURE_NAMES) - 1)
            for _ in range(2)
        ]

    monkeypatch.setattr(MODULE, "extract_order_features", fake_features)
    features, targets, n = MODULE.layout_examples(_sample(), message_steps=3)
    target_positions = captured["args"][6]
    assert n == 2
    assert len(features[0]) == len(MODULE.FEATURE_NAMES)
    assert captured["kwargs"] == {"message_steps": 3}
    assert target_positions == [[-1.0, -1.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]]
    assert targets == [[0.25, 0.25], [0.75, 0.75]]


def test_layout_stream_is_lazy_and_keeps_only_one_layout_batch(tmp_path):
    data_root = tmp_path / "data"
    files = [data_root / "floorset_lite/worker_0/layouts_0.th"]
    dataset = FakeDataset(files, _sample, layouts_per_file=3)
    source = MODULE.SourceFile(0, "floorset_lite/worker_0/layouts_0.th")
    stats = MODULE.ScanStats(source_files=1)
    stream = MODULE.stream_layouts(
        dataset,
        [source],
        min_blocks=2,
        max_blocks=2,
        max_layouts_per_file=None,
        message_steps=1,
        layout_seed=9,
        stats=stats,
    )
    features, targets = next(stream)
    assert dataset.calls == [0]
    assert features.shape == (2, len(MODULE.FEATURE_NAMES))
    assert targets.shape == (2, 2)
    assert stats.layouts_seen == 1


def test_standardized_ridge_recovers_two_linear_coordinate_models():
    raw = np.asarray(
        [
            [1.0, -2.0, 0.0],
            [1.0, -1.0, 2.0],
            [1.0, 0.0, -1.0],
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 3.0],
        ]
    )
    targets = np.column_stack(
        (0.3 + 0.2 * raw[:, 1] - 0.1 * raw[:, 2], 0.7 - 0.4 * raw[:, 1] + 0.3 * raw[:, 2])
    )
    moments = MODULE.RidgeMoments(3, 2)
    moments.update(raw[:2], targets[:2])
    moments.update(raw[2:], targets[2:])
    fitted = MODULE.fit_standardized_ridge(moments, ridge_lambda=0.0)
    predictions = ((raw - fitted["center"]) / fitted["scale"]) @ fitted["weights"]
    assert fitted["center"][0] == 0.0
    assert fitted["scale"][0] == 1.0
    assert np.allclose(predictions, targets, atol=1e-12)
    assert np.all(fitted["train_r2"] > 0.999999999)


def test_pairwise_inversion_counts_prediction_ties_as_half_errors():
    metrics = MODULE.PredictionMetrics(output_count=1)
    metrics.update(
        np.asarray([[0.0], [0.0], [1.0]]),
        np.asarray([[0.0], [1.0], [2.0]]),
    )
    summary = metrics.as_json()
    # Three target-ordered pairs: one prediction tie and two concordant pairs.
    assert summary["pairwise_inversion_fraction"] == pytest.approx([1.0 / 6.0])


def test_pairwise_inversion_excludes_target_ties_but_penalizes_prediction_ties():
    metrics = MODULE.PredictionMetrics(output_count=1)
    metrics.update(
        np.asarray([[0.0], [0.0], [0.0]]),
        np.asarray([[1.0], [1.0], [2.0]]),
    )
    summary = metrics.as_json()
    # The tied target pair is incomparable; each of the other two pairs is a
    # prediction tie and therefore contributes half an inversion.
    assert summary["pairwise_inversion_fraction"] == pytest.approx([0.5])


def test_end_to_end_artifact_is_reproducible_and_records_schema(tmp_path):
    data_root = tmp_path / "data"
    files = [data_root / "floorset_lite" / "worker_0" / f"layouts_{i}.th" for i in range(8)]
    for index, source in enumerate(files):
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"fake-source-{index}".encode())
    heldout = "floorset_lite/worker_0/layouts_3.th"
    manifest = _manifest(tmp_path / "folds.json", [heldout])
    kwargs = dict(
        data_root=data_root,
        holdout_manifest=manifest,
        seed=2026,
        validation_fraction=0.25,
        ridge_lambda=1e-3,
        min_blocks=2,
        max_blocks=2,
        max_files=None,
        max_layouts_per_file=2,
        message_steps=2,
    )
    first = MODULE.train_order_model(FakeDataset(files, _sample), **kwargs)
    second = MODULE.train_order_model(FakeDataset(files, _sample), **kwargs)
    assert first == second
    assert first["schema_version"] == 1
    assert first["feature_schema"] == {
        "version": MODULE.FEATURE_VERSION,
        "names": list(MODULE.FEATURE_NAMES),
        "message_steps": 2,
    }
    assert len(first["coefficients"]) == len(MODULE.FEATURE_NAMES)
    assert all(len(row) == 2 for row in first["coefficients"])
    assert len(first["normalization"]["center"]) == len(MODULE.FEATURE_NAMES)
    assert first["provenance"]["source_files_excluded"] == 1
    assert len(first["provenance"]["source_partition_content_sha256"]) == 64
    scans = first["stats"]["train_scan"], first["stats"]["validation_scan"]
    assert sum(scan["source_files_selected"] for scan in scans) == 7
    assert first["stats"]["validation_fit"]["examples"] > 0
    assert first["stats"]["validation_fit"]["layouts"] > 0
    assert len(first["stats"]["validation_fit"]["pairwise_inversion_fraction"]) == 2
    assert len(first["payload_sha256"]) == 64
