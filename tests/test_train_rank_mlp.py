import argparse
import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "train_rank_mlp", ROOT / "scripts" / "train_rank_mlp.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _seal_manifest(manifest):
    sealed = copy.deepcopy(manifest)
    sealed.pop("payload_sha256", None)
    sealed["payload_sha256"] = MODULE._canonical_sha256(sealed)
    return sealed


def _cache_manifest(tmp_path):
    shards = []
    sources = {"train": ["train/a.th"], "validation": ["validation/b.th"]}
    for partition in ("train", "validation"):
        path = tmp_path / f"{partition}.npz"
        path.write_bytes(f"fixture-{partition}".encode())
        shards.append(
            {
                "partition": partition,
                "path": path.name,
                "sha256": MODULE._sha256(path),
                "layouts": 1,
                "blocks": 2,
                "identities": [
                    {
                        "source_file": sources[partition][0],
                        "file_offset": 0,
                        "block_count": 2,
                    }
                ],
            }
        )
    return {
        "schema_version": MODULE.CACHE_SCHEMA_VERSION,
        "cache_type": "dual_parent_supervision_shards",
        "feature_schema": {
            "version": MODULE.FEATURE_VERSION,
            "names": list(MODULE.FEATURE_NAMES),
            "message_steps": 4,
        },
        "selection": {
            "split_unit": "source_file",
            "selected_source_counts": {
                partition: len(paths) for partition, paths in sources.items()
            },
            "selected_source_sha256": {
                partition: MODULE._canonical_sha256(paths)
                for partition, paths in sources.items()
            },
        },
        "holdouts": {"excluded_source_count": 741},
        "shards": shards,
    }


def test_manifest_proves_source_disjointness_and_binds_source_order(tmp_path):
    manifest = _seal_manifest(_cache_manifest(tmp_path))
    (tmp_path / "manifest.json").write_text(MODULE.json.dumps(manifest))
    assert MODULE.load_cache_manifest(tmp_path)["payload_sha256"] == manifest[
        "payload_sha256"
    ]

    overlapping = copy.deepcopy(manifest)
    overlapping["shards"][1]["identities"][0]["source_file"] = "train/a.th"
    overlapping["selection"]["selected_source_sha256"]["validation"] = (
        MODULE._canonical_sha256(["train/a.th"])
    )
    overlapping = _seal_manifest(overlapping)
    (tmp_path / "manifest.json").write_text(MODULE.json.dumps(overlapping))
    with pytest.raises(ValueError, match="sources overlap"):
        MODULE.load_cache_manifest(tmp_path)


def test_partition_masks_only_mib_channels_on_inconsistent_layouts(tmp_path):
    width = len(MODULE.FEATURE_NAMES)
    features = np.ones((4, width), dtype=np.float32)
    path = tmp_path / "train.npz"
    np.savez_compressed(
        path,
        features=features,
        x_rank=np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        y_rank=np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        layout_offsets=np.asarray([0, 2, 4], dtype=np.int64),
        mib_inconsistent=np.asarray([1, 0], dtype=np.uint8),
    )
    manifest = {
        "shards": [
            {
                "partition": "train",
                "path": path.name,
                "layouts": 2,
                "blocks": 4,
            }
        ]
    }
    partition = MODULE.load_partition(tmp_path, manifest, "train")
    non_mib = next(index for index in range(width) if index not in MODULE.MIB_FEATURE_INDICES)
    assert np.all(partition.features[:2, MODULE.MIB_FEATURE_INDICES] == 0.0)
    assert np.all(partition.features[2:, MODULE.MIB_FEATURE_INDICES] == 1.0)
    assert np.all(partition.features[:, non_mib] == 1.0)
    assert partition.inconsistent_count == 1
    assert MODULE.MIB_FEATURE_INDICES == MODULE.INFERENCE_MIB_FEATURE_INDICES


def test_fractional_ranks_average_ties_and_metrics_penalize_prediction_ties():
    assert MODULE.fractional_ranks(np.asarray([4.0, 1.0, 1.0])).tolist() == [
        1.0,
        0.25,
        0.25,
    ]
    targets = np.asarray([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    predictions = np.asarray([[2.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    metrics = MODULE.rank_metrics(predictions, targets, [(0, 3)])
    assert metrics["rank_mae"] == pytest.approx([2.0 / 3.0, 1.0 / 6.0])
    assert metrics["pairwise_inversion_fraction"] == pytest.approx([1.0, 1.0 / 6.0])
    assert metrics["comparable_pairs"] == [3, 3]


def test_preregistered_gate_requires_both_material_improvements():
    baseline = {
        "rank_mae": [0.08, 0.06],
        "pairwise_inversion_fraction": [0.10, 0.08],
    }
    passing = {
        "rank_mae": [0.96 * value for value in baseline["rank_mae"]],
        "pairwise_inversion_fraction": [
            0.96 * value for value in baseline["pairwise_inversion_fraction"]
        ],
    }
    assert MODULE._gate(passing, baseline)["passed"] is True

    insufficient_mae = copy.deepcopy(passing)
    insufficient_mae["rank_mae"] = [0.98 * value for value in baseline["rank_mae"]]
    assert MODULE._gate(insufficient_mae, baseline)["passed"] is False

    axis_regression = copy.deepcopy(passing)
    axis_regression["rank_mae"][0] = baseline["rank_mae"][0] + 0.0006
    assert MODULE._gate(axis_regression, baseline)["passed"] is False


def test_training_arguments_fail_closed_before_cache_access():
    args = argparse.Namespace(
        epochs=10,
        minimum_epochs=2,
        patience=2,
        batch_layouts=8,
        pairs_per_block=2,
        learning_rate=1e-3,
        pair_temperature=0.1,
        weight_decay=0.0,
        ridge=1e-4,
        pair_weight=0.05,
    )
    MODULE._validate_args(args)
    args.pair_temperature = 0.0
    with pytest.raises(ValueError, match="positive"):
        MODULE._validate_args(args)
