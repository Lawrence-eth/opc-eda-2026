import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "train_structured_predictor", ROOT / "scripts" / "train_structured_predictor.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _manifest(path, sources):
    path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "split_unit": "source_file",
                "manifests": [
                    {"cases": [{"source_file": source, "score": 999} for source in sources]}
                ],
            }
        )
    )
    return path


def test_holdout_loader_unions_three_manifests_at_source_level_only(tmp_path):
    paths = [
        _manifest(tmp_path / "clean.json", ["floorset_lite/worker_0/a.th"]),
        _manifest(
            tmp_path / "raw.json",
            ["floorset_lite/worker_0/a.th", "floorset_lite/worker_1/b.th"],
        ),
        _manifest(tmp_path / "sealed.json", ["floorset_lite/worker_2/c.th"]),
    ]
    union = MODULE.load_holdout_union(paths)
    assert union.sources == frozenset(
        {
            "floorset_lite/worker_0/a.th",
            "floorset_lite/worker_1/b.th",
            "floorset_lite/worker_2/c.th",
        }
    )
    assert len(union.sha256s) == 3
    assert len(union.aggregate_sha256) == 64


def test_source_cap_is_applied_after_heavy_filter_and_exclusions():
    records = [
        MODULE.SourceRecord(index, f"light/{index}.th", 50, 10)
        for index in range(20)
    ] + [
        MODULE.SourceRecord(20 + index, f"heavy/{index}.th", 100 + index % 2, 10)
        for index in range(8)
    ]
    training, validation = MODULE.partition_sources(
        records,
        excluded=frozenset({"heavy/7.th"}),
        min_blocks=100,
        max_blocks=120,
        validation_fraction=0.25,
        seed=17,
        max_sources=6,
    )
    selected = training + validation
    assert len(selected) == 6
    assert all(row.block_count >= 100 for row in selected)
    assert all(row.relative_path != "heavy/7.th" for row in selected)
    assert {row.relative_path for row in training}.isdisjoint(
        row.relative_path for row in validation
    )


def test_layout_cache_labels_are_structural_and_features_are_inference_visible():
    sample = {
        "input": (
            torch.tensor([4.0, 4.0]),
            torch.tensor([[0.0, 1.0, 1.0]]),
            torch.empty((0, 3)),
            torch.empty((0, 2)),
            torch.zeros((2, 5)),
        ),
        "label": (
            torch.tensor([[0.0, 1.0, 0.0]]),
            torch.tensor([[2.0, 2.0, 0.0, 0.0], [2.0, 2.0, 2.0, 0.0]]),
            torch.zeros(3),
        ),
    }
    arrays = MODULE.layout_arrays(sample, message_steps=1)
    assert arrays["features"].shape == (2, len(MODULE.FEATURE_NAMES))
    assert arrays["root"].tolist() == [0]
    assert arrays["horizontal_parent"].tolist() == [-1, 0]
    assert arrays["horizontal_side"].tolist() == [-1, 0]
    assert arrays["vertical_parent"].tolist() == [-1, -1]
    assert arrays["shape_index"].tolist() == [0, 0]
    assert arrays["pair_direct"].shape == (
        4,
        len(MODULE.PAIR_DIRECT_FEATURE_NAMES),
    )
    assert all(
        not any(token in name.lower() for token in MODULE.FORBIDDEN_FEATURE_TOKENS)
        for name in MODULE.FEATURE_NAMES
    )


def test_shard_and_manifest_hashes_fail_closed(tmp_path):
    builder = MODULE.ShardBuilder(tmp_path, "train", shard_layouts=2)
    arrays = {
        "features": np.zeros((2, len(MODULE.FEATURE_NAMES)), dtype=np.float32),
        "shape_index": np.zeros(2, dtype=np.int8),
        "shape_count": np.ones(2, dtype=np.int8),
        "shape_mask": np.ones(2, dtype=np.uint8),
        "x_rank": np.asarray([0.0, 1.0], dtype=np.float32),
        "y_rank": np.asarray([0.0, 1.0], dtype=np.float32),
        "horizontal_parent": np.asarray([-1, 0], dtype=np.int16),
        "horizontal_side": np.asarray([-1, 0], dtype=np.int8),
        "vertical_parent": np.asarray([-1, 0], dtype=np.int16),
        "root": np.asarray([0], dtype=np.int16),
        "mib_inconsistent": np.asarray([0], dtype=np.uint8),
        "pair_direct": np.zeros(
            (4, len(MODULE.PAIR_DIRECT_FEATURE_NAMES)), dtype=np.float32
        ),
    }
    builder.add(arrays, {"source_file": "provenance-only", "file_offset": 0})
    builder.flush()
    manifest = {
        "schema_version": MODULE.CACHE_SCHEMA_VERSION,
        "cache_type": "dual_parent_supervision_shards",
        "feature_schema": {
            "version": MODULE.FEATURE_VERSION,
            "names": list(MODULE.FEATURE_NAMES),
            "message_steps": 1,
            "pair_direct_names": list(MODULE.PAIR_DIRECT_FEATURE_NAMES),
        },
        "shards": builder.shards,
    }
    manifest["payload_sha256"] = MODULE._canonical_json_sha256(manifest)
    MODULE._atomic_json(tmp_path / "manifest.json", manifest)
    assert MODULE.load_cache(tmp_path)["shards"][0]["blocks"] == 2

    shard = tmp_path / builder.shards[0]["path"]
    shard.write_bytes(shard.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="shard hash"):
        MODULE.load_cache(tmp_path)
