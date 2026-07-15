import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from scripts import build_dual_parent_cache as cache


def _sample(*, inconsistent_mib=False, broken_tree=False):
    areas = [4.0, 2.0, 1.0, 2.0]
    constraints = [[0.0] * 5 for _ in areas]
    if inconsistent_mib:
        constraints[1][2] = 1.0
        constraints[2][2] = 1.0
    tree = [[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [2.0, 3.0, 1.0]]
    if broken_tree:
        tree[1] = [0.0, 2.0, 0.0]
    # FloorSet labels are (width, height, x, y), unlike decoder rectangles.
    fp_solution = [
        [2.0, 2.0, 0.0, 0.0],
        [2.0, 1.0, 2.0, 0.0],
        [1.0, 1.0, 2.0, 3.0],
        [1.0, 2.0, 2.0, 1.0],
    ]
    return {
        "input": (
            areas,
            [[0.0, 1.0, 2.0], [1.0, 3.0, 1.0]],
            [],
            [],
            constraints,
        ),
        "label": (tree, fp_solution, [12.0, 3.0, 4.0]),
    }


class _FakeDataset:
    layouts_per_file = 2

    def __init__(self, files, *, broken_source=None):
        self.all_files = [str(path) for path in reversed(files)]
        self._samples = {}
        for path in files:
            resolved = str(path.resolve())
            self._samples[resolved] = (
                _sample(broken_tree=path.name == broken_source),
                _sample(inconsistent_mib=True),
            )

    def __getitem__(self, index):
        file_index, offset = divmod(index, self.layouts_per_file)
        return self._samples[str(Path(self.all_files[file_index]).resolve())][offset]


def _holdout_manifest(path, source):
    value = {
        "schema_version": 1,
        "split_unit": "source_file",
        "cases": [{"source_file": source}],
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _fixture_inputs(tmp_path, *, broken_source=None):
    data_root = tmp_path / "data"
    source_dir = data_root / "floorset_lite" / "worker_0"
    source_dir.mkdir(parents=True)
    files = []
    for index in range(13):
        path = source_dir / f"layouts_{index}.th"
        path.write_bytes(f"synthetic-source-{index}\n".encode())
        files.append(path)
    holdout_source = str(files[-1].relative_to(data_root)).replace("\\", "/")
    holdout = _holdout_manifest(tmp_path / "holdout.json", holdout_source)
    dataset = _FakeDataset(files, broken_source=broken_source)
    provenance = {
        "repository": {
            "commit": "1" * 40,
            "tree": "2" * 40,
            "dirty": False,
            "dirty_status_sha256": "0" * 64,
        },
        "floorset": {
            "commit": "3" * 40,
            "tree": "4" * 40,
            "official_sources_sha256": "5" * 64,
        },
        "code_sha256": {"synthetic": "6" * 64},
        "runtime": {"python": "test", "numpy": np.__version__, "platform": "test"},
    }
    return data_root, files, dataset, holdout, provenance, holdout_source


def _build(tmp_path, output_name="cache", **overrides):
    inputs = _fixture_inputs(tmp_path, broken_source=overrides.pop("broken_source", None))
    data_root, _files, dataset, holdout, provenance, _holdout_source = inputs
    options = {
        "data_root": data_root,
        "output_dir": tmp_path / output_name,
        "holdout_paths": [holdout],
        "min_blocks": 4,
        "max_blocks": 4,
        "max_layouts_per_source": 2,
        "shard_layouts": 3,
        "provenance_context": provenance,
    }
    options.update(overrides)
    report = cache.build_cache(dataset, **options)
    return options["output_dir"], inputs, report


def test_deterministic_identity_free_shards_and_source_partition_contract(tmp_path):
    first, inputs, first_report = _build(tmp_path, "first")
    data_root, _files, dataset, holdout, provenance, holdout_source = inputs
    second = tmp_path / "second"
    second_report = cache.build_cache(
        dataset,
        data_root=data_root,
        output_dir=second,
        holdout_paths=[holdout],
        min_blocks=4,
        max_blocks=4,
        max_layouts_per_source=2,
        shard_layouts=3,
        provenance_context=provenance,
    )

    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()
    manifest = json.loads((first / "manifest.json").read_text())
    assert first_report["layout_count"] == second_report["layout_count"] == 24
    assert manifest["array_contract"]["model_input_arrays"] == list(
        cache.MODEL_INPUT_ARRAYS
    )
    assert manifest["array_contract"]["supervision_arrays"] == list(
        cache.SUPERVISION_ARRAYS
    )

    partition_sources = {
        name: {
            row["source_file"]
            for row in manifest["records"]
            if row["partition"] == name
        }
        for name in cache.PARTITION_NAMES
    }
    assert all(partition_sources.values())
    assert not (partition_sources["train"] & partition_sources["development"])
    assert not (partition_sources["train"] & partition_sources["calibration"])
    assert not (partition_sources["development"] & partition_sources["calibration"])
    assert holdout_source not in set().union(*partition_sources.values())

    for descriptor in manifest["shards"]:
        first_bytes = (first / descriptor["path"]).read_bytes()
        assert first_bytes == (second / descriptor["path"]).read_bytes()
        with np.load(first / descriptor["path"], allow_pickle=False) as shard:
            assert set(shard.files) == set(cache.ALL_ARRAYS)
            for name in shard.files:
                assert shard[name].dtype.kind not in "OUSV"
                assert not any(
                    fragment in name.lower()
                    for fragment in cache.IDENTITY_KEY_FRAGMENTS
                )


def test_raw_mib_inconsistency_masks_only_affected_shape_supervision(tmp_path):
    output, _inputs, _report = _build(tmp_path)
    manifest = json.loads((output / "manifest.json").read_text())
    record = next(row for row in manifest["records"] if row["file_offset"] == 1)
    assert record["mib_inconsistent_groups"] == [1]
    assert record["strict_mib_decodable"] is False
    assert record["oracle_max_coordinate_delta"] == 0.0
    assert record["oracle_max_dimension_delta"] == 0.0

    with np.load(output / record["shard"], allow_pickle=False) as shard:
        layout = record["local_layout"]
        start = int(shard["layout_ptr"][layout])
        end = int(shard["layout_ptr"][layout + 1])
        assert shard["shape_supervision_mask"][start:end].tolist() == [1, 0, 0, 1]
        assert shard["mib_consistent_mask"][start:end].tolist() == [1, 0, 0, 1]
        assert int(shard["strict_mib_decodable"][layout]) == 0
        assert int(shard["mib_input_compatible"][layout]) == 0
        assert int(shard["mib_features_masked"][layout]) == 1
        # The topology and individual shape categories remain available.
        assert shard["horizontal_parent"][start:end].tolist() == [-1, 0, 1, 2]
        assert np.all(np.diff(shard["shape_ptr"][start : end + 1]) > 0)


def test_validator_rejects_tampering_path_escape_and_duplicate_json_keys(tmp_path):
    output, _inputs, _report = _build(tmp_path)
    tampered = tmp_path / "tampered"
    shutil.copytree(output, tampered)
    manifest = json.loads((tampered / "manifest.json").read_text())
    shard_path = tampered / manifest["shards"][0]["path"]
    payload = bytearray(shard_path.read_bytes())
    payload[-1] ^= 1
    shard_path.write_bytes(payload)
    with pytest.raises(cache.CacheError, match="shard_sha256_mismatch"):
        cache.validate_cache(tampered)

    escaped = tmp_path / "escaped"
    shutil.copytree(output, escaped)
    manifest_path = escaped / "manifest.json"
    value = json.loads(manifest_path.read_text())
    value["shards"][0]["path"] = "../outside.npz"
    manifest_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(cache.CacheError, match="unsafe_relative_path"):
        cache.validate_cache(escaped)

    duplicate = tmp_path / "duplicate"
    shutil.copytree(output, duplicate)
    manifest_path = duplicate / "manifest.json"
    original = manifest_path.read_text()
    manifest_path.write_text(original.replace("{", '{"schema_version":1,', 1))
    with pytest.raises(cache.CacheError, match="duplicate_json_key"):
        cache.validate_cache(duplicate)

    extra = tmp_path / "extra"
    shutil.copytree(output, extra)
    (extra / "source_paths.txt").write_text("worker_0/layouts_0.th\n")
    with pytest.raises(cache.CacheError, match="unexpected_cache_files"):
        cache.validate_cache(extra)


def test_validator_replays_semantics_even_after_shard_descriptor_is_rehashed(tmp_path):
    output, _inputs, _report = _build(tmp_path)
    changed = tmp_path / "semantic-tamper"
    shutil.copytree(output, changed)
    manifest_path = changed / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    descriptor = manifest["shards"][0]
    shard_path = changed / descriptor["path"]
    with np.load(shard_path, allow_pickle=False) as shard:
        arrays = {name: shard[name] for name in shard.files}
    first_start = int(arrays["layout_ptr"][0])
    root = int(arrays["root"][0])
    victim = next(block for block in range(4) if block != root)
    arrays["horizontal_parent"][first_start + victim] = victim
    cache._write_deterministic_npz(shard_path, arrays)
    descriptor["size_bytes"] = shard_path.stat().st_size
    descriptor["sha256"] = cache._file_sha256(shard_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(cache.CacheError, match="invalid_horizontal_parent"):
        cache.validate_cache(changed)


def test_failed_build_is_not_published_and_existing_output_is_never_replaced(tmp_path):
    data_root, _files, dataset, holdout, provenance, _held_out = _fixture_inputs(
        tmp_path, broken_source="layouts_0.th"
    )
    output = tmp_path / "failed-cache"
    with pytest.raises(cache.CacheError, match="selected_layout_rejected"):
        cache.build_cache(
            dataset,
            data_root=data_root,
            output_dir=output,
            holdout_paths=[holdout],
            min_blocks=4,
            max_blocks=4,
            max_layouts_per_source=2,
            shard_layouts=3,
            provenance_context=provenance,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".failed-cache.tmp-*"))

    output.mkdir()
    sentinel = output / "owned-by-user"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(cache.CacheError, match="output_exists"):
        cache.build_cache(
            dataset,
            data_root=data_root,
            output_dir=output,
            holdout_paths=[holdout],
            min_blocks=4,
            max_blocks=4,
            provenance_context=provenance,
        )
    assert sentinel.read_text() == "keep"


def test_max_sources_smoke_is_deterministic_and_explicitly_partial(tmp_path):
    output, _inputs, report = _build(
        tmp_path,
        max_sources=2,
        max_layouts_per_source=1,
        allow_partial_partitions=True,
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert report["source_count"] == 2
    assert report["layout_count"] == 2
    assert manifest["source_partition"]["partial_partitions_allowed"] is True
    assert manifest["source_partition"]["selection"]["selected_after_limit"] == 2
