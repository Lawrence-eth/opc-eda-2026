import json
import shutil
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pytest

from scripts import build_dual_parent_cache as cache


REAL_VERIFY_DATA_ROOT_PROVENANCE = cache._verify_data_root_provenance
_TEST_DATASETS = {}


def _synthetic_provenance(
    data_root, holdout_paths, *, allow_dirty, ignored_roots=()
):
    del allow_dirty
    status = cache._repository_status(ignored_roots=ignored_roots)
    official_sha = cache._file_sha256(cache.ROOT / "docs" / "official_sources.json")
    loader_sha = cache._file_sha256(data_root / "lite_dataset.py")
    return {
        "repository": {
            "commit": cache._git_output(["rev-parse", "HEAD"], cache.ROOT),
            "tree": cache._git_output(["rev-parse", "HEAD^{tree}"], cache.ROOT),
            "dirty": bool(status),
            "dirty_status_sha256": cache.hashlib.sha256(
                status.encode("utf-8", errors="surrogateescape")
            ).hexdigest(),
        },
        "floorset": {
            "repository": "https://example.invalid/FloorSet.git",
            "commit": "3" * 40,
            "tree": "4" * 40,
            "data_root_mode": "pinned_git_checkout_root",
            "loader_relative_path": "lite_dataset.py",
            "loader_sha256": loader_sha,
            "loader_module_name": f"_verified_floorset_lite_{loader_sha[:16]}",
            "dataset_class_name": "FloorplanDatasetLite",
            "official_sources_sha256": official_sha,
        },
        "code_sha256": cache._code_hashes(data_root, holdout_paths),
        "runtime": {
            "python": "test",
            "numpy": np.__version__,
            "platform": "test",
        },
    }


@pytest.fixture(autouse=True)
def _synthetic_provenance_boundary(monkeypatch):
    monkeypatch.setattr(cache, "_provenance_context", _synthetic_provenance)
    monkeypatch.setattr(
        cache, "_verify_data_root_provenance", lambda provenance, data_root: None
    )
    monkeypatch.setattr(
        cache,
        "_instantiate_verified_dataset",
        lambda data_root, provenance: _TEST_DATASETS[str(data_root.resolve())],
    )


def _sample(*, inconsistent_mib=False, broken_tree=False):
    areas = [4.0, 2.0, 1.0, 2.0]
    constraints = [[0.0] * 5 for _ in areas]
    if inconsistent_mib:
        constraints[1][2] = 1.0
        constraints[2][2] = 1.0
    for block in (0, 1, 2):
        constraints[block][3] = 1.0
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
    data_root.mkdir()
    (data_root / "lite_dataset.py").write_text("# synthetic loader\n")
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
    _TEST_DATASETS[str(data_root.resolve())] = dataset
    return data_root, files, dataset, holdout, holdout_source


def _build(tmp_path, output_name="cache", **overrides):
    inputs = _fixture_inputs(tmp_path, broken_source=overrides.pop("broken_source", None))
    data_root, _files, dataset, holdout, _holdout_source = inputs
    options = {
        "data_root": data_root,
        "output_dir": tmp_path / output_name,
        "holdout_paths": [holdout],
        "min_blocks": 4,
        "max_blocks": 4,
        "max_layouts_per_source": 2,
        "shard_layouts": 3,
    }
    options.update(overrides)
    report = cache.build_cache(**options)
    return options["output_dir"], inputs, report


def _copy_and_mutate_manifest(output, destination, mutate):
    shutil.copytree(output, destination)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mutate(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return destination


def _copy_and_mutate_first_shard(output, destination, mutate):
    shutil.copytree(output, destination)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    descriptor = manifest["shards"][0]
    shard_path = destination / descriptor["path"]
    with np.load(shard_path, allow_pickle=False) as shard:
        arrays = {name: shard[name] for name in shard.files}
    mutate(arrays)
    cache._write_deterministic_npz(shard_path, arrays)
    descriptor["size_bytes"] = shard_path.stat().st_size
    descriptor["sha256"] = cache._file_sha256(shard_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return destination


def test_deterministic_identity_free_shards_and_source_partition_contract(tmp_path):
    first, inputs, first_report = _build(tmp_path, "first")
    data_root, _files, dataset, holdout, holdout_source = inputs
    second = tmp_path / "second"
    second_report = cache.build_cache(
        data_root=data_root,
        output_dir=second,
        holdout_paths=[holdout],
        min_blocks=4,
        max_blocks=4,
        max_layouts_per_source=2,
        shard_layouts=3,
    )

    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()
    manifest = json.loads((first / "manifest.json").read_text())
    assert first_report["layout_count"] == second_report["layout_count"] == 24
    assert manifest["schema_version"] == 3
    assert manifest["provenance"]["floorset"]["dataset_class_name"] == (
        "FloorplanDatasetLite"
    )
    assert manifest["provenance"]["floorset"]["loader_module_name"].startswith(
        "_verified_floorset_lite_"
    )
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
    data_root, _files, dataset, holdout, _held_out = _fixture_inputs(
        tmp_path, broken_source="layouts_0.th"
    )
    output = tmp_path / "failed-cache"
    with pytest.raises(cache.CacheError, match="selected_layout_rejected"):
        cache.build_cache(
            data_root=data_root,
            output_dir=output,
            holdout_paths=[holdout],
            min_blocks=4,
            max_blocks=4,
            max_layouts_per_source=2,
            shard_layouts=3,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".failed-cache.tmp-*"))

    output.mkdir()
    sentinel = output / "owned-by-user"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(cache.CacheError, match="output_exists"):
        cache.build_cache(
            data_root=data_root,
            output_dir=output,
            holdout_paths=[holdout],
            min_blocks=4,
            max_blocks=4,
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


def test_exact_manifest_schemas_and_cross_fields_are_fail_closed(tmp_path):
    output, _inputs, _report = _build(tmp_path)

    def empty_provenance(manifest):
        manifest["provenance"] = {}

    def invalid_seed(manifest):
        manifest["configuration"]["seed"] = True

    def invalid_tolerance(manifest):
        manifest["configuration"]["oracle_tolerance"] = 1e-5

    def invalid_rejection(manifest):
        manifest["rejection_taxonomy"]["policy"] = "skip_bad_layouts"

    def invalid_partition_algorithm(manifest):
        manifest["source_partition"]["algorithm"] = "random_layout_split"

    def invalid_code_hash(manifest):
        manifest["provenance"]["code_sha256"]["builder"] = "0" * 64

    def invalid_dirty_status(manifest):
        manifest["provenance"]["repository"]["dirty"] = False
        manifest["provenance"]["repository"]["dirty_status_sha256"] = "0" * 64

    cases = (
        ("provenance", empty_provenance, "invalid_provenance_schema"),
        ("seed", invalid_seed, "invalid_configuration_seed"),
        ("tolerance", invalid_tolerance, "invalid_oracle_tolerance"),
        ("rejection", invalid_rejection, "invalid_rejection_taxonomy"),
        (
            "partition-algorithm",
            invalid_partition_algorithm,
            "invalid_source_partition_algorithm",
        ),
        ("code-hash", invalid_code_hash, "code_provenance_mismatch"),
        (
            "dirty-status",
            invalid_dirty_status,
            "invalid_repository_provenance",
        ),
    )
    for name, mutate, expected in cases:
        changed = _copy_and_mutate_manifest(output, tmp_path / name, mutate)
        with pytest.raises(cache.CacheError, match=expected):
            cache.validate_cache(changed)


def test_source_holdout_partition_record_stats_and_manifest_tampering(tmp_path):
    output, inputs, report = _build(tmp_path)
    data_root, _files, _dataset, holdout, _held_out = inputs

    with pytest.raises(cache.CacheError, match="manifest_sha256_mismatch"):
        cache.validate_cache(output, expected_manifest_sha256="0" * 64)
    assert report["manifest_sha256"] == cache._file_sha256(output / "manifest.json")

    manifest = json.loads((output / "manifest.json").read_text())
    referenced = data_root / manifest["records"][0]["source_file"]
    original_source = referenced.read_bytes()
    referenced.write_bytes(bytes([original_source[0] ^ 1]) + original_source[1:])
    with pytest.raises(cache.CacheError, match="source_sha256_mismatch"):
        cache.validate_cache(
            output,
            data_root=data_root,
            holdout_paths=[holdout],
            expected_manifest_sha256=report["manifest_sha256"],
        )
    referenced.write_bytes(original_source)

    original_holdout = holdout.read_text()
    holdout.write_text(original_holdout + "\n")
    with pytest.raises(cache.CacheError, match="holdout_manifest_sha256_mismatch"):
        cache.validate_cache(output, holdout_paths=[holdout])
    holdout.write_text(original_holdout)

    def partition_count(manifest_value):
        manifest_value["source_partition"]["partition_source_counts"]["train"] += 1

    changed = _copy_and_mutate_manifest(output, tmp_path / "partition-count", partition_count)
    with pytest.raises(cache.CacheError, match="partition_block_total"):
        cache.validate_cache(changed)

    def record_count(manifest_value):
        manifest_value["records"][0]["b2b_edge_count"] += 1
        manifest_value["record_provenance_sha256"] = cache._canonical_json_sha256(
            manifest_value["records"]
        )

    changed = _copy_and_mutate_manifest(output, tmp_path / "record-count", record_count)
    with pytest.raises(cache.CacheError, match="record_shard_metadata_mismatch"):
        cache.validate_cache(changed)

    def stats_count(manifest_value):
        manifest_value["stats"]["node_count"] += 1

    changed = _copy_and_mutate_manifest(output, tmp_path / "stats-count", stats_count)
    with pytest.raises(cache.CacheError, match="manifest_stats_mismatch"):
        cache.validate_cache(changed)


def test_symlinks_duplicate_zip_members_and_semantic_tensor_tampering(tmp_path):
    output, _inputs, _report = _build(tmp_path)
    symlinked = tmp_path / "symlinked"
    shutil.copytree(output, symlinked)
    (symlinked / "identity-link").symlink_to(output / "manifest.json")
    with pytest.raises(cache.CacheError, match="cache_symlink_forbidden"):
        cache.validate_cache(symlinked)

    duplicate = tmp_path / "duplicate-member"
    shutil.copytree(output, duplicate)
    manifest_path = duplicate / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    descriptor = manifest["shards"][0]
    shard_path = duplicate / descriptor["path"]
    with zipfile.ZipFile(shard_path, "r") as archive:
        member_name = archive.namelist()[0]
        member_payload = archive.read(member_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(shard_path, "a") as archive:
            archive.writestr(member_name, member_payload)
    descriptor["size_bytes"] = shard_path.stat().st_size
    descriptor["sha256"] = cache._file_sha256(shard_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(cache.CacheError, match="duplicate_npz_member"):
        cache.validate_cache(duplicate)

    def bad_unselected_shape(arrays):
        block = 1
        start = int(arrays["shape_ptr"][block])
        end = int(arrays["shape_ptr"][block + 1])
        selected = int(arrays["selected_shape"][block])
        victim = next(index for index in range(end - start) if index != selected)
        arrays["shape_options"][start + victim, 0] += 1.0

    changed = _copy_and_mutate_first_shard(
        output, tmp_path / "shape-option", bad_unselected_shape
    )
    with pytest.raises(cache.CacheError, match="shape_option_set_mismatch"):
        cache.validate_cache(changed)

    def incomplete_cluster(arrays):
        arrays["cluster_pair_src"][2] = 2
        arrays["cluster_pair_dst"][2] = 3

    changed = _copy_and_mutate_first_shard(
        output, tmp_path / "cluster-clique", incomplete_cluster
    )
    with pytest.raises(cache.CacheError, match="incomplete_cluster_clique"):
        cache.validate_cache(changed)

    def incompatible_flag(arrays):
        arrays["mib_input_compatible"][0] ^= 1

    changed = _copy_and_mutate_first_shard(
        output, tmp_path / "mib-compatible", incompatible_flag
    )
    with pytest.raises(cache.CacheError, match="mib_input_compatible_mismatch"):
        cache.validate_cache(changed)


def test_npz_member_order_and_external_attributes_are_canonical(tmp_path):
    output, _inputs, _report = _build(tmp_path)

    def rewrite(destination, *, reverse=False, bad_external_attr=False):
        shutil.copytree(output, destination)
        manifest_path = destination / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        descriptor = manifest["shards"][0]
        shard_path = destination / descriptor["path"]
        with zipfile.ZipFile(shard_path, "r") as archive:
            members = [
                (info.filename, archive.read(info.filename))
                for info in archive.infolist()
            ]
        if reverse:
            members.reverse()
        temporary = shard_path.with_suffix(".rewritten")
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for index, (name, payload) in enumerate(members):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (
                    0 if bad_external_attr and index == 0 else 0o100644 << 16
                )
                archive.writestr(info, payload, compresslevel=6)
        temporary.replace(shard_path)
        descriptor["size_bytes"] = shard_path.stat().st_size
        descriptor["sha256"] = cache._file_sha256(shard_path)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return destination

    reordered = rewrite(tmp_path / "reordered", reverse=True)
    with pytest.raises(cache.CacheError, match="noncanonical_npz_member_order"):
        cache.validate_cache(reordered)
    bad_attributes = rewrite(tmp_path / "bad-attributes", bad_external_attr=True)
    with pytest.raises(cache.CacheError, match="nondeterministic_npz_member"):
        cache.validate_cache(bad_attributes)


@pytest.mark.parametrize(
    ("column", "value", "expected"),
    (
        (2, 1.5, "noninteger_constraint_group"),
        (3, 2.5, "noninteger_constraint_group"),
        (4, 3.5, "noninteger_boundary_mask"),
    ),
)
def test_fractional_constraint_identifiers_are_rejected(column, value, expected):
    constraints = [[0.0] * 5 for _ in range(4)]
    constraints[0][column] = value
    with pytest.raises(cache.CacheError, match=expected):
        cache._constraint_state(constraints, 4)


@pytest.mark.parametrize("name", ("mib", "cluster"))
def test_equality_pairs_must_encode_complete_cliques(name):
    with pytest.raises(cache.CacheError, match=f"incomplete_{name}_clique"):
        cache._validated_clique_components(
            np.asarray([0, 0], dtype=np.int16),
            np.asarray([1, 2], dtype=np.int16),
            3,
            name=name,
        )


def test_atomic_publication_refuses_a_target_created_during_build(tmp_path, monkeypatch):
    data_root, _files, dataset, holdout, _held_out = _fixture_inputs(tmp_path)
    output = tmp_path / "raced-output"
    original_publish = cache._atomic_publish_noreplace

    def racing_publish(staging, target):
        target.mkdir()
        (target / "racer-owned").write_text("preserve")
        original_publish(staging, target)

    monkeypatch.setattr(cache, "_atomic_publish_noreplace", racing_publish)
    with pytest.raises(cache.CacheError, match="output_exists"):
        cache.build_cache(
            data_root=data_root,
            output_dir=output,
            holdout_paths=[holdout],
            min_blocks=4,
            max_blocks=4,
            max_sources=2,
            max_layouts_per_source=1,
            allow_partial_partitions=True,
        )
    assert (output / "racer-owned").read_text() == "preserve"
    assert not list(tmp_path.glob(".raced-output.tmp-*"))


def test_unverified_data_root_is_rejected_by_real_provenance_gate(tmp_path):
    fake_root = tmp_path / "not-a-git-checkout"
    fake_root.mkdir()
    provenance = {
        "floorset": {
            "commit": "1" * 40,
            "tree": "2" * 40,
            "loader_relative_path": "lite_dataset.py",
            "loader_sha256": "3" * 64,
            "official_sources_sha256": "4" * 64,
            "repository": "https://example.invalid/FloorSet.git",
        }
    }
    with pytest.raises(cache.CacheError, match="data_root_not_verified_git_checkout"):
        REAL_VERIFY_DATA_ROOT_PROVENANCE(provenance, fake_root)


def test_programmatic_build_rejects_the_fake_dataset_injection_attack(tmp_path):
    data_root, _files, forged_dataset, holdout, _held_out = _fixture_inputs(tmp_path)
    arguments = {
        "data_root": data_root,
        "output_dir": tmp_path / "must-not-exist",
        "holdout_paths": [holdout],
        "min_blocks": 4,
        "max_blocks": 4,
        "max_sources": 2,
        "allow_partial_partitions": True,
    }
    with pytest.raises(TypeError, match="unexpected keyword argument 'dataset'"):
        cache.build_cache(dataset=forged_dataset, **arguments)
    with pytest.raises(TypeError, match="takes 0 positional arguments"):
        cache.build_cache(forged_dataset, **arguments)
    assert not arguments["output_dir"].exists()


def test_every_record_field_has_an_exact_json_type_contract(tmp_path):
    output, _inputs, _report = _build(tmp_path)
    record = json.loads((output / "manifest.json").read_text())["records"][0]
    cache._validate_record_types(record, 0)
    invalid_values = {}
    integer_fields = {
        "record_index",
        "file_offset",
        "source_size_bytes",
        "block_count",
        "b2b_edge_count",
        "mib_pair_count",
        "cluster_pair_count",
        "shape_option_count",
        "mib_inconsistent_block_count",
        "local_layout",
    }
    boolean_fields = {
        "clean_offset_selected",
        "strict_mib_decodable",
        "mib_input_compatible",
        "mib_features_masked",
    }
    float_fields = {
        "oracle_max_coordinate_delta",
        "oracle_max_dimension_delta",
    }
    for field in cache.RECORD_KEYS:
        if field in integer_fields:
            invalid_values[field] = False
        elif field in boolean_fields:
            invalid_values[field] = 0
        elif field in float_fields:
            invalid_values[field] = 0
        elif field == "mib_inconsistent_groups":
            invalid_values[field] = [True]
        else:
            invalid_values[field] = 7
    assert set(invalid_values) == cache.RECORD_KEYS
    for field, invalid in invalid_values.items():
        changed = dict(record)
        changed[field] = invalid
        with pytest.raises(cache.CacheError, match=f"invalid_record_type: 0: {field}"):
            cache._validate_record_types(changed, 0)


def test_full_validation_requires_digest_and_replays_every_payload_hash(tmp_path):
    output, inputs, report = _build(tmp_path)
    data_root, _files, _dataset, holdout, _held_out = inputs
    with pytest.raises(
        cache.CacheError, match="expected_manifest_required_for_full_validation"
    ):
        cache.validate_cache(output, data_root=data_root, holdout_paths=[holdout])

    payload_fields = (
        "input_sha256",
        "optimizer_target_sha256",
        "tree_sha256",
        "golden_geometry_sha256",
        "golden_metrics_sha256",
    )
    for ordinal, field in enumerate(payload_fields):
        changed = tmp_path / f"payload-{ordinal}"

        def mutate(manifest, selected=field):
            manifest["records"][0][selected] = "0" * 64
            manifest["record_provenance_sha256"] = cache._canonical_json_sha256(
                manifest["records"]
            )

        _copy_and_mutate_manifest(output, changed, mutate)
        digest = cache._file_sha256(changed / "manifest.json")
        with pytest.raises(cache.CacheError, match=f"{field}$"):
            cache.validate_cache(
                changed,
                data_root=data_root,
                holdout_paths=[holdout],
                expected_manifest_sha256=digest,
            )
    assert report["manifest_sha256"] == cache._file_sha256(output / "manifest.json")


def test_vertical_support_requires_physical_x_overlap():
    golden = np.asarray(
        [[0.0, 1.0, 1.0, 1.0], [2.0, 0.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    supports = np.asarray([1, -1], dtype=np.int16)
    with pytest.raises(cache.CacheError, match="vertical_support_no_x_overlap"):
        cache._validate_vertical_support_overlap(golden, supports)
