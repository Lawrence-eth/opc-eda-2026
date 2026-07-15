#!/usr/bin/env python3
"""Build and verify a leakage-safe dual-parent FloorSet training cache.

The cache is deliberately a research artifact, not a deployable model.  Source
identity and raw-label provenance live only in ``manifest.json``.  Numeric NPZ
shards contain input-visible tensors and explicitly separated supervision; they
never contain a source path, worker number, file offset, or opaque Python object.

Successful builds are published with one atomic directory rename.  Any selected
layout that cannot be extracted and decoded exactly aborts the complete build.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOLUTION_DIR = ROOT / "contest_solution"
FLOORSET = ROOT / "external" / "FloorSet"
for import_path in (ROOT, SOLUTION_DIR, FLOORSET):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from dual_parent_decoder import (  # noqa: E402
    DualParentError,
    compare_geometry,
    decode_dual_parent,
    extract_oracle_labels,
    hard_targets_from_golden,
    training_rectangles,
)
from learned_order import (  # noqa: E402
    FEATURE_NAMES,
    FEATURE_VERSION,
    apply_mib_feature_policy,
    extract_order_features,
)
from lite_dataset import FloorplanDatasetLite  # noqa: E402
from scripts.train_order_model import (  # noqa: E402
    SourceFile,
    _block_count,
    load_holdout_sources,
    partition_source_files,
    select_layout_offsets,
)


CACHE_SCHEMA_VERSION = 1
BUILDER_VERSION = 1
MANIFEST_NAME = "manifest.json"
DEFAULT_HOLDOUTS = (
    ROOT / "results" / "folds" / "heavy_clean_v1.json",
    ROOT / "results" / "folds" / "heavy_raw_hash_v1.json",
    ROOT / "results" / "folds" / "heavy_sealed_v2.json",
)
PARTITION_NAMES = ("train", "development", "calibration")
MODEL_INPUT_ARRAYS = (
    "layout_ptr",
    "edge_ptr",
    "mib_pair_ptr",
    "cluster_pair_ptr",
    "node_features",
    "area_targets",
    "hard_targets",
    "b2b_src",
    "b2b_dst",
    "b2b_weight",
    "mib_pair_src",
    "mib_pair_dst",
    "cluster_pair_src",
    "cluster_pair_dst",
    "fixed_mask",
    "preplaced_mask",
    "boundary_mask",
)
SUPERVISION_ARRAYS = (
    "shape_ptr",
    "dimensions",
    "shape_options",
    "selected_shape",
    "shape_supervision_mask",
    "mib_consistent_mask",
    "horizontal_parent",
    "horizontal_side",
    "vertical_support",
    "golden_rectangles",
    "root",
    "strict_mib_decodable",
    "mib_input_compatible",
    "mib_features_masked",
)
ARRAY_DTYPES = {
    "layout_ptr": "int64",
    "edge_ptr": "int64",
    "mib_pair_ptr": "int64",
    "cluster_pair_ptr": "int64",
    "shape_ptr": "int64",
    "node_features": "float32",
    "area_targets": "float32",
    "hard_targets": "float32",
    "b2b_src": "int16",
    "b2b_dst": "int16",
    "b2b_weight": "float32",
    "mib_pair_src": "int16",
    "mib_pair_dst": "int16",
    "cluster_pair_src": "int16",
    "cluster_pair_dst": "int16",
    "fixed_mask": "uint8",
    "preplaced_mask": "uint8",
    "boundary_mask": "uint8",
    "dimensions": "float32",
    "shape_options": "float32",
    "selected_shape": "int16",
    "shape_supervision_mask": "uint8",
    "mib_consistent_mask": "uint8",
    "horizontal_parent": "int16",
    "horizontal_side": "int8",
    "vertical_support": "int16",
    "golden_rectangles": "float32",
    "root": "int16",
    "strict_mib_decodable": "uint8",
    "mib_input_compatible": "uint8",
    "mib_features_masked": "uint8",
}
ALL_ARRAYS = MODEL_INPUT_ARRAYS + SUPERVISION_ARRAYS
IDENTITY_KEY_FRAGMENTS = (
    "source",
    "worker",
    "path",
    "filename",
    "file_name",
    "file_offset",
    "layout_offset",
    "record_id",
    "instance_id",
)
MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "builder_version",
        "mode",
        "array_contract",
        "configuration",
        "provenance",
        "source_partition",
        "rejection_taxonomy",
        "shards",
        "records",
        "record_provenance_sha256",
        "stats",
    }
)
SHARD_DESCRIPTOR_KEYS = frozenset(
    {
        "path",
        "partition",
        "sha256",
        "size_bytes",
        "layout_count",
        "node_count",
        "b2b_edge_count",
        "mib_pair_count",
        "cluster_pair_count",
        "shape_option_count",
    }
)
RECORD_KEYS = frozenset(
    {
        "record_index",
        "partition",
        "source_file",
        "file_offset",
        "source_sha256",
        "source_size_bytes",
        "clean_offset_selected",
        "block_count",
        "b2b_edge_count",
        "mib_pair_count",
        "cluster_pair_count",
        "shape_option_count",
        "mib_inconsistent_groups",
        "mib_inconsistent_block_count",
        "strict_mib_decodable",
        "mib_input_compatible",
        "mib_features_masked",
        "oracle_max_coordinate_delta",
        "oracle_max_dimension_delta",
        "input_sha256",
        "optimizer_target_sha256",
        "tree_sha256",
        "golden_geometry_sha256",
        "golden_metrics_sha256",
        "shard",
        "local_layout",
    }
)


class CacheError(ValueError):
    """A stable, fail-closed cache construction or validation error."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(code if not detail else f"{code}: {detail}")


def _scalar(value: Any) -> float:
    value = value.item() if hasattr(value, "item") else value
    result = float(value)
    if not math.isfinite(result):
        raise CacheError("nonfinite_value", repr(value))
    return result


def _rows(value: Any) -> list[list[float]]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    rows = []
    for row in value:
        if not isinstance(row, (list, tuple)):
            row = [row]
        rows.append([_scalar(item) for item in row])
    return rows


def _values(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    return [_scalar(item) for item in value]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_path(value: str) -> str:
    text = str(value).replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    path = PurePosixPath(text)
    if not text or str(path) == "." or path.is_absolute() or ".." in path.parts:
        raise CacheError("unsafe_relative_path", repr(value))
    return str(path)


def _json_object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CacheError("duplicate_json_key", key)
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_json_object_no_duplicates,
        )
    except CacheError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CacheError("invalid_manifest_json", str(exc)) from exc
    if not isinstance(value, dict):
        raise CacheError("manifest_not_object")
    return value


def _hash_tensor(namespace: str, value: Any) -> str:
    """Hash a tensor without converting its numeric payload through JSON."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise CacheError("unsupported_tensor_dtype", str(array.dtype))
    if array.dtype.byteorder == ">" or (
        array.dtype.byteorder == "=" and sys.byteorder == "big"
    ):
        array = array.byteswap().view(array.dtype.newbyteorder("<"))
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(b"dual_parent_tensor_v1\0")
    digest.update(namespace.encode("utf-8") + b"\0")
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _hash_input(sample: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(b"dual_parent_input_v1\0")
    for index, value in enumerate(sample["input"]):
        digest.update(bytes.fromhex(_hash_tensor(f"input[{index}]", value)))
    return digest.hexdigest()


def _hash_hard_targets(targets: list[tuple[float, float, float, float]]) -> str:
    return _hash_tensor("optimizer_hard_targets", np.asarray(targets, dtype="<f8"))


def _git_output(arguments: list[str], cwd: Path) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CacheError("git_provenance_failed", f"{cwd}: {arguments}: {exc}") from exc
    return result.stdout.strip()


def _code_hashes(holdout_paths: Iterable[Path]) -> dict[str, str]:
    paths = {
        "builder": Path(__file__),
        "dual_parent_decoder": SOLUTION_DIR / "dual_parent_decoder.py",
        "learned_order_features": SOLUTION_DIR / "learned_order.py",
        "source_partition_implementation": ROOT / "scripts" / "train_order_model.py",
        "floorset_lite_loader": FLOORSET / "lite_dataset.py",
        "official_sources": ROOT / "docs" / "official_sources.json",
    }
    hashes = {name: _file_sha256(path) for name, path in paths.items()}
    hashes["holdout_manifest_union"] = _canonical_json_sha256(
        [_file_sha256(Path(path)) for path in holdout_paths]
    )
    return dict(sorted(hashes.items()))


def _provenance_context(
    holdout_paths: list[Path], *, allow_dirty: bool
) -> dict[str, Any]:
    repository_status = _git_output(
        ["status", "--porcelain=v1", "--untracked-files=all"], ROOT
    )
    if repository_status and not allow_dirty:
        preview = repository_status.splitlines()[0]
        raise CacheError("dirty_repository", preview)
    floorset_commit = _git_output(["rev-parse", "HEAD"], FLOORSET)
    floorset_tree = _git_output(["rev-parse", "HEAD^{tree}"], FLOORSET)
    official = _load_json(ROOT / "docs" / "official_sources.json")
    pinned = official.get("floorset", {})
    if floorset_commit != pinned.get("commit") or floorset_tree != pinned.get("tree"):
        raise CacheError(
            "floorset_revision_mismatch",
            f"observed {floorset_commit}/{floorset_tree}",
        )
    return {
        "repository": {
            "commit": _git_output(["rev-parse", "HEAD"], ROOT),
            "tree": _git_output(["rev-parse", "HEAD^{tree}"], ROOT),
            "dirty": bool(repository_status),
            "dirty_status_sha256": hashlib.sha256(
                repository_status.encode("utf-8")
            ).hexdigest(),
        },
        "floorset": {
            "commit": floorset_commit,
            "tree": floorset_tree,
            "official_sources_sha256": _file_sha256(
                ROOT / "docs" / "official_sources.json"
            ),
        },
        "code_sha256": _code_hashes(holdout_paths),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }


def _split_development_calibration(
    candidates: list[SourceFile], seed: int
) -> tuple[list[SourceFile], list[SourceFile]]:
    buckets: dict[int, list[SourceFile]] = {}
    for source in candidates:
        buckets.setdefault(source.block_count, []).append(source)
    development: list[SourceFile] = []
    calibration: list[SourceFile] = []
    for block_count in sorted(buckets):
        ordered = sorted(
            buckets[block_count],
            key=lambda source: hashlib.sha256(
                f"dual-parent-devcal:{seed}:{source.relative_path}".encode()
            ).digest(),
        )
        if len(ordered) == 1:
            development.extend(ordered)
            continue
        midpoint = (len(ordered) + 1) // 2
        development.extend(ordered[:midpoint])
        calibration.extend(ordered[midpoint:])
    if len(candidates) >= 2 and not calibration:
        development.sort(key=lambda source: source.relative_path)
        calibration.append(development.pop())
    development.sort(key=lambda source: source.relative_path)
    calibration.sort(key=lambda source: source.relative_path)
    return development, calibration


def _partition_sources(
    dataset: Any,
    *,
    data_root: Path,
    holdout_paths: list[Path],
    seed: int,
    max_sources: int | None,
    min_blocks: int,
    max_blocks: int,
    source_index_cache: Path | None,
    progress_every_sources: int,
    allow_partial_partitions: bool,
) -> tuple[dict[str, list[SourceFile]], dict[str, Any]]:
    holdout = load_holdout_sources(holdout_paths)
    train, reserve, selection, source_index = partition_source_files(
        dataset,
        data_root=data_root,
        excluded_sources=holdout.paths,
        seed=seed,
        validation_fraction=0.2,
        max_files=max_sources,
        min_blocks=min_blocks,
        max_blocks=max_blocks,
        source_index_cache=source_index_cache,
        progress_every_files=progress_every_sources,
    )
    development, calibration = _split_development_calibration(reserve, seed)
    partitions = {
        "train": train,
        "development": development,
        "calibration": calibration,
    }
    source_sets = {
        name: {source.relative_path for source in sources}
        for name, sources in partitions.items()
    }
    for index, first in enumerate(PARTITION_NAMES):
        if source_sets[first] & holdout.paths:
            raise CacheError("holdout_leakage", first)
        for second in PARTITION_NAMES[index + 1 :]:
            overlap = source_sets[first] & source_sets[second]
            if overlap:
                raise CacheError(
                    "source_partition_overlap", f"{first}/{second}: {min(overlap)}"
                )
    if not allow_partial_partitions and any(not partitions[name] for name in PARTITION_NAMES):
        raise CacheError("empty_source_partition")
    partition_rows = {
        name: [
            {"source_file": source.relative_path, "block_count": source.block_count}
            for source in partitions[name]
        ]
        for name in PARTITION_NAMES
    }
    return partitions, {
        "algorithm": "source_file_disjoint_block_stratified_train_dev_cal_v1",
        "partition_sha256": _canonical_json_sha256(partition_rows),
        "partition_source_counts": {
            name: len(partitions[name]) for name in PARTITION_NAMES
        },
        "partition_block_counts": {
            name: dict(
                sorted(
                    Counter(source.block_count for source in partitions[name]).items()
                )
            )
            for name in PARTITION_NAMES
        },
        "selection": selection,
        "source_index": {
            key: value for key, value in source_index.items() if key != "cache_path"
        },
        "holdout": {
            "split_unit": holdout.manifest_split_unit,
            "excluded_source_count": len(holdout.paths),
            "manifest_union_sha256": holdout.manifest_sha256,
            "manifest_sha256s": list(holdout.manifest_sha256s),
            "manifest_schema_versions": list(holdout.manifest_schema_versions),
        },
        "partial_partitions_allowed": allow_partial_partitions,
    }


def _valid_b2b_edges(value: Any, block_count: int) -> tuple[list[int], list[int], list[float]]:
    sources: list[int] = []
    destinations: list[int] = []
    weights: list[float] = []
    for row_index, row in enumerate(_rows(value)):
        if len(row) < 3:
            raise CacheError("malformed_b2b_edge", str(row_index))
        first_number, second_number, weight = row[:3]
        first, second = int(first_number), int(second_number)
        if first_number != first or second_number != second:
            raise CacheError("noninteger_b2b_index", str(row_index))
        if weight <= 0.0:
            # Match the shared feature implementation: non-positive rows are
            # padding/sentinels and never become model edges.
            continue
        if not (0 <= first < block_count and 0 <= second < block_count):
            raise CacheError("b2b_index_out_of_range", str(row_index))
        if first == second:
            raise CacheError("b2b_self_edge", str(row_index))
        sources.append(first)
        destinations.append(second)
        weights.append(weight)
    return sources, destinations, weights


def _constraint_state(
    constraints: Any, block_count: int
) -> tuple[
    list[int],
    list[int],
    list[list[int]],
    dict[int, list[int]],
    dict[int, list[int]],
]:
    rows = _rows(constraints)
    if rows and len(rows) < block_count:
        raise CacheError("constraint_count", f"{len(rows)} < {block_count}")
    fixed: list[int] = []
    preplaced: list[int] = []
    boundary: list[list[int]] = []
    mib: dict[int, list[int]] = {}
    cluster: dict[int, list[int]] = {}
    for block in range(block_count):
        row = rows[block] if rows else []
        fixed.append(int(len(row) > 0 and row[0] != 0.0))
        preplaced.append(int(len(row) > 1 and row[1] != 0.0))
        mib_id = int(row[2]) if len(row) > 2 else 0
        cluster_id = int(row[3]) if len(row) > 3 else 0
        if mib_id < 0 or cluster_id < 0:
            raise CacheError("negative_constraint_group", str(block))
        if mib_id:
            mib.setdefault(mib_id, []).append(block)
        if cluster_id:
            cluster.setdefault(cluster_id, []).append(block)
        boundary_code = int(row[4]) if len(row) > 4 else 0
        if boundary_code < 0 or boundary_code > 15:
            raise CacheError("invalid_boundary_mask", str(block))
        boundary.append([(boundary_code >> bit) & 1 for bit in range(4)])
    return fixed, preplaced, boundary, mib, cluster


def _group_pairs(groups: dict[int, list[int]]) -> tuple[list[int], list[int]]:
    first: list[int] = []
    second: list[int] = []
    for members in groups.values():
        ordered = sorted(members)
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                first.append(left)
                second.append(right)
    return first, second


def _extract_layout(
    sample: dict[str, Any], *, message_steps: int, mib_feature_policy: str
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    try:
        area_targets, b2b, p2b, pins, constraints = sample["input"]
        tree, fp_solution, metrics = sample["label"]
    except (KeyError, TypeError, ValueError) as exc:
        raise CacheError("malformed_sample", str(exc)) from exc
    block_count = _block_count(area_targets)
    if not 1 <= block_count <= np.iinfo(np.int16).max:
        raise CacheError("invalid_block_count", str(block_count))
    areas = _values(area_targets)[:block_count]
    if len(areas) != block_count or any(value <= 0.0 for value in areas):
        raise CacheError("invalid_area_targets")
    golden = training_rectangles(fp_solution, block_count)
    hard_targets = hard_targets_from_golden(constraints, golden)
    fixed, preplaced, boundary, mib_groups, cluster_groups = _constraint_state(
        constraints, block_count
    )
    try:
        features = extract_order_features(
            block_count,
            area_targets,
            b2b,
            p2b,
            pins,
            constraints,
            hard_targets,
            message_steps=message_steps,
        )
        features, mib_metadata = apply_mib_feature_policy(
            features,
            policy=mib_feature_policy,
            block_count=block_count,
            area_targets=area_targets,
            constraints=constraints,
            target_positions=hard_targets,
        )
        labels = extract_oracle_labels(
            area_targets, constraints, tree, golden
        )
    except DualParentError:
        raise
    except (IndexError, TypeError, ValueError, OverflowError) as exc:
        raise CacheError("layout_extraction_failed", str(exc)) from exc

    strict_mib_decodable = True
    try:
        decoded = decode_dual_parent(
            labels,
            constraints=constraints,
            hard_targets=hard_targets,
            enforce_mib=True,
        )
    except DualParentError as exc:
        if exc.code != "mib_shape_mismatch" or not labels.mib_inconsistent_groups:
            raise
        strict_mib_decodable = False
        decoded = decode_dual_parent(
            labels,
            constraints=constraints,
            hard_targets=hard_targets,
            enforce_mib=False,
        )
    comparison = compare_geometry(decoded, golden)
    if not comparison.is_exact():
        raise CacheError(
            "oracle_roundtrip_mismatch",
            f"coordinate={comparison.max_coordinate_delta}, "
            f"dimension={comparison.max_dimension_delta}",
        )

    horizontal_parent = [-1] * block_count
    horizontal_side = [-1] * block_count
    for relation in labels.horizontal:
        horizontal_parent[relation.child] = relation.parent
        horizontal_side[relation.child] = relation.side
    vertical_support = [
        -1 if support is None else support for support in labels.vertical_supports
    ]
    inconsistent_members = {
        block
        for group_id in labels.mib_inconsistent_groups
        for block in mib_groups.get(group_id, [])
    }
    shape_supervision_mask = [
        int(not (fixed[index] or preplaced[index]) and index not in inconsistent_members)
        for index in range(block_count)
    ]
    mib_consistent_mask = [
        int(index not in inconsistent_members) for index in range(block_count)
    ]
    shape_options = [shape for options in labels.shape_options for shape in options]
    shape_counts = [len(options) for options in labels.shape_options]
    b2b_src, b2b_dst, b2b_weight = _valid_b2b_edges(b2b, block_count)
    mib_src, mib_dst = _group_pairs(mib_groups)
    cluster_src, cluster_dst = _group_pairs(cluster_groups)

    arrays = {
        "node_features": np.asarray(features, dtype=np.float32),
        "area_targets": np.asarray(areas, dtype=np.float32),
        "hard_targets": np.asarray(hard_targets, dtype=np.float32),
        "b2b_src": np.asarray(b2b_src, dtype=np.int16),
        "b2b_dst": np.asarray(b2b_dst, dtype=np.int16),
        "b2b_weight": np.asarray(b2b_weight, dtype=np.float32),
        "mib_pair_src": np.asarray(mib_src, dtype=np.int16),
        "mib_pair_dst": np.asarray(mib_dst, dtype=np.int16),
        "cluster_pair_src": np.asarray(cluster_src, dtype=np.int16),
        "cluster_pair_dst": np.asarray(cluster_dst, dtype=np.int16),
        "fixed_mask": np.asarray(fixed, dtype=np.uint8),
        "preplaced_mask": np.asarray(preplaced, dtype=np.uint8),
        "boundary_mask": np.asarray(boundary, dtype=np.uint8),
        "dimensions": np.asarray(labels.dimensions, dtype=np.float32),
        "shape_options": np.asarray(shape_options, dtype=np.float32).reshape(-1, 2),
        "selected_shape": np.asarray(labels.selected_shape_indices, dtype=np.int16),
        "shape_supervision_mask": np.asarray(shape_supervision_mask, dtype=np.uint8),
        "mib_consistent_mask": np.asarray(mib_consistent_mask, dtype=np.uint8),
        "horizontal_parent": np.asarray(horizontal_parent, dtype=np.int16),
        "horizontal_side": np.asarray(horizontal_side, dtype=np.int8),
        "vertical_support": np.asarray(vertical_support, dtype=np.int16),
        "golden_rectangles": np.asarray(golden, dtype=np.float32),
        "root": np.asarray(labels.root, dtype=np.int16),
        "strict_mib_decodable": np.asarray(strict_mib_decodable, dtype=np.uint8),
        "mib_input_compatible": np.asarray(
            mib_metadata["input_compatible"], dtype=np.uint8
        ),
        "mib_features_masked": np.asarray(mib_metadata["masked"], dtype=np.uint8),
        "shape_counts": np.asarray(shape_counts, dtype=np.int64),
    }
    metadata = {
        "block_count": block_count,
        "b2b_edge_count": len(b2b_src),
        "mib_pair_count": len(mib_src),
        "cluster_pair_count": len(cluster_src),
        "shape_option_count": len(shape_options),
        "mib_inconsistent_groups": sorted(labels.mib_inconsistent_groups),
        "mib_inconsistent_block_count": len(inconsistent_members),
        "strict_mib_decodable": strict_mib_decodable,
        "mib_input_compatible": bool(mib_metadata["input_compatible"]),
        "mib_features_masked": bool(mib_metadata["masked"]),
        "oracle_max_coordinate_delta": comparison.max_coordinate_delta,
        "oracle_max_dimension_delta": comparison.max_dimension_delta,
        "input_sha256": _hash_input(sample),
        "optimizer_target_sha256": _hash_hard_targets(hard_targets),
        "tree_sha256": _hash_tensor("tree", tree),
        "golden_geometry_sha256": _hash_tensor("fp_solution", fp_solution),
        "golden_metrics_sha256": _hash_tensor("metrics", metrics),
    }
    return arrays, metadata


def _concatenate(
    layouts: list[dict[str, np.ndarray]], name: str, shape: tuple[int, ...]
) -> np.ndarray:
    values = [layout[name] for layout in layouts]
    if values and any(value.size for value in values):
        return np.concatenate(values, axis=0).astype(ARRAY_DTYPES[name], copy=False)
    return np.empty(shape, dtype=ARRAY_DTYPES[name])


def _pack_shard(layouts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not layouts:
        raise CacheError("empty_shard")
    layout_ptr = [0]
    edge_ptr = [0]
    mib_pair_ptr = [0]
    cluster_pair_ptr = [0]
    shape_ptr = [0]
    for layout in layouts:
        node_count = len(layout["area_targets"])
        layout_ptr.append(layout_ptr[-1] + node_count)
        edge_ptr.append(edge_ptr[-1] + len(layout["b2b_src"]))
        mib_pair_ptr.append(mib_pair_ptr[-1] + len(layout["mib_pair_src"]))
        cluster_pair_ptr.append(
            cluster_pair_ptr[-1] + len(layout["cluster_pair_src"])
        )
        for count in layout.pop("shape_counts").tolist():
            shape_ptr.append(shape_ptr[-1] + int(count))
    packed: dict[str, np.ndarray] = {
        "layout_ptr": np.asarray(layout_ptr, dtype=np.int64),
        "edge_ptr": np.asarray(edge_ptr, dtype=np.int64),
        "mib_pair_ptr": np.asarray(mib_pair_ptr, dtype=np.int64),
        "cluster_pair_ptr": np.asarray(cluster_pair_ptr, dtype=np.int64),
        "shape_ptr": np.asarray(shape_ptr, dtype=np.int64),
    }
    node_vectors = (
        "area_targets",
        "fixed_mask",
        "preplaced_mask",
        "selected_shape",
        "shape_supervision_mask",
        "mib_consistent_mask",
        "horizontal_parent",
        "horizontal_side",
        "vertical_support",
    )
    for name in node_vectors:
        packed[name] = _concatenate(layouts, name, (0,))
    for name, width in (
        ("node_features", len(FEATURE_NAMES)),
        ("hard_targets", 4),
        ("boundary_mask", 4),
        ("dimensions", 2),
        ("golden_rectangles", 4),
        ("shape_options", 2),
    ):
        packed[name] = _concatenate(layouts, name, (0, width))
    for name in (
        "b2b_src",
        "b2b_dst",
        "b2b_weight",
        "mib_pair_src",
        "mib_pair_dst",
        "cluster_pair_src",
        "cluster_pair_dst",
    ):
        packed[name] = _concatenate(layouts, name, (0,))
    for name in (
        "root",
        "strict_mib_decodable",
        "mib_input_compatible",
        "mib_features_masked",
    ):
        packed[name] = np.asarray(
            [layout[name].item() for layout in layouts], dtype=ARRAY_DTYPES[name]
        )
    if set(packed) != set(ALL_ARRAYS):
        raise CacheError(
            "internal_array_contract_mismatch",
            repr(sorted(set(packed) ^ set(ALL_ARRAYS))),
        )
    return packed


def _write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write an allow_pickle=False-compatible NPZ with fixed ZIP metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for name in sorted(arrays):
                array = np.ascontiguousarray(arrays[name])
                if array.dtype.kind in "OUSV":
                    raise CacheError("unsafe_array_dtype", f"{name}: {array.dtype}")
                payload = io.BytesIO()
                np.lib.format.write_array(payload, array, allow_pickle=False)
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, payload.getvalue(), compresslevel=6)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _array_contract() -> dict[str, Any]:
    return {
        "model_input_arrays": list(MODEL_INPUT_ARRAYS),
        "supervision_arrays": list(SUPERVISION_ARRAYS),
        "dtypes": dict(sorted(ARRAY_DTYPES.items())),
        "feature_schema": {
            "version": FEATURE_VERSION,
            "names": list(FEATURE_NAMES),
        },
        "identity_policy": {
            "identity_location": "manifest_records_only",
            "prohibited_shard_key_fragments": list(IDENTITY_KEY_FRAGMENTS),
            "raw_group_identifiers_stored": False,
        },
        "sentinels": {
            "horizontal_parent_root": -1,
            "horizontal_side_root": -1,
            "vertical_support_floor": -1,
        },
        "golden_rectangle_order": ["x", "y", "width", "height"],
        "boundary_mask_order": ["left", "right", "top", "bottom"],
    }


def _validate_pointer(name: str, pointer: np.ndarray, expected_end: int) -> None:
    if pointer.ndim != 1 or len(pointer) < 1 or pointer[0] != 0:
        raise CacheError("invalid_pointer_array", name)
    if np.any(pointer[1:] < pointer[:-1]) or int(pointer[-1]) != expected_end:
        raise CacheError("invalid_pointer_array", name)


def _require_binary(name: str, value: np.ndarray) -> None:
    if np.any((value != 0) & (value != 1)):
        raise CacheError("nonbinary_mask", name)


def _validate_layout_semantics(
    arrays: dict[str, np.ndarray], layout_index: int, shard_name: str
) -> dict[str, Any]:
    """Validate one packed layout without consulting manifest identities."""
    start = int(arrays["layout_ptr"][layout_index])
    end = int(arrays["layout_ptr"][layout_index + 1])
    block_count = end - start
    if block_count < 1:
        raise CacheError("empty_packed_layout", f"{shard_name}#{layout_index}")
    node = slice(start, end)
    root = int(arrays["root"][layout_index])
    if not 0 <= root < block_count:
        raise CacheError("root_out_of_range", f"{shard_name}#{layout_index}")

    for name in (
        "fixed_mask",
        "preplaced_mask",
        "boundary_mask",
        "shape_supervision_mask",
        "mib_consistent_mask",
    ):
        _require_binary(name, arrays[name][node])
    for name in (
        "strict_mib_decodable",
        "mib_input_compatible",
        "mib_features_masked",
    ):
        _require_binary(name, arrays[name][layout_index : layout_index + 1])

    dimensions = arrays["dimensions"][node]
    golden = arrays["golden_rectangles"][node]
    if np.any(dimensions <= 0.0) or np.any(arrays["area_targets"][node] <= 0.0):
        raise CacheError("nonpositive_geometry", f"{shard_name}#{layout_index}")
    if not np.allclose(golden[:, 2:4], dimensions, rtol=0.0, atol=1e-6):
        raise CacheError("golden_dimension_mismatch", f"{shard_name}#{layout_index}")
    for local_block, global_block in enumerate(range(start, end)):
        option_start = int(arrays["shape_ptr"][global_block])
        option_end = int(arrays["shape_ptr"][global_block + 1])
        selected = int(arrays["selected_shape"][global_block])
        if not 0 <= selected < option_end - option_start:
            raise CacheError("selected_shape_out_of_range", str(global_block))
        selected_dimensions = arrays["shape_options"][option_start + selected]
        if not np.allclose(
            selected_dimensions, dimensions[local_block], rtol=0.0, atol=1e-6
        ):
            raise CacheError("selected_shape_dimension_mismatch", str(global_block))

    fixed = arrays["fixed_mask"][node].astype(bool)
    preplaced = arrays["preplaced_mask"][node].astype(bool)
    hard = arrays["hard_targets"][node]
    hard_blocks = fixed | preplaced
    if np.any(hard_blocks) and not np.allclose(
        hard[hard_blocks, 2:4], dimensions[hard_blocks], rtol=0.0, atol=1e-6
    ):
        raise CacheError("hard_target_dimension_mismatch", f"{shard_name}#{layout_index}")
    if np.any(preplaced) and not np.allclose(
        hard[preplaced, 0:2], golden[preplaced, 0:2], rtol=0.0, atol=1e-6
    ):
        raise CacheError("preplaced_target_mismatch", f"{shard_name}#{layout_index}")
    free = ~hard_blocks
    if np.any(free) and not np.all(hard[free] == -1.0):
        raise CacheError("free_block_hard_target", f"{shard_name}#{layout_index}")
    fixed_only = fixed & ~preplaced
    if np.any(fixed_only) and not np.all(hard[fixed_only, 0:2] == -1.0):
        raise CacheError("fixed_position_leakage", f"{shard_name}#{layout_index}")

    edge_start = int(arrays["edge_ptr"][layout_index])
    edge_end = int(arrays["edge_ptr"][layout_index + 1])
    edge_src = arrays["b2b_src"][edge_start:edge_end]
    edge_dst = arrays["b2b_dst"][edge_start:edge_end]
    if (
        np.any(edge_src < 0)
        or np.any(edge_src >= block_count)
        or np.any(edge_dst < 0)
        or np.any(edge_dst >= block_count)
        or np.any(edge_src == edge_dst)
        or np.any(arrays["b2b_weight"][edge_start:edge_end] <= 0.0)
    ):
        raise CacheError("invalid_packed_b2b_edge", f"{shard_name}#{layout_index}")

    horizontal_parent = arrays["horizontal_parent"][node].astype(np.int64)
    horizontal_side = arrays["horizontal_side"][node].astype(np.int64)
    if horizontal_parent[root] != -1 or horizontal_side[root] != -1:
        raise CacheError("invalid_root_sentinel", f"{shard_name}#{layout_index}")
    non_root = np.arange(block_count) != root
    if (
        np.any(horizontal_parent[non_root] < 0)
        or np.any(horizontal_parent[non_root] >= block_count)
        or np.any(horizontal_parent[non_root] == np.arange(block_count)[non_root])
        or np.any((horizontal_side[non_root] != 0) & (horizontal_side[non_root] != 1))
    ):
        raise CacheError("invalid_horizontal_parent", f"{shard_name}#{layout_index}")
    occupied_slots: set[tuple[int, int]] = set()
    for block in range(block_count):
        if block == root:
            continue
        slot = (int(horizontal_parent[block]), int(horizontal_side[block]))
        if slot in occupied_slots:
            raise CacheError("duplicate_horizontal_slot", f"{shard_name}#{layout_index}")
        occupied_slots.add(slot)

    x_values: list[float | None] = [None] * block_count
    x_visiting: set[int] = set()

    def resolve_x(block: int) -> float:
        known = x_values[block]
        if known is not None:
            return known
        if block in x_visiting:
            raise CacheError("horizontal_cycle", f"{shard_name}#{layout_index}")
        x_visiting.add(block)
        if block == root:
            value = 0.0
        else:
            parent = int(horizontal_parent[block])
            value = resolve_x(parent) + (
                float(dimensions[parent, 0]) if horizontal_side[block] == 0 else 0.0
            )
        x_visiting.remove(block)
        x_values[block] = value
        return value

    for block in range(block_count):
        resolve_x(block)

    support = arrays["vertical_support"][node].astype(np.int64)
    if np.any(support < -1) or np.any(support >= block_count):
        raise CacheError("invalid_vertical_support", f"{shard_name}#{layout_index}")
    if any(int(support[block]) == block for block in range(block_count)):
        raise CacheError("vertical_self_support", f"{shard_name}#{layout_index}")
    y_values: list[float | None] = [None] * block_count
    y_visiting: set[int] = set()

    def resolve_y(block: int) -> float:
        known = y_values[block]
        if known is not None:
            return known
        if block in y_visiting:
            raise CacheError("vertical_support_cycle", f"{shard_name}#{layout_index}")
        y_visiting.add(block)
        parent = int(support[block])
        value = 0.0 if parent == -1 else resolve_y(parent) + float(dimensions[parent, 1])
        y_visiting.remove(block)
        y_values[block] = value
        return value

    for block in range(block_count):
        resolve_y(block)
    decoded_xy = np.asarray(list(zip(x_values, y_values)), dtype=np.float64)
    translation = golden[0, 0:2].astype(np.float64) - decoded_xy[0]
    if not np.allclose(
        decoded_xy + translation, golden[:, 0:2], rtol=0.0, atol=1e-6
    ):
        raise CacheError("packed_oracle_roundtrip_mismatch", f"{shard_name}#{layout_index}")

    mib_start = int(arrays["mib_pair_ptr"][layout_index])
    mib_end = int(arrays["mib_pair_ptr"][layout_index + 1])
    mib_src = arrays["mib_pair_src"][mib_start:mib_end].astype(np.int64)
    mib_dst = arrays["mib_pair_dst"][mib_start:mib_end].astype(np.int64)
    if (
        np.any(mib_src < 0)
        or np.any(mib_src >= block_count)
        or np.any(mib_dst < 0)
        or np.any(mib_dst >= block_count)
        or np.any(mib_src >= mib_dst)
        or len(set(zip(mib_src.tolist(), mib_dst.tolist()))) != len(mib_src)
    ):
        raise CacheError("invalid_mib_pair", f"{shard_name}#{layout_index}")
    parent = list(range(block_count))

    def find(block: int) -> int:
        while parent[block] != block:
            parent[block] = parent[parent[block]]
            block = parent[block]
        return block

    for left, right in zip(mib_src.tolist(), mib_dst.tolist()):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root
    components: dict[int, list[int]] = {}
    paired_members = set(mib_src.tolist()) | set(mib_dst.tolist())
    for block in paired_members:
        components.setdefault(find(block), []).append(block)
    inconsistent_members: set[int] = set()
    inconsistent_group_count = 0
    for members in components.values():
        first_dimensions = dimensions[members[0]]
        if any(
            not np.allclose(dimensions[block], first_dimensions, rtol=0.0, atol=1e-6)
            for block in members[1:]
        ):
            inconsistent_group_count += 1
            inconsistent_members.update(members)
    expected_mib_consistent = np.ones(block_count, dtype=np.uint8)
    if inconsistent_members:
        expected_mib_consistent[list(inconsistent_members)] = 0
    if not np.array_equal(arrays["mib_consistent_mask"][node], expected_mib_consistent):
        raise CacheError("mib_consistency_mask_mismatch", f"{shard_name}#{layout_index}")
    expected_shape_mask = ((~hard_blocks) & expected_mib_consistent.astype(bool)).astype(
        np.uint8
    )
    if not np.array_equal(arrays["shape_supervision_mask"][node], expected_shape_mask):
        raise CacheError("shape_supervision_mask_mismatch", f"{shard_name}#{layout_index}")
    strict_expected = int(not inconsistent_members)
    if int(arrays["strict_mib_decodable"][layout_index]) != strict_expected:
        raise CacheError("strict_mib_flag_mismatch", f"{shard_name}#{layout_index}")

    cluster_start = int(arrays["cluster_pair_ptr"][layout_index])
    cluster_end = int(arrays["cluster_pair_ptr"][layout_index + 1])
    cluster_src = arrays["cluster_pair_src"][cluster_start:cluster_end]
    cluster_dst = arrays["cluster_pair_dst"][cluster_start:cluster_end]
    if (
        np.any(cluster_src < 0)
        or np.any(cluster_src >= block_count)
        or np.any(cluster_dst < 0)
        or np.any(cluster_dst >= block_count)
        or np.any(cluster_src >= cluster_dst)
        or len(set(zip(cluster_src.tolist(), cluster_dst.tolist()))) != len(cluster_src)
    ):
        raise CacheError("invalid_cluster_pair", f"{shard_name}#{layout_index}")
    return {
        "block_count": block_count,
        "b2b_edge_count": edge_end - edge_start,
        "mib_pair_count": mib_end - mib_start,
        "cluster_pair_count": cluster_end - cluster_start,
        "shape_option_count": int(arrays["shape_ptr"][end] - arrays["shape_ptr"][start]),
        "mib_inconsistent_group_count": inconsistent_group_count,
        "mib_inconsistent_block_count": len(inconsistent_members),
        "strict_mib_decodable": bool(arrays["strict_mib_decodable"][layout_index]),
        "mib_input_compatible": bool(arrays["mib_input_compatible"][layout_index]),
        "mib_features_masked": bool(arrays["mib_features_masked"][layout_index]),
    }


def _validate_shard_arrays(
    arrays: dict[str, np.ndarray], descriptor: dict[str, Any]
) -> list[dict[str, Any]]:
    if set(arrays) != set(ALL_ARRAYS):
        raise CacheError("shard_array_names", descriptor.get("path", ""))
    for name, array in arrays.items():
        if str(array.dtype) != ARRAY_DTYPES[name]:
            raise CacheError(
                "shard_array_dtype", f"{name}: {array.dtype} != {ARRAY_DTYPES[name]}"
            )
        if array.dtype.kind in "OUSV":
            raise CacheError("unsafe_array_dtype", name)
        lowered = name.lower()
        if any(fragment in lowered for fragment in IDENTITY_KEY_FRAGMENTS):
            raise CacheError("identity_array_forbidden", name)
        if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
            raise CacheError("nonfinite_shard_array", name)
    layout_count = int(descriptor["layout_count"])
    node_count = int(descriptor["node_count"])
    _validate_pointer("layout_ptr", arrays["layout_ptr"], node_count)
    if len(arrays["layout_ptr"]) != layout_count + 1:
        raise CacheError("layout_pointer_count")
    _validate_pointer("edge_ptr", arrays["edge_ptr"], len(arrays["b2b_src"]))
    _validate_pointer(
        "mib_pair_ptr", arrays["mib_pair_ptr"], len(arrays["mib_pair_src"])
    )
    _validate_pointer(
        "cluster_pair_ptr",
        arrays["cluster_pair_ptr"],
        len(arrays["cluster_pair_src"]),
    )
    for pointer_name in ("edge_ptr", "mib_pair_ptr", "cluster_pair_ptr"):
        if len(arrays[pointer_name]) != layout_count + 1:
            raise CacheError("per_layout_pointer_count", pointer_name)
    _validate_pointer("shape_ptr", arrays["shape_ptr"], len(arrays["shape_options"]))
    if len(arrays["shape_ptr"]) != node_count + 1:
        raise CacheError("shape_pointer_count")
    node_vectors = (
        "area_targets",
        "fixed_mask",
        "preplaced_mask",
        "selected_shape",
        "shape_supervision_mask",
        "mib_consistent_mask",
        "horizontal_parent",
        "horizontal_side",
        "vertical_support",
    )
    if any(arrays[name].shape != (node_count,) for name in node_vectors):
        raise CacheError("node_vector_shape")
    expected_matrices = {
        "node_features": (node_count, len(FEATURE_NAMES)),
        "hard_targets": (node_count, 4),
        "boundary_mask": (node_count, 4),
        "dimensions": (node_count, 2),
        "golden_rectangles": (node_count, 4),
        "shape_options": (len(arrays["shape_options"]), 2),
    }
    for name, shape in expected_matrices.items():
        if arrays[name].shape != shape:
            raise CacheError("matrix_shape", f"{name}: {arrays[name].shape}")
    for name in (
        "root",
        "strict_mib_decodable",
        "mib_input_compatible",
        "mib_features_masked",
    ):
        if arrays[name].shape != (layout_count,):
            raise CacheError("layout_vector_shape", name)
    paired = (
        ("b2b_src", "b2b_dst", "b2b_weight"),
        ("mib_pair_src", "mib_pair_dst"),
        ("cluster_pair_src", "cluster_pair_dst"),
    )
    for names in paired:
        if len({len(arrays[name]) for name in names}) != 1:
            raise CacheError("edge_array_length", "/".join(names))
    descriptor_counts = {
        "b2b_edge_count": len(arrays["b2b_src"]),
        "mib_pair_count": len(arrays["mib_pair_src"]),
        "cluster_pair_count": len(arrays["cluster_pair_src"]),
        "shape_option_count": len(arrays["shape_options"]),
    }
    for key, observed in descriptor_counts.items():
        if descriptor.get(key) != observed:
            raise CacheError("shard_descriptor_count", f"{key}: {descriptor.get(key)}")
    summaries = []
    for layout_index in range(layout_count):
        start = int(arrays["layout_ptr"][layout_index])
        end = int(arrays["layout_ptr"][layout_index + 1])
        count = end - start
        if not 0 <= int(arrays["root"][layout_index]) < count:
            raise CacheError("root_out_of_range", str(layout_index))
        selected = arrays["selected_shape"][start:end]
        shape_start = arrays["shape_ptr"][start:end]
        shape_end = arrays["shape_ptr"][start + 1 : end + 1]
        if np.any(selected < 0) or np.any(selected >= shape_end - shape_start):
            raise CacheError("selected_shape_out_of_range", str(layout_index))
        for pointer_name, source_name, destination_name in (
            ("edge_ptr", "b2b_src", "b2b_dst"),
            ("mib_pair_ptr", "mib_pair_src", "mib_pair_dst"),
            ("cluster_pair_ptr", "cluster_pair_src", "cluster_pair_dst"),
        ):
            edge_start = int(arrays[pointer_name][layout_index])
            edge_end = int(arrays[pointer_name][layout_index + 1])
            for array_name in (source_name, destination_name):
                indices = arrays[array_name][edge_start:edge_end]
                if np.any(indices < 0) or np.any(indices >= count):
                    raise CacheError("edge_index_out_of_range", array_name)
        summaries.append(
            _validate_layout_semantics(
                arrays, layout_index, str(descriptor.get("path", "shard"))
            )
        )
    return summaries


def validate_cache(
    cache_dir: Path,
    *,
    data_root: Path | None = None,
    holdout_paths: Iterable[Path] | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    cache_dir = cache_dir.resolve()
    manifest_path = cache_dir / MANIFEST_NAME
    if manifest_path.is_symlink():
        raise CacheError("manifest_symlink_forbidden")
    observed_manifest_sha256 = _file_sha256(manifest_path)
    if expected_manifest_sha256 is not None and not _is_sha256(
        expected_manifest_sha256
    ):
        raise CacheError("invalid_expected_manifest_sha256")
    if (
        expected_manifest_sha256 is not None
        and observed_manifest_sha256 != expected_manifest_sha256
    ):
        raise CacheError("manifest_sha256_mismatch")
    manifest = _load_json(manifest_path)
    if set(manifest) != MANIFEST_KEYS:
        raise CacheError("manifest_keys", repr(sorted(set(manifest) ^ MANIFEST_KEYS)))
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise CacheError("cache_schema_version")
    if manifest.get("builder_version") != BUILDER_VERSION:
        raise CacheError("builder_version")
    if manifest.get("mode") != "dual_parent_supervision_cache":
        raise CacheError("cache_mode")
    if manifest.get("array_contract") != _array_contract():
        raise CacheError("array_contract_mismatch")
    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict):
        raise CacheError("invalid_configuration")
    layouts_per_source = configuration.get("layouts_per_source")
    if (
        isinstance(layouts_per_source, bool)
        or not isinstance(layouts_per_source, int)
        or layouts_per_source < 1
    ):
        raise CacheError("invalid_layouts_per_source")
    mib_policy = configuration.get("mib_feature_policy")
    if mib_policy not in {"unmasked", "mask_incompatible", "mask_all"}:
        raise CacheError("invalid_mib_feature_policy")

    shards = manifest.get("shards")
    records = manifest.get("records")
    if not isinstance(shards, list) or not shards:
        raise CacheError("missing_shards")
    if not isinstance(records, list) or not records:
        raise CacheError("missing_records")
    seen_paths: set[str] = set()
    shard_layout_counts: dict[str, int] = {}
    shard_partitions: dict[str, str] = {}
    shard_summaries: dict[str, list[dict[str, Any]]] = {}
    aggregate_descriptor = Counter()
    for descriptor in shards:
        if not isinstance(descriptor, dict) or set(descriptor) != SHARD_DESCRIPTOR_KEYS:
            raise CacheError("invalid_shard_descriptor")
        relative = _canonical_path(descriptor.get("path", ""))
        if PurePosixPath(relative).parts[0] != "shards":
            raise CacheError("shard_outside_shard_directory", relative)
        if relative in seen_paths:
            raise CacheError("duplicate_shard_path", relative)
        seen_paths.add(relative)
        partition = descriptor.get("partition")
        if partition not in PARTITION_NAMES:
            raise CacheError("shard_partition", repr(partition))
        if not _is_sha256(descriptor.get("sha256")):
            raise CacheError("invalid_shard_sha256", relative)
        for name in (
            "size_bytes",
            "layout_count",
            "node_count",
            "b2b_edge_count",
            "mib_pair_count",
            "cluster_pair_count",
            "shape_option_count",
        ):
            value = descriptor.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CacheError("invalid_shard_descriptor_count", f"{relative}: {name}")
        if descriptor["layout_count"] < 1 or descriptor["node_count"] < 1:
            raise CacheError("empty_shard_descriptor", relative)
        shard_path = cache_dir / relative
        try:
            if shard_path.is_symlink():
                raise CacheError("shard_symlink_forbidden", relative)
            if not shard_path.is_file() or shard_path.stat().st_size != descriptor["size_bytes"]:
                raise CacheError("shard_size_mismatch", relative)
            if _file_sha256(shard_path) != descriptor["sha256"]:
                raise CacheError("shard_sha256_mismatch", relative)
            with np.load(shard_path, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
        except CacheError:
            raise
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            raise CacheError("invalid_npz_shard", f"{relative}: {exc}") from exc
        summaries = _validate_shard_arrays(arrays, descriptor)
        shard_layout_counts[relative] = descriptor["layout_count"]
        shard_partitions[relative] = partition
        shard_summaries[relative] = summaries
        for name in (
            "layout_count",
            "node_count",
            "b2b_edge_count",
            "mib_pair_count",
            "cluster_pair_count",
            "shape_option_count",
        ):
            aggregate_descriptor[name] += descriptor[name]
    expected_files = {MANIFEST_NAME, *seen_paths}
    actual_files: set[str] = set()
    for path in cache_dir.rglob("*"):
        if path.is_symlink():
            raise CacheError("cache_symlink_forbidden", str(path.relative_to(cache_dir)))
        if path.is_file():
            actual_files.add(path.relative_to(cache_dir).as_posix())
    if actual_files != expected_files:
        raise CacheError("unexpected_cache_files", repr(sorted(actual_files ^ expected_files)))

    seen_layouts: set[tuple[str, int]] = set()
    seen_source_offsets: set[tuple[str, int]] = set()
    partition_sources: dict[str, set[str]] = {name: set() for name in PARTITION_NAMES}
    source_hashes: dict[str, tuple[str, int]] = {}
    source_block_counts: dict[str, int] = {}
    clean_offsets: Counter[str] = Counter()
    aggregate_records = Counter()
    hash_fields = (
        "source_sha256",
        "input_sha256",
        "optimizer_target_sha256",
        "tree_sha256",
        "golden_geometry_sha256",
        "golden_metrics_sha256",
    )
    summary_fields = (
        "block_count",
        "b2b_edge_count",
        "mib_pair_count",
        "cluster_pair_count",
        "shape_option_count",
        "mib_inconsistent_block_count",
        "strict_mib_decodable",
        "mib_input_compatible",
        "mib_features_masked",
    )
    for expected_index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != RECORD_KEYS:
            raise CacheError("record_keys", str(expected_index))
        if record.get("record_index") != expected_index:
            raise CacheError("record_index_sequence", str(expected_index))
        partition = record.get("partition")
        if partition not in PARTITION_NAMES:
            raise CacheError("record_partition", repr(partition))
        source = _canonical_path(record.get("source_file", ""))
        offset = record.get("file_offset")
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or not 0 <= offset < layouts_per_source
        ):
            raise CacheError("record_file_offset", repr(offset))
        identity = (source, offset)
        if identity in seen_source_offsets:
            raise CacheError("duplicate_source_offset", repr(identity))
        seen_source_offsets.add(identity)
        partition_sources[partition].add(source)
        if not isinstance(record.get("clean_offset_selected"), bool):
            raise CacheError("clean_offset_type", str(expected_index))
        clean_offsets[source] += int(record["clean_offset_selected"])
        if clean_offsets[source] > 1:
            raise CacheError("multiple_clean_offsets", source)
        for field in hash_fields:
            if not _is_sha256(record.get(field)):
                raise CacheError("invalid_record_sha256", f"{expected_index}: {field}")
        source_size = record.get("source_size_bytes")
        if isinstance(source_size, bool) or not isinstance(source_size, int) or source_size < 0:
            raise CacheError("invalid_source_size", source)
        shard = _canonical_path(record.get("shard", ""))
        local_layout = record.get("local_layout")
        if (
            shard not in shard_layout_counts
            or isinstance(local_layout, bool)
            or not isinstance(local_layout, int)
        ):
            raise CacheError("record_shard_reference", str(expected_index))
        if shard_partitions[shard] != partition:
            raise CacheError("record_shard_partition", str(expected_index))
        if not 0 <= local_layout < shard_layout_counts[shard]:
            raise CacheError("record_local_layout", str(expected_index))
        layout_key = (shard, local_layout)
        if layout_key in seen_layouts:
            raise CacheError("duplicate_shard_layout_reference", repr(layout_key))
        seen_layouts.add(layout_key)
        summary = shard_summaries[shard][local_layout]
        for field in summary_fields:
            if record.get(field) != summary[field]:
                raise CacheError("record_shard_metadata_mismatch", f"{expected_index}: {field}")
        group_ids = record.get("mib_inconsistent_groups")
        if (
            not isinstance(group_ids, list)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in group_ids
            )
            or group_ids != sorted(set(group_ids))
            or len(group_ids) != summary["mib_inconsistent_group_count"]
        ):
            raise CacheError("mib_inconsistent_groups", str(expected_index))
        for field in ("oracle_max_coordinate_delta", "oracle_max_dimension_delta"):
            value = record.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1e-6
            ):
                raise CacheError("oracle_delta", f"{expected_index}: {field}")
        source_fingerprint = (record["source_sha256"], source_size)
        previous = source_hashes.setdefault(source, source_fingerprint)
        if previous != source_fingerprint:
            raise CacheError("inconsistent_source_fingerprint", source)
        block_count = record["block_count"]
        previous_count = source_block_counts.setdefault(source, block_count)
        if previous_count != block_count:
            raise CacheError("inconsistent_source_block_count", source)
        for field in (
            "block_count",
            "b2b_edge_count",
            "mib_pair_count",
            "cluster_pair_count",
            "shape_option_count",
        ):
            aggregate_records[field] += record[field]
        aggregate_records["mib_inconsistent_layout_count"] += int(
            not record["strict_mib_decodable"]
        )
        aggregate_records["mib_inconsistent_block_count"] += record[
            "mib_inconsistent_block_count"
        ]
        aggregate_records["mib_input_incompatible_layout_count"] += int(
            not record["mib_input_compatible"]
        )
        aggregate_records["mib_features_masked_layout_count"] += int(
            record["mib_features_masked"]
        )
        expected_masked = {
            "unmasked": False,
            "mask_all": True,
            "mask_incompatible": not record["mib_input_compatible"],
        }[mib_policy]
        if record["mib_features_masked"] != expected_masked:
            raise CacheError("mib_feature_policy_mismatch", str(expected_index))

    if len(seen_layouts) != sum(shard_layout_counts.values()):
        raise CacheError("unreferenced_shard_layout")
    for index, first in enumerate(PARTITION_NAMES):
        for second in PARTITION_NAMES[index + 1 :]:
            overlap = partition_sources[first] & partition_sources[second]
            if overlap:
                raise CacheError("source_partition_overlap", min(overlap))
    if data_root is not None:
        root = data_root.resolve()
        for source, (expected_sha, expected_size) in sorted(source_hashes.items()):
            path = root / source
            if not path.is_file() or path.stat().st_size != expected_size:
                raise CacheError("source_size_mismatch", source)
            if _file_sha256(path) != expected_sha:
                raise CacheError("source_sha256_mismatch", source)

    partition_metadata = manifest.get("source_partition")
    if not isinstance(partition_metadata, dict):
        raise CacheError("invalid_source_partition_metadata")
    reconstructed_rows = {
        partition: [
            {"source_file": source, "block_count": source_block_counts[source]}
            for source in sorted(partition_sources[partition])
        ]
        for partition in PARTITION_NAMES
    }
    if partition_metadata.get("partition_sha256") != _canonical_json_sha256(
        reconstructed_rows
    ):
        raise CacheError("source_partition_sha256_mismatch")
    reconstructed_counts = {
        partition: len(partition_sources[partition]) for partition in PARTITION_NAMES
    }
    if partition_metadata.get("partition_source_counts") != reconstructed_counts:
        raise CacheError("partition_source_counts")
    reconstructed_blocks = {
        partition: {
            str(key): value
            for key, value in sorted(
                Counter(
                    source_block_counts[source]
                    for source in partition_sources[partition]
                ).items()
            )
        }
        for partition in PARTITION_NAMES
    }
    if partition_metadata.get("partition_block_counts") != reconstructed_blocks:
        raise CacheError("partition_block_counts")
    if not partition_metadata.get("partial_partitions_allowed") and any(
        not partition_sources[partition] for partition in PARTITION_NAMES
    ):
        raise CacheError("empty_source_partition")
    selection = partition_metadata.get("selection", {})
    if selection.get("selected_after_limit") != len(source_hashes):
        raise CacheError("selected_source_count")

    holdout_metadata = partition_metadata.get("holdout")
    if (
        not isinstance(holdout_metadata, dict)
        or holdout_metadata.get("split_unit") != "source_file"
    ):
        raise CacheError("invalid_holdout_metadata")
    manifest_hashes = holdout_metadata.get("manifest_sha256s")
    if (
        not isinstance(manifest_hashes, list)
        or not manifest_hashes
        or any(not _is_sha256(value) for value in manifest_hashes)
        or not _is_sha256(holdout_metadata.get("manifest_union_sha256"))
    ):
        raise CacheError("invalid_holdout_hashes")
    if holdout_paths is not None:
        verified_holdout = load_holdout_sources([Path(path) for path in holdout_paths])
        if list(verified_holdout.manifest_sha256s) != manifest_hashes:
            raise CacheError("holdout_manifest_sha256_mismatch")
        if verified_holdout.manifest_sha256 != holdout_metadata["manifest_union_sha256"]:
            raise CacheError("holdout_union_sha256_mismatch")
        if len(verified_holdout.paths) != holdout_metadata.get("excluded_source_count"):
            raise CacheError("holdout_source_count_mismatch")
        leaked = set(source_hashes) & verified_holdout.paths
        if leaked:
            raise CacheError("holdout_leakage", min(leaked))

    expected_record_sha = manifest.get("record_provenance_sha256")
    if (
        not _is_sha256(expected_record_sha)
        or expected_record_sha != _canonical_json_sha256(records)
    ):
        raise CacheError("record_provenance_sha256_mismatch")
    stats = manifest.get("stats")
    expected_stats = {
        "layout_count": len(records),
        "node_count": aggregate_records["block_count"],
        "b2b_edge_count": aggregate_records["b2b_edge_count"],
        "mib_pair_count": aggregate_records["mib_pair_count"],
        "cluster_pair_count": aggregate_records["cluster_pair_count"],
        "shape_option_count": aggregate_records["shape_option_count"],
        "mib_inconsistent_layout_count": aggregate_records[
            "mib_inconsistent_layout_count"
        ],
        "mib_inconsistent_block_count": aggregate_records[
            "mib_inconsistent_block_count"
        ],
        "mib_input_incompatible_layout_count": aggregate_records[
            "mib_input_incompatible_layout_count"
        ],
        "mib_features_masked_layout_count": aggregate_records[
            "mib_features_masked_layout_count"
        ],
    }
    if stats != expected_stats:
        raise CacheError("manifest_stats_mismatch")
    for name in (
        "layout_count",
        "node_count",
        "b2b_edge_count",
        "mib_pair_count",
        "cluster_pair_count",
        "shape_option_count",
    ):
        expected = len(records) if name == "layout_count" else aggregate_records[
            "block_count" if name == "node_count" else name
        ]
        if aggregate_descriptor[name] != expected:
            raise CacheError("aggregate_shard_descriptor_mismatch", name)
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "shard_count": len(shards),
        "layout_count": len(records),
        "source_count": len(source_hashes),
        "manifest_sha256": observed_manifest_sha256,
    }


def build_cache(
    dataset: Any,
    *,
    data_root: Path,
    output_dir: Path,
    holdout_paths: list[Path],
    seed: int = 20260710,
    min_blocks: int = 100,
    max_blocks: int = 120,
    max_sources: int | None = None,
    max_layouts_per_source: int | None = 8,
    shard_layouts: int = 64,
    message_steps: int = 4,
    mib_feature_policy: str = "mask_incompatible",
    layout_selection: str = "clean_plus_hash_raw",
    source_index_cache: Path | None = None,
    progress_every_sources: int = 0,
    allow_partial_partitions: bool = False,
    provenance_context: dict[str, Any] | None = None,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    data_root = data_root.resolve()
    output_dir = output_dir.resolve()
    holdout_paths = [Path(path).resolve() for path in holdout_paths]
    if output_dir.exists():
        raise CacheError("output_exists", str(output_dir))
    if shard_layouts < 1:
        raise CacheError("invalid_shard_layout_count")
    if max_layouts_per_source is not None and max_layouts_per_source < 1:
        raise CacheError("invalid_max_layouts_per_source")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        provenance = provenance_context or _provenance_context(
            holdout_paths, allow_dirty=allow_dirty
        )
        partitions, partition_provenance = _partition_sources(
            dataset,
            data_root=data_root,
            holdout_paths=holdout_paths,
            seed=seed,
            max_sources=max_sources,
            min_blocks=min_blocks,
            max_blocks=max_blocks,
            source_index_cache=source_index_cache,
            progress_every_sources=progress_every_sources,
            allow_partial_partitions=allow_partial_partitions,
        )
        source_fingerprints: dict[str, tuple[str, int]] = {}
        records: list[dict[str, Any]] = []
        shard_descriptors: list[dict[str, Any]] = []
        stats = Counter()
        layouts_per_file = int(dataset.layouts_per_file)
        global_shard_index = 0

        for partition in PARTITION_NAMES:
            pending: list[dict[str, np.ndarray]] = []
            pending_records: list[dict[str, Any]] = []

            def flush() -> None:
                nonlocal global_shard_index
                if not pending:
                    return
                relative = f"shards/{partition}-{global_shard_index:05d}.npz"
                arrays = _pack_shard(pending)
                path = staging / relative
                _write_deterministic_npz(path, arrays)
                descriptor = {
                    "path": relative,
                    "partition": partition,
                    "sha256": _file_sha256(path),
                    "size_bytes": path.stat().st_size,
                    "layout_count": len(pending),
                    "node_count": len(arrays["area_targets"]),
                    "b2b_edge_count": len(arrays["b2b_src"]),
                    "mib_pair_count": len(arrays["mib_pair_src"]),
                    "cluster_pair_count": len(arrays["cluster_pair_src"]),
                    "shape_option_count": len(arrays["shape_options"]),
                }
                shard_descriptors.append(descriptor)
                for local_layout, record in enumerate(pending_records):
                    record["shard"] = relative
                    record["local_layout"] = local_layout
                    records.append(record)
                pending.clear()
                pending_records.clear()
                global_shard_index += 1

            for source_ordinal, source in enumerate(partitions[partition], 1):
                source_path = data_root / source.relative_path
                before = (_file_sha256(source_path), source_path.stat().st_size)
                previous = source_fingerprints.setdefault(source.relative_path, before)
                if previous != before:
                    raise CacheError("source_changed_during_build", source.relative_path)
                offsets, clean_offset = select_layout_offsets(
                    dataset,
                    source,
                    max_layouts_per_file=max_layouts_per_source,
                    layout_seed=seed,
                    layout_selection=layout_selection,
                )
                if not offsets:
                    raise CacheError("source_has_no_selected_layout", source.relative_path)
                base = source.file_index * layouts_per_file
                for offset in offsets:
                    try:
                        arrays, metadata = _extract_layout(
                            dataset[base + offset],
                            message_steps=message_steps,
                            mib_feature_policy=mib_feature_policy,
                        )
                    except (CacheError, DualParentError) as exc:
                        code = exc.code if hasattr(exc, "code") else type(exc).__name__
                        raise CacheError(
                            "selected_layout_rejected",
                            f"{source.relative_path}#{offset}: {code}: {exc}",
                        ) from exc
                    if metadata["block_count"] != source.block_count:
                        raise CacheError(
                            "source_block_count_changed",
                            f"{source.relative_path}#{offset}",
                        )
                    record = {
                        "record_index": len(records) + len(pending_records),
                        "partition": partition,
                        "source_file": source.relative_path,
                        "file_offset": offset,
                        "source_sha256": before[0],
                        "source_size_bytes": before[1],
                        "clean_offset_selected": offset == clean_offset,
                        **metadata,
                    }
                    pending.append(arrays)
                    pending_records.append(record)
                    stats["layout_count"] += 1
                    stats["node_count"] += metadata["block_count"]
                    stats["b2b_edge_count"] += metadata["b2b_edge_count"]
                    stats["mib_pair_count"] += metadata["mib_pair_count"]
                    stats["cluster_pair_count"] += metadata["cluster_pair_count"]
                    stats["shape_option_count"] += metadata["shape_option_count"]
                    stats["mib_inconsistent_layout_count"] += int(
                        not metadata["strict_mib_decodable"]
                    )
                    stats["mib_inconsistent_block_count"] += metadata[
                        "mib_inconsistent_block_count"
                    ]
                    stats["mib_input_incompatible_layout_count"] += int(
                        not metadata["mib_input_compatible"]
                    )
                    stats["mib_features_masked_layout_count"] += int(
                        metadata["mib_features_masked"]
                    )
                    if len(pending) >= shard_layouts:
                        flush()
                after = (_file_sha256(source_path), source_path.stat().st_size)
                if before != after:
                    raise CacheError("source_changed_during_build", source.relative_path)
                if progress_every_sources and source_ordinal % progress_every_sources == 0:
                    print(
                        f"{partition}: {source_ordinal}/{len(partitions[partition])} sources",
                        flush=True,
                    )
            flush()

        if not records or not shard_descriptors:
            raise CacheError("empty_cache")
        manifest = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "builder_version": BUILDER_VERSION,
            "mode": "dual_parent_supervision_cache",
            "array_contract": _array_contract(),
            "configuration": {
                "seed": seed,
                "block_count_range": [min_blocks, max_blocks],
                "layouts_per_source": layouts_per_file,
                "max_sources": max_sources,
                "max_layouts_per_source": max_layouts_per_source,
                "shard_layouts": shard_layouts,
                "message_steps": message_steps,
                "mib_feature_policy": mib_feature_policy,
                "layout_selection": layout_selection,
                "oracle_tolerance": 1e-6,
                "oracle_mib_policy": "strict_then_mask_inconsistent_shape_supervision",
            },
            "provenance": provenance,
            "source_partition": partition_provenance,
            "rejection_taxonomy": {
                "policy": "abort_on_first_selected_layout_error",
                "successful_build_counts": {},
                "known_masked_condition": "golden_mib_dimension_inconsistency",
            },
            "shards": shard_descriptors,
            "records": records,
            "record_provenance_sha256": _canonical_json_sha256(records),
            "stats": {
                key: stats[key]
                for key in (
                    "layout_count",
                    "node_count",
                    "b2b_edge_count",
                    "mib_pair_count",
                    "cluster_pair_count",
                    "shape_option_count",
                    "mib_inconsistent_layout_count",
                    "mib_inconsistent_block_count",
                    "mib_input_incompatible_layout_count",
                    "mib_features_masked_layout_count",
                )
            },
        }
        manifest_path = staging / MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        validation = validate_cache(
            staging, data_root=data_root, holdout_paths=holdout_paths
        )
        os.replace(staging, output_dir)
        validation["cache_dir"] = str(output_dir)
        return validation
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _build_command(args: argparse.Namespace) -> int:
    data_root = args.data_root.resolve()
    dataset = FloorplanDatasetLite(str(data_root))
    holdouts = args.holdout_manifest or list(DEFAULT_HOLDOUTS)
    report = build_cache(
        dataset,
        data_root=data_root,
        output_dir=args.output,
        holdout_paths=holdouts,
        seed=args.seed,
        min_blocks=args.min_blocks,
        max_blocks=args.max_blocks,
        max_sources=None if args.max_sources == 0 else args.max_sources,
        max_layouts_per_source=(
            None if args.max_layouts_per_source == 0 else args.max_layouts_per_source
        ),
        shard_layouts=args.shard_layouts,
        message_steps=args.message_steps,
        mib_feature_policy=args.mib_feature_policy,
        layout_selection=args.layout_selection,
        source_index_cache=args.source_index_cache,
        progress_every_sources=args.progress_every_sources,
        allow_partial_partitions=args.allow_partial_partitions,
        allow_dirty=args.allow_dirty,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _validate_command(args: argparse.Namespace) -> int:
    report = validate_cache(
        args.cache_dir,
        data_root=args.data_root,
        holdout_paths=args.holdout_manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build and atomically publish a cache")
    build.add_argument("--data-root", type=Path, default=FLOORSET)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--holdout-manifest", type=Path, action="append")
    build.add_argument("--seed", type=int, default=20260710)
    build.add_argument("--min-blocks", type=int, default=100)
    build.add_argument("--max-blocks", type=int, default=120)
    build.add_argument(
        "--max-sources", type=int, default=0, help="0 selects every eligible source"
    )
    build.add_argument("--max-layouts-per-source", type=int, default=8)
    build.add_argument("--shard-layouts", type=int, default=64)
    build.add_argument("--message-steps", type=int, default=4)
    build.add_argument(
        "--mib-feature-policy",
        choices=("unmasked", "mask_incompatible", "mask_all"),
        default="mask_incompatible",
    )
    build.add_argument(
        "--layout-selection",
        choices=("clean_plus_hash_raw", "hash_raw"),
        default="clean_plus_hash_raw",
    )
    build.add_argument("--source-index-cache", type=Path)
    build.add_argument("--progress-every-sources", type=int, default=0)
    build.add_argument(
        "--allow-partial-partitions",
        action="store_true",
        help="permit empty dev/cal partitions only for tiny smoke builds",
    )
    build.add_argument(
        "--allow-dirty",
        action="store_true",
        help="record a dirty tree instead of rejecting it (smoke/debug only)",
    )
    build.set_defaults(handler=_build_command)

    validate = subparsers.add_parser("validate", help="validate an existing cache")
    validate.add_argument("--cache-dir", type=Path, required=True)
    validate.add_argument(
        "--data-root",
        type=Path,
        help="also re-hash every referenced FloorSet source file",
    )
    validate.add_argument(
        "--holdout-manifest",
        type=Path,
        action="append",
        help="also re-verify configured holdout hashes and absence of leakage",
    )
    validate.add_argument(
        "--expected-manifest-sha256",
        help="require an externally recorded manifest digest",
    )
    validate.set_defaults(handler=_validate_command)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        return args.handler(args)
    except CacheError as exc:
        print(f"ERROR [{exc.code}] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
