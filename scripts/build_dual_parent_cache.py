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
import ctypes
import errno
import hashlib
import importlib.util
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
    enumerate_oriented_factor_shapes,
    extract_oracle_labels,
    hard_targets_from_golden,
    training_rectangles,
)
from learned_order import (  # noqa: E402
    FEATURE_NAMES,
    FEATURE_VERSION,
    apply_mib_feature_policy,
    extract_order_features,
    mib_is_input_compatible,
)
from scripts.train_order_model import (  # noqa: E402
    SourceFile,
    _block_count,
    load_holdout_sources,
    partition_source_files,
    select_layout_offsets,
)


CACHE_SCHEMA_VERSION = 3
BUILDER_VERSION = 3
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
CONFIGURATION_KEYS = frozenset(
    {
        "seed",
        "block_count_range",
        "layouts_per_source",
        "max_sources",
        "max_layouts_per_source",
        "shard_layouts",
        "message_steps",
        "mib_feature_policy",
        "layout_selection",
        "oracle_tolerance",
        "oracle_mib_policy",
    }
)
PROVENANCE_KEYS = frozenset({"repository", "floorset", "code_sha256", "runtime"})
REPOSITORY_PROVENANCE_KEYS = frozenset(
    {"commit", "tree", "dirty", "dirty_status_sha256"}
)
FLOORSET_PROVENANCE_KEYS = frozenset(
    {
        "repository",
        "commit",
        "tree",
        "data_root_mode",
        "loader_relative_path",
        "loader_sha256",
        "loader_module_name",
        "dataset_class_name",
        "official_sources_sha256",
    }
)
CODE_HASH_KEYS = frozenset(
    {
        "builder",
        "dual_parent_decoder",
        "learned_order_features",
        "source_partition_implementation",
        "floorset_lite_loader",
        "official_sources",
        "holdout_manifest_union",
    }
)
RUNTIME_PROVENANCE_KEYS = frozenset({"python", "numpy", "platform"})
SOURCE_PARTITION_KEYS = frozenset(
    {
        "algorithm",
        "partition_sha256",
        "partition_source_counts",
        "partition_block_counts",
        "selection",
        "source_index",
        "holdout",
        "partial_partitions_allowed",
    }
)
SELECTION_KEYS = frozenset(
    {
        "source_files_discovered",
        "source_files_excluded",
        "source_files_outside_block_range",
        "eligible_before_limit",
        "selected_after_limit",
        "eligible_by_block_count",
    }
)
SOURCE_INDEX_KEYS = frozenset(
    {"schema_version", "source_inventory_sha256", "payload_sha256"}
)
HOLDOUT_KEYS = frozenset(
    {
        "split_unit",
        "excluded_source_count",
        "manifest_union_sha256",
        "manifest_sha256s",
        "manifest_schema_versions",
    }
)
REJECTION_TAXONOMY_KEYS = frozenset(
    {"policy", "successful_build_counts", "known_masked_condition"}
)
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


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


def _is_git_oid(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) in (40, 64)
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_plain_int(value: Any, *, minimum: int | None = None) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int)
        and (minimum is None or value >= minimum)
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


def _load_holdouts_strict(paths: Iterable[Path]):
    normalized = [Path(path) for path in paths]
    for path in normalized:
        _load_json(path)
    return load_holdout_sources(normalized)


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


def _repository_status(*, ignored_roots: Iterable[Path] = ()) -> str:
    ignored = [Path(path).resolve() for path in ignored_roots]
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CacheError("git_provenance_failed", f"repository status: {exc}") from exc
    kept: list[bytes] = []
    entries = result.stdout.split(b"\0")
    index = 0
    while index < len(entries):
        entry = entries[index]
        index += 1
        if not entry:
            continue
        if len(entry) < 4:
            raise CacheError("malformed_git_status")
        status = entry[:2]
        raw_path = entry[3:]
        candidate = (ROOT / os.fsdecode(raw_path)).resolve()
        excluded = any(root == candidate or root in candidate.parents for root in ignored)
        if not excluded:
            kept.append(entry)
        if status[:1] in (b"R", b"C") or status[1:2] in (b"R", b"C"):
            if index >= len(entries) or not entries[index]:
                raise CacheError("malformed_git_status_rename")
            old_entry = entries[index]
            index += 1
            old_candidate = (ROOT / os.fsdecode(old_entry)).resolve()
            old_excluded = any(
                root == old_candidate or root in old_candidate.parents
                for root in ignored
            )
            if not excluded or not old_excluded:
                kept.append(old_entry)
    return b"\0".join(kept).decode("utf-8", errors="surrogateescape")


def _code_hashes(
    data_root: Path, holdout_paths: Iterable[Path]
) -> dict[str, str]:
    paths = {
        "builder": Path(__file__),
        "dual_parent_decoder": SOLUTION_DIR / "dual_parent_decoder.py",
        "learned_order_features": SOLUTION_DIR / "learned_order.py",
        "source_partition_implementation": ROOT / "scripts" / "train_order_model.py",
        "floorset_lite_loader": data_root / "lite_dataset.py",
        "official_sources": ROOT / "docs" / "official_sources.json",
    }
    hashes = {name: _file_sha256(path) for name, path in paths.items()}
    hashes["holdout_manifest_union"] = _canonical_json_sha256(
        [_file_sha256(Path(path)) for path in holdout_paths]
    )
    return dict(sorted(hashes.items()))


def _provenance_context(
    data_root: Path,
    holdout_paths: list[Path],
    *,
    allow_dirty: bool,
    ignored_roots: Iterable[Path] = (),
) -> dict[str, Any]:
    data_root = data_root.resolve()
    repository_status = _repository_status(ignored_roots=ignored_roots)
    if repository_status and not allow_dirty:
        preview = repository_status.splitlines()[0]
        raise CacheError("dirty_repository", preview)
    actual_toplevel = Path(
        _git_output(["rev-parse", "--show-toplevel"], data_root)
    ).resolve()
    if actual_toplevel != data_root:
        raise CacheError(
            "data_root_not_floorset_checkout",
            f"expected git top-level {data_root}, observed {actual_toplevel}",
        )
    floorset_commit = _git_output(["rev-parse", "HEAD"], data_root)
    floorset_tree = _git_output(["rev-parse", "HEAD^{tree}"], data_root)
    official = _load_json(ROOT / "docs" / "official_sources.json")
    pinned = official.get("floorset", {})
    if floorset_commit != pinned.get("commit") or floorset_tree != pinned.get("tree"):
        raise CacheError(
            "floorset_revision_mismatch",
            f"observed {floorset_commit}/{floorset_tree}",
        )
    loader_path = data_root / "lite_dataset.py"
    if not loader_path.is_file() or loader_path.is_symlink():
        raise CacheError("floorset_loader_missing", str(loader_path))
    if _git_output(["hash-object", str(loader_path)], data_root) != _git_output(
        ["rev-parse", "HEAD:lite_dataset.py"], data_root
    ):
        raise CacheError("floorset_loader_worktree_mismatch", str(loader_path))
    loader_sha256 = _file_sha256(loader_path)
    return {
        "repository": {
            "commit": _git_output(["rev-parse", "HEAD"], ROOT),
            "tree": _git_output(["rev-parse", "HEAD^{tree}"], ROOT),
            "dirty": bool(repository_status),
            "dirty_status_sha256": hashlib.sha256(
                repository_status.encode("utf-8", errors="surrogateescape")
            ).hexdigest(),
        },
        "floorset": {
            "repository": pinned.get("repository"),
            "commit": floorset_commit,
            "tree": floorset_tree,
            "data_root_mode": "pinned_git_checkout_root",
            "loader_relative_path": "lite_dataset.py",
            "loader_sha256": loader_sha256,
            "loader_module_name": f"_verified_floorset_lite_{loader_sha256[:16]}",
            "dataset_class_name": "FloorplanDatasetLite",
            "official_sources_sha256": _file_sha256(
                ROOT / "docs" / "official_sources.json"
            ),
        },
        "code_sha256": _code_hashes(data_root, holdout_paths),
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
    holdout = _load_holdouts_strict(holdout_paths)
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
        mib_number = row[2] if len(row) > 2 else 0.0
        cluster_number = row[3] if len(row) > 3 else 0.0
        mib_id = int(mib_number)
        cluster_id = int(cluster_number)
        if mib_number != mib_id or cluster_number != cluster_id:
            raise CacheError("noninteger_constraint_group", str(block))
        if mib_id < 0 or cluster_id < 0:
            raise CacheError("negative_constraint_group", str(block))
        if mib_id:
            mib.setdefault(mib_id, []).append(block)
        if cluster_id:
            cluster.setdefault(cluster_id, []).append(block)
        boundary_number = row[4] if len(row) > 4 else 0.0
        boundary_code = int(boundary_number)
        if boundary_number != boundary_code:
            raise CacheError("noninteger_boundary_mask", str(block))
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


def _validate_npz_container(path: Path) -> None:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            expected = [f"{name}.npy" for name in sorted(ALL_ARRAYS)]
            if len(names) != len(set(names)):
                raise CacheError("duplicate_npz_member", str(path))
            if names != expected:
                raise CacheError("noncanonical_npz_member_order", str(path))
            if archive.comment:
                raise CacheError("npz_archive_comment", str(path))
            for info in infos:
                if (
                    info.is_dir()
                    or info.flag_bits != 0
                    or info.date_time != (1980, 1, 1, 0, 0, 0)
                    or info.compress_type != zipfile.ZIP_DEFLATED
                    or info.external_attr != 0o100644 << 16
                    or info.internal_attr != 0
                    or info.create_system != 3
                    or info.create_version != 20
                    or info.extract_version != 20
                    or info.extra
                    or info.comment
                ):
                    raise CacheError("nondeterministic_npz_member", info.filename)
    except CacheError:
        raise
    except (OSError, zipfile.BadZipFile) as exc:
        raise CacheError("invalid_npz_container", f"{path}: {exc}") from exc


def _atomic_publish_noreplace(staging: Path, output_dir: Path) -> None:
    """Atomically publish a directory while refusing every existing target.

    Linux ``renameat2(RENAME_NOREPLACE)`` is required.  Falling back to
    ``os.replace`` or a check-then-rename sequence would reintroduce a target
    clobber race, so unsupported hosts fail closed.
    """
    if not sys.platform.startswith("linux"):
        raise CacheError("atomic_noreplace_unavailable", sys.platform)
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise CacheError("atomic_noreplace_unavailable", "renameat2")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(staging),
        at_fdcwd,
        os.fsencode(output_dir),
        rename_noreplace,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in (errno.EEXIST, errno.ENOTEMPTY):
        raise CacheError("output_exists", str(output_dir))
    if error in (errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP):
        raise CacheError("atomic_noreplace_unavailable", os.strerror(error))
    raise CacheError("atomic_publish_failed", os.strerror(error))


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


def _validated_clique_components(
    sources: np.ndarray,
    destinations: np.ndarray,
    block_count: int,
    *,
    name: str,
) -> list[list[int]]:
    sources = sources.astype(np.int64)
    destinations = destinations.astype(np.int64)
    pairs = set(zip(sources.tolist(), destinations.tolist()))
    if (
        np.any(sources < 0)
        or np.any(sources >= block_count)
        or np.any(destinations < 0)
        or np.any(destinations >= block_count)
        or np.any(sources >= destinations)
        or len(pairs) != len(sources)
    ):
        raise CacheError(f"invalid_{name}_pair")
    parents = list(range(block_count))

    def find(block: int) -> int:
        while parents[block] != block:
            parents[block] = parents[parents[block]]
            block = parents[block]
        return block

    for left, right in pairs:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root
    grouped: dict[int, list[int]] = {}
    for block in set(sources.tolist()) | set(destinations.tolist()):
        grouped.setdefault(find(block), []).append(block)
    components = [sorted(members) for members in grouped.values()]
    components.sort(key=lambda members: members[0])
    expected_pairs = {
        (left, right)
        for members in components
        for left_index, left in enumerate(members)
        for right in members[left_index + 1 :]
    }
    if pairs != expected_pairs:
        raise CacheError(f"incomplete_{name}_clique")
    return components


def _validate_vertical_support_overlap(
    golden_rectangles: np.ndarray, supports: np.ndarray, *, tolerance: float = 1e-6
) -> None:
    for child, support_value in enumerate(supports.tolist()):
        parent = int(support_value)
        if parent < 0:
            continue
        child_x, _child_y, child_width, _child_height = golden_rectangles[child]
        parent_x, _parent_y, parent_width, _parent_height = golden_rectangles[parent]
        overlap = min(
            float(child_x + child_width), float(parent_x + parent_width)
        ) - max(float(child_x), float(parent_x))
        if overlap <= tolerance:
            raise CacheError(
                "vertical_support_no_x_overlap", f"child {child}, parent {parent}"
            )


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
    fixed = arrays["fixed_mask"][node].astype(bool)
    preplaced = arrays["preplaced_mask"][node].astype(bool)
    hard_blocks = fixed | preplaced
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
        options = arrays["shape_options"][option_start:option_end]
        if np.any(options <= 0.0) or len({tuple(row) for row in options.tolist()}) != len(
            options
        ):
            raise CacheError("invalid_shape_options", str(global_block))
        if hard_blocks[local_block]:
            expected_options = np.asarray(
                [dimensions[local_block]], dtype=np.float32
            )
        else:
            expected_options = np.asarray(
                enumerate_oriented_factor_shapes(
                    arrays["area_targets"][global_block]
                ),
                dtype=np.float32,
            ).reshape(-1, 2)
        if options.shape != expected_options.shape or not np.array_equal(
            options, expected_options
        ):
            raise CacheError("shape_option_set_mismatch", str(global_block))
        selected_dimensions = options[selected]
        if not np.allclose(
            selected_dimensions, dimensions[local_block], rtol=0.0, atol=1e-6
        ):
            raise CacheError("selected_shape_dimension_mismatch", str(global_block))

    hard = arrays["hard_targets"][node]
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
    _validate_vertical_support_overlap(golden, support)
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
    mib_components = _validated_clique_components(
        mib_src, mib_dst, block_count, name="mib"
    )
    inconsistent_members: set[int] = set()
    inconsistent_group_count = 0
    for members in mib_components:
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

    reconstructed_constraints = [
        [float(fixed[block]), float(preplaced[block]), 0.0, 0.0, 0.0]
        for block in range(block_count)
    ]
    for group_identifier, members in enumerate(mib_components, 1):
        for block in members:
            reconstructed_constraints[block][2] = float(group_identifier)
    recomputed_compatible = mib_is_input_compatible(
        block_count,
        arrays["area_targets"][node].tolist(),
        reconstructed_constraints,
        hard.tolist(),
    )
    if bool(arrays["mib_input_compatible"][layout_index]) != recomputed_compatible:
        raise CacheError("mib_input_compatible_mismatch", f"{shard_name}#{layout_index}")

    cluster_start = int(arrays["cluster_pair_ptr"][layout_index])
    cluster_end = int(arrays["cluster_pair_ptr"][layout_index + 1])
    cluster_src = arrays["cluster_pair_src"][cluster_start:cluster_end]
    cluster_dst = arrays["cluster_pair_dst"][cluster_start:cluster_end]
    _validated_clique_components(
        cluster_src, cluster_dst, block_count, name="cluster"
    )
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


def _validate_configuration(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != CONFIGURATION_KEYS:
        raise CacheError("invalid_configuration_schema")
    if not _is_plain_int(value["seed"]):
        raise CacheError("invalid_configuration_seed")
    block_range = value["block_count_range"]
    if (
        not isinstance(block_range, list)
        or len(block_range) != 2
        or any(not _is_plain_int(item, minimum=1) for item in block_range)
        or block_range[0] > block_range[1]
        or block_range[1] > 120
    ):
        raise CacheError("invalid_configuration_block_range")
    layouts_per_source = value["layouts_per_source"]
    if not _is_plain_int(layouts_per_source, minimum=1):
        raise CacheError("invalid_layouts_per_source")
    max_sources = value["max_sources"]
    if max_sources is not None and not _is_plain_int(max_sources, minimum=2):
        raise CacheError("invalid_configuration_max_sources")
    max_layouts = value["max_layouts_per_source"]
    if max_layouts is not None and (
        not _is_plain_int(max_layouts, minimum=1)
        or max_layouts > layouts_per_source
    ):
        raise CacheError("invalid_configuration_max_layouts")
    if not _is_plain_int(value["shard_layouts"], minimum=1):
        raise CacheError("invalid_configuration_shard_layouts")
    if (
        not _is_plain_int(value["message_steps"], minimum=0)
        or value["message_steps"] > 16
    ):
        raise CacheError("invalid_configuration_message_steps")
    if value["mib_feature_policy"] not in {
        "unmasked",
        "mask_incompatible",
        "mask_all",
    }:
        raise CacheError("invalid_mib_feature_policy")
    if value["layout_selection"] not in {"hash_raw", "clean_plus_hash_raw"}:
        raise CacheError("invalid_layout_selection")
    tolerance = value["oracle_tolerance"]
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or tolerance != 1e-6
    ):
        raise CacheError("invalid_oracle_tolerance")
    if (
        value["oracle_mib_policy"]
        != "strict_then_mask_inconsistent_shape_supervision"
    ):
        raise CacheError("invalid_oracle_mib_policy")
    return value


def _validate_provenance_schema(
    value: Any, holdout_metadata: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != PROVENANCE_KEYS:
        raise CacheError("invalid_provenance_schema")
    repository = value["repository"]
    if (
        not isinstance(repository, dict)
        or set(repository) != REPOSITORY_PROVENANCE_KEYS
        or not _is_git_oid(repository["commit"])
        or not _is_git_oid(repository["tree"])
        or not isinstance(repository["dirty"], bool)
        or not _is_sha256(repository["dirty_status_sha256"])
        or (
            not repository["dirty"]
            and repository["dirty_status_sha256"] != EMPTY_SHA256
        )
    ):
        raise CacheError("invalid_repository_provenance")
    floorset = value["floorset"]
    if (
        not isinstance(floorset, dict)
        or set(floorset) != FLOORSET_PROVENANCE_KEYS
        or not isinstance(floorset["repository"], str)
        or not floorset["repository"]
        or not _is_git_oid(floorset["commit"])
        or not _is_git_oid(floorset["tree"])
        or floorset["data_root_mode"] != "pinned_git_checkout_root"
        or floorset["loader_relative_path"] != "lite_dataset.py"
        or not _is_sha256(floorset["loader_sha256"])
        or floorset["loader_module_name"]
        != f"_verified_floorset_lite_{floorset['loader_sha256'][:16]}"
        or floorset["dataset_class_name"] != "FloorplanDatasetLite"
        or not _is_sha256(floorset["official_sources_sha256"])
    ):
        raise CacheError("invalid_floorset_provenance")
    code_hashes = value["code_sha256"]
    if (
        not isinstance(code_hashes, dict)
        or set(code_hashes) != CODE_HASH_KEYS
        or any(not _is_sha256(digest) for digest in code_hashes.values())
        or code_hashes["floorset_lite_loader"] != floorset["loader_sha256"]
        or code_hashes["official_sources"]
        != floorset["official_sources_sha256"]
        or code_hashes["holdout_manifest_union"]
        != _canonical_json_sha256(holdout_metadata["manifest_sha256s"])
    ):
        raise CacheError("invalid_code_provenance")
    runtime = value["runtime"]
    if (
        not isinstance(runtime, dict)
        or set(runtime) != RUNTIME_PROVENANCE_KEYS
        or any(not isinstance(item, str) or not item for item in runtime.values())
    ):
        raise CacheError("invalid_runtime_provenance")
    return value


def _validate_rejection_taxonomy(value: Any) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != REJECTION_TAXONOMY_KEYS
        or value["policy"] != "abort_on_first_selected_layout_error"
        or value["successful_build_counts"] != {}
        or value["known_masked_condition"]
        != "golden_mib_dimension_inconsistency"
    ):
        raise CacheError("invalid_rejection_taxonomy")


def _validate_source_partition_schema(
    value: Any, configuration: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != SOURCE_PARTITION_KEYS:
        raise CacheError("invalid_source_partition_schema")
    if value["algorithm"] != "source_file_disjoint_block_stratified_train_dev_cal_v1":
        raise CacheError("invalid_source_partition_algorithm")
    if not _is_sha256(value["partition_sha256"]):
        raise CacheError("invalid_source_partition_sha256")
    if not isinstance(value["partial_partitions_allowed"], bool):
        raise CacheError("invalid_partial_partition_flag")
    counts = value["partition_source_counts"]
    if (
        not isinstance(counts, dict)
        or set(counts) != set(PARTITION_NAMES)
        or any(not _is_plain_int(item, minimum=0) for item in counts.values())
    ):
        raise CacheError("invalid_partition_source_counts")
    block_counts = value["partition_block_counts"]
    if not isinstance(block_counts, dict) or set(block_counts) != set(PARTITION_NAMES):
        raise CacheError("invalid_partition_block_counts")
    minimum, maximum = configuration["block_count_range"]
    for partition, bucket in block_counts.items():
        if not isinstance(bucket, dict):
            raise CacheError("invalid_partition_block_counts", partition)
        total = 0
        for key, count in bucket.items():
            try:
                block_count = int(key)
            except (TypeError, ValueError) as exc:
                raise CacheError("invalid_partition_block_key", repr(key)) from exc
            if str(block_count) != key or not minimum <= block_count <= maximum:
                raise CacheError("invalid_partition_block_key", repr(key))
            if not _is_plain_int(count, minimum=0):
                raise CacheError("invalid_partition_block_value", repr(count))
            total += count
        if total != counts[partition]:
            raise CacheError("partition_block_total", partition)
    selection = value["selection"]
    if not isinstance(selection, dict) or set(selection) != SELECTION_KEYS:
        raise CacheError("invalid_source_selection_schema")
    for key in SELECTION_KEYS - {"eligible_by_block_count"}:
        if not _is_plain_int(selection[key], minimum=0):
            raise CacheError("invalid_source_selection_count", key)
    eligible_buckets = selection["eligible_by_block_count"]
    if not isinstance(eligible_buckets, dict):
        raise CacheError("invalid_eligible_block_counts")
    eligible_total = 0
    for key, count in eligible_buckets.items():
        try:
            block_count = int(key)
        except (TypeError, ValueError) as exc:
            raise CacheError("invalid_eligible_block_key", repr(key)) from exc
        if str(block_count) != key or not minimum <= block_count <= maximum:
            raise CacheError("invalid_eligible_block_key", repr(key))
        if not _is_plain_int(count, minimum=0):
            raise CacheError("invalid_eligible_block_value", repr(count))
        eligible_total += count
    if eligible_total != selection["eligible_before_limit"]:
        raise CacheError("eligible_source_total")
    if selection["selected_after_limit"] != sum(counts.values()):
        raise CacheError("selected_partition_total")
    max_sources = configuration["max_sources"]
    expected_selected = (
        selection["eligible_before_limit"]
        if max_sources is None
        else min(max_sources, selection["eligible_before_limit"])
    )
    if selection["selected_after_limit"] != expected_selected:
        raise CacheError("max_source_selection_mismatch")
    source_index = value["source_index"]
    if (
        not isinstance(source_index, dict)
        or set(source_index) != SOURCE_INDEX_KEYS
        or source_index["schema_version"] != 1
        or not _is_sha256(source_index["source_inventory_sha256"])
        or not _is_sha256(source_index["payload_sha256"])
    ):
        raise CacheError("invalid_source_index_provenance")
    holdout = value["holdout"]
    if not isinstance(holdout, dict) or set(holdout) != HOLDOUT_KEYS:
        raise CacheError("invalid_holdout_schema")
    hashes = holdout["manifest_sha256s"]
    versions = holdout["manifest_schema_versions"]
    if (
        holdout["split_unit"] != "source_file"
        or not _is_plain_int(holdout["excluded_source_count"], minimum=1)
        or not _is_sha256(holdout["manifest_union_sha256"])
        or not isinstance(hashes, list)
        or not hashes
        or any(not _is_sha256(digest) for digest in hashes)
        or not isinstance(versions, list)
        or len(versions) != len(hashes)
        or any(not _is_plain_int(version, minimum=1) for version in versions)
        or selection["source_files_excluded"] != holdout["excluded_source_count"]
    ):
        raise CacheError("invalid_holdout_metadata")
    return value


def _verify_data_root_provenance(
    provenance: dict[str, Any], data_root: Path
) -> None:
    data_root = data_root.resolve()
    try:
        top_level = Path(
            _git_output(["rev-parse", "--show-toplevel"], data_root)
        ).resolve()
    except CacheError as exc:
        raise CacheError("data_root_not_verified_git_checkout", str(exc)) from exc
    if top_level != data_root:
        raise CacheError("data_root_not_floorset_checkout", str(top_level))
    floorset = provenance["floorset"]
    if (
        _git_output(["rev-parse", "HEAD"], data_root) != floorset["commit"]
        or _git_output(["rev-parse", "HEAD^{tree}"], data_root) != floorset["tree"]
    ):
        raise CacheError("data_root_revision_mismatch")
    loader_path = data_root / floorset["loader_relative_path"]
    if (
        not loader_path.is_file()
        or loader_path.is_symlink()
        or _file_sha256(loader_path) != floorset["loader_sha256"]
        or _git_output(["hash-object", str(loader_path)], data_root)
        != _git_output(
            ["rev-parse", f"HEAD:{floorset['loader_relative_path']}"], data_root
        )
    ):
        raise CacheError("data_root_loader_mismatch")
    official = _load_json(ROOT / "docs" / "official_sources.json")
    pinned = official.get("floorset", {})
    if (
        _file_sha256(ROOT / "docs" / "official_sources.json")
        != floorset["official_sources_sha256"]
        or pinned.get("repository") != floorset["repository"]
        or pinned.get("commit") != floorset["commit"]
        or pinned.get("tree") != floorset["tree"]
    ):
        raise CacheError("official_floorset_provenance_mismatch")


def _instantiate_verified_dataset(
    data_root: Path, provenance: dict[str, Any]
) -> Any:
    """Execute the exact hash-bound loader bytes and instantiate its class."""
    data_root = data_root.resolve()
    _verify_data_root_provenance(provenance, data_root)
    floorset = provenance["floorset"]
    loader_path = data_root / floorset["loader_relative_path"]
    try:
        loader_bytes = loader_path.read_bytes()
    except OSError as exc:
        raise CacheError("floorset_loader_read_failed", str(exc)) from exc
    observed_sha256 = hashlib.sha256(loader_bytes).hexdigest()
    if observed_sha256 != floorset["loader_sha256"]:
        raise CacheError("floorset_loader_import_hash_mismatch")
    module_name = floorset["loader_module_name"]
    spec = importlib.util.spec_from_file_location(module_name, loader_path)
    if spec is None or spec.loader is None:
        raise CacheError("floorset_loader_import_spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        code = compile(loader_bytes, str(loader_path), "exec", dont_inherit=True)
        exec(code, module.__dict__)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise CacheError("floorset_loader_import_failed", str(exc)) from exc
    if Path(module.__file__ or "").resolve() != loader_path:
        raise CacheError("floorset_loader_module_path_mismatch")
    dataset_class = getattr(module, floorset["dataset_class_name"], None)
    if (
        not isinstance(dataset_class, type)
        or dataset_class.__module__ != module_name
        or dataset_class.__name__ != floorset["dataset_class_name"]
    ):
        raise CacheError("floorset_dataset_class_mismatch")
    try:
        dataset = dataset_class(str(data_root))
    except Exception as exc:
        raise CacheError("floorset_dataset_instantiation_failed", str(exc)) from exc
    layouts_per_file = getattr(dataset, "layouts_per_file", None)
    all_files = getattr(dataset, "all_files", None)
    if (
        not _is_plain_int(layouts_per_file, minimum=1)
        or not isinstance(all_files, list)
        or not all_files
    ):
        raise CacheError("floorset_dataset_contract")
    return dataset


def _verify_local_provenance(
    provenance: dict[str, Any],
    *,
    cache_dir: Path,
    data_root: Path | None,
    holdout_paths: Iterable[Path] | None,
) -> None:
    repository = provenance["repository"]
    status = _repository_status(ignored_roots=(cache_dir,))
    if (
        _git_output(["rev-parse", "HEAD"], ROOT) != repository["commit"]
        or _git_output(["rev-parse", "HEAD^{tree}"], ROOT) != repository["tree"]
        or bool(status) != repository["dirty"]
        or hashlib.sha256(
            status.encode("utf-8", errors="surrogateescape")
        ).hexdigest()
        != repository["dirty_status_sha256"]
    ):
        raise CacheError("repository_provenance_mismatch")
    local_code_paths = {
        "builder": Path(__file__),
        "dual_parent_decoder": SOLUTION_DIR / "dual_parent_decoder.py",
        "learned_order_features": SOLUTION_DIR / "learned_order.py",
        "source_partition_implementation": ROOT / "scripts" / "train_order_model.py",
        "official_sources": ROOT / "docs" / "official_sources.json",
    }
    for name, path in local_code_paths.items():
        if _file_sha256(path) != provenance["code_sha256"][name]:
            raise CacheError("code_provenance_mismatch", name)
    if data_root is None:
        return
    _verify_data_root_provenance(provenance, data_root)
    paths = [] if holdout_paths is None else [Path(path) for path in holdout_paths]
    expected = _code_hashes(data_root.resolve(), paths)
    if holdout_paths is None:
        expected["holdout_manifest_union"] = provenance["code_sha256"][
            "holdout_manifest_union"
        ]
    if expected != provenance["code_sha256"]:
        raise CacheError("code_provenance_mismatch")


def _validate_record_types(record: Any, expected_index: int) -> None:
    if not isinstance(record, dict) or set(record) != RECORD_KEYS:
        raise CacheError("record_keys", str(expected_index))
    integer_fields = {
        "record_index": 0,
        "file_offset": 0,
        "source_size_bytes": 0,
        "block_count": 1,
        "b2b_edge_count": 0,
        "mib_pair_count": 0,
        "cluster_pair_count": 0,
        "shape_option_count": 1,
        "mib_inconsistent_block_count": 0,
        "local_layout": 0,
    }
    for field, minimum in integer_fields.items():
        if not _is_plain_int(record[field], minimum=minimum):
            raise CacheError("invalid_record_type", f"{expected_index}: {field}")
    for field in (
        "clean_offset_selected",
        "strict_mib_decodable",
        "mib_input_compatible",
        "mib_features_masked",
    ):
        if not isinstance(record[field], bool):
            raise CacheError("invalid_record_type", f"{expected_index}: {field}")
    for field in (
        "partition",
        "source_file",
        "source_sha256",
        "input_sha256",
        "optimizer_target_sha256",
        "tree_sha256",
        "golden_geometry_sha256",
        "golden_metrics_sha256",
        "shard",
    ):
        if not isinstance(record[field], str) or not record[field]:
            raise CacheError("invalid_record_type", f"{expected_index}: {field}")
    for field in ("oracle_max_coordinate_delta", "oracle_max_dimension_delta"):
        if not isinstance(record[field], float) or not math.isfinite(record[field]):
            raise CacheError("invalid_record_type", f"{expected_index}: {field}")
    groups = record["mib_inconsistent_groups"]
    if not isinstance(groups, list) or any(
        not _is_plain_int(group, minimum=1) for group in groups
    ):
        raise CacheError(
            "invalid_record_type", f"{expected_index}: mib_inconsistent_groups"
        )


def _verify_record_payloads_from_dataset(
    dataset: Any,
    *,
    data_root: Path,
    layouts_per_source: int,
    records: list[dict[str, Any]],
) -> None:
    if int(dataset.layouts_per_file) != layouts_per_source:
        raise CacheError("dataset_layout_count_mismatch")
    root = data_root.resolve()
    source_indices: dict[str, int] = {}
    for file_index, path_value in enumerate(dataset.all_files):
        path = Path(path_value).resolve()
        try:
            relative = _canonical_path(str(path.relative_to(root)))
        except ValueError as exc:
            raise CacheError("dataset_source_outside_data_root", str(path)) from exc
        if relative in source_indices:
            raise CacheError("duplicate_dataset_source", relative)
        source_indices[relative] = file_index
    for record in records:
        source = record["source_file"]
        file_index = source_indices.get(source)
        if file_index is None:
            raise CacheError("record_source_missing_from_dataset", source)
        dataset_index = file_index * layouts_per_source + record["file_offset"]
        try:
            sample = dataset[dataset_index]
            area_targets, _b2b, _p2b, _pins, constraints = sample["input"]
            tree, fp_solution, metrics = sample["label"]
            block_count = _block_count(area_targets)
            golden = training_rectangles(fp_solution, block_count)
            hard_targets = hard_targets_from_golden(constraints, golden)
            observed = {
                "input_sha256": _hash_input(sample),
                "optimizer_target_sha256": _hash_hard_targets(hard_targets),
                "tree_sha256": _hash_tensor("tree", tree),
                "golden_geometry_sha256": _hash_tensor(
                    "fp_solution", fp_solution
                ),
                "golden_metrics_sha256": _hash_tensor("metrics", metrics),
            }
        except CacheError:
            raise
        except (KeyError, IndexError, TypeError, ValueError, OSError) as exc:
            raise CacheError(
                "record_payload_reload_failed",
                f"{source}#{record['file_offset']}: {exc}",
            ) from exc
        if block_count != record["block_count"]:
            raise CacheError(
                "record_payload_block_count_mismatch",
                f"{source}#{record['file_offset']}",
            )
        for field, digest in observed.items():
            if digest != record[field]:
                raise CacheError(
                    "record_payload_hash_mismatch",
                    f"{source}#{record['file_offset']}: {field}",
                )


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
    if data_root is not None and expected_manifest_sha256 is None:
        raise CacheError("expected_manifest_required_for_full_validation")
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
    configuration = _validate_configuration(manifest.get("configuration"))
    partition_metadata = _validate_source_partition_schema(
        manifest.get("source_partition"), configuration
    )
    provenance = _validate_provenance_schema(
        manifest.get("provenance"), partition_metadata["holdout"]
    )
    _validate_rejection_taxonomy(manifest.get("rejection_taxonomy"))
    _verify_local_provenance(
        provenance,
        cache_dir=cache_dir,
        data_root=data_root,
        holdout_paths=holdout_paths,
    )
    verified_dataset = (
        None
        if data_root is None
        else _instantiate_verified_dataset(data_root, provenance)
    )
    layouts_per_source = configuration.get("layouts_per_source")
    mib_policy = configuration.get("mib_feature_policy")

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
        if descriptor["layout_count"] > configuration["shard_layouts"]:
            raise CacheError("shard_layout_limit", relative)
        shard_path = cache_dir / relative
        try:
            if shard_path.is_symlink():
                raise CacheError("shard_symlink_forbidden", relative)
            if not shard_path.is_file() or shard_path.stat().st_size != descriptor["size_bytes"]:
                raise CacheError("shard_size_mismatch", relative)
            if _file_sha256(shard_path) != descriptor["sha256"]:
                raise CacheError("shard_sha256_mismatch", relative)
            _validate_npz_container(shard_path)
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
    source_layout_counts: Counter[str] = Counter()
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
        _validate_record_types(record, expected_index)
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
        source_layout_counts[source] += 1
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
        minimum_blocks, maximum_blocks = configuration["block_count_range"]
        if not minimum_blocks <= block_count <= maximum_blocks:
            raise CacheError("record_block_count_range", str(expected_index))
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

    configured_layout_limit = configuration["max_layouts_per_source"]
    effective_layout_limit = (
        layouts_per_source
        if configured_layout_limit is None
        else configured_layout_limit
    )
    if any(count > effective_layout_limit for count in source_layout_counts.values()):
        raise CacheError("source_layout_limit")
    if configuration["layout_selection"] == "hash_raw" and any(
        clean_offsets.values()
    ):
        raise CacheError("clean_offset_in_hash_raw_cache")

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
        assert verified_dataset is not None
        _verify_record_payloads_from_dataset(
            verified_dataset,
            data_root=root,
            layouts_per_source=layouts_per_source,
            records=records,
        )

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
        verified_holdout = _load_holdouts_strict(holdout_paths)
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
    allow_dirty: bool = False,
) -> dict[str, Any]:
    data_root = data_root.resolve()
    output_dir = output_dir.resolve()
    holdout_paths = [Path(path).resolve() for path in holdout_paths]
    if output_dir.exists():
        raise CacheError("output_exists", str(output_dir))
    if source_index_cache is not None:
        resolved_index = source_index_cache.resolve()
        if resolved_index == output_dir or output_dir in resolved_index.parents:
            raise CacheError("source_index_inside_output", str(resolved_index))
    if shard_layouts < 1:
        raise CacheError("invalid_shard_layout_count")
    if max_layouts_per_source is not None and max_layouts_per_source < 1:
        raise CacheError("invalid_max_layouts_per_source")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        bootstrap_provenance = _provenance_context(
            data_root,
            holdout_paths,
            allow_dirty=allow_dirty,
            ignored_roots=(staging,),
        )
        dataset = _instantiate_verified_dataset(data_root, bootstrap_provenance)
        layouts_per_file = int(dataset.layouts_per_file)
        configuration = _validate_configuration(
            {
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
                "oracle_mib_policy": (
                    "strict_then_mask_inconsistent_shape_supervision"
                ),
            }
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
        provenance = _provenance_context(
            data_root,
            holdout_paths,
            allow_dirty=allow_dirty,
            ignored_roots=(staging,),
        )
        if (
            provenance["floorset"] != bootstrap_provenance["floorset"]
            or provenance["code_sha256"] != bootstrap_provenance["code_sha256"]
            or provenance["repository"]["commit"]
            != bootstrap_provenance["repository"]["commit"]
            or provenance["repository"]["tree"]
            != bootstrap_provenance["repository"]["tree"]
        ):
            raise CacheError("provenance_changed_during_dataset_scan")
        source_fingerprints: dict[str, tuple[str, int]] = {}
        records: list[dict[str, Any]] = []
        shard_descriptors: list[dict[str, Any]] = []
        stats = Counter()
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
            "configuration": configuration,
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
        manifest_sha256 = _file_sha256(manifest_path)
        validation = validate_cache(
            staging,
            data_root=data_root,
            holdout_paths=holdout_paths,
            expected_manifest_sha256=manifest_sha256,
        )
        _atomic_publish_noreplace(staging, output_dir)
        validation["cache_dir"] = str(output_dir)
        return validation
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _build_command(args: argparse.Namespace) -> int:
    data_root = args.data_root.resolve()
    holdouts = args.holdout_manifest or list(DEFAULT_HOLDOUTS)
    report = build_cache(
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
        help=(
            "full validation: verify the loader and replay every recorded "
            "source offset (requires --expected-manifest-sha256)"
        ),
    )
    validate.add_argument(
        "--holdout-manifest",
        type=Path,
        action="append",
        help="also re-verify configured holdout hashes and absence of leakage",
    )
    validate.add_argument(
        "--expected-manifest-sha256",
        help="require an externally recorded manifest digest; mandatory with --data-root",
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
