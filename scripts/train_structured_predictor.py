#!/usr/bin/env python3
"""Build a sharded oracle-label cache and train a compact structured baseline.

The cache is a versioned research artifact, not an inference input.  Its 60
features are produced by the exact stdlib inference implementation.  Labels
contain only supervised geometry structure: legal shape category, horizontal
parent/side, vertical support/floor, and fractional x/y ranks.  Source paths
and offsets are retained solely in cache provenance and are never columns in a
training or inference matrix.

All supplied holdout manifests are unioned at source-file granularity.  A run
intended for contest research must supply clean-v1, raw-v1, and sealed-v2; the
script refuses fewer than three manifests unless ``--allow-test-holdouts`` is
explicitly set for unit fixtures.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import sys
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOLUTION_DIR = ROOT / "contest_solution"
DEFAULT_DATA_ROOT = ROOT / "external" / "FloorSet"
sys.path.insert(0, str(SOLUTION_DIR))

from dual_parent_decoder import (  # noqa: E402
    DualParentError,
    extract_oracle_labels,
    hard_targets_from_golden,
    training_rectangles,
)
from learned_order import FEATURE_NAMES, FEATURE_VERSION, extract_order_features  # noqa: E402
from structured_predictor import (  # noqa: E402
    ARTIFACT_SCHEMA_VERSION,
    MAX_SHAPE_OPTIONS,
    MODEL_TYPE,
    NODE_TARGET_NAMES,
    PAIR_DIRECT_FEATURE_NAMES,
    PAIR_HEAD_NAMES,
    _matmul,
    _pair_features,
    _project_horizontal,
    direct_pair_features,
    extract_pair_feature_context,
    seal_artifact,
)


CACHE_SCHEMA_VERSION = 2
TRAINER_VERSION = 1
DEFAULT_MANIFESTS = (
    ROOT / "results" / "folds" / "heavy_clean_v1.json",
    ROOT / "results" / "folds" / "heavy_raw_hash_v1.json",
    ROOT / "results" / "folds" / "heavy_sealed_v2.json",
)
FORBIDDEN_FEATURE_TOKENS = (
    "source",
    "file",
    "worker",
    "instance",
    "generator",
    "prng",
    "seed",
    "offset",
)


def _canonical_source(value: str) -> str:
    text = str(value).replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    path = PurePosixPath(text)
    if not text or str(path) == "." or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"invalid manifest source path: {value!r}")
    return str(path)


def _sha256(path: Path) -> str:
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


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True)
class HoldoutUnion:
    sources: frozenset[str]
    sha256s: tuple[str, ...]
    aggregate_sha256: str


def load_holdout_union(paths: Iterable[Path]) -> HoldoutUnion:
    """Read only source_file fields; labels, offsets, and scores are ignored."""

    manifest_paths = list(paths)
    if not manifest_paths:
        raise ValueError("at least one holdout manifest is required")
    sources: set[str] = set()
    digests = []
    for path in manifest_paths:
        raw = path.read_bytes()
        data = json.loads(raw)
        if data.get("split_unit") != "source_file":
            raise ValueError(f"{path} must split at source_file granularity")
        manifests = data.get("manifests", [data])
        if not isinstance(manifests, list) or not manifests:
            raise ValueError(f"{path} has no manifest folds")
        found = 0
        for manifest in manifests:
            cases = manifest.get("cases", [])
            if not isinstance(cases, list):
                raise ValueError(f"{path} contains a non-list cases field")
            for case in cases:
                source = case.get("source_file")
                if not isinstance(source, str):
                    raise ValueError(f"{path} contains a case without source_file")
                sources.add(_canonical_source(source))
                found += 1
        if not found:
            raise ValueError(f"{path} contains no held-out sources")
        digests.append(hashlib.sha256(raw).hexdigest())
    aggregate = hashlib.sha256(b"structured_holdout_union_v1\0")
    for digest in digests:
        aggregate.update(bytes.fromhex(digest))
    return HoldoutUnion(frozenset(sources), tuple(digests), aggregate.hexdigest())


@dataclass(frozen=True)
class SourceRecord:
    file_index: int
    relative_path: str
    block_count: int
    size_bytes: int


def _source_order(namespace: str, seed: int, value: str) -> bytes:
    return hashlib.sha256(f"{namespace}:{seed}:{value}".encode("utf-8")).digest()


def load_source_index(
    dataset: Any, data_root: Path, cache_path: Path | None
) -> list[SourceRecord]:
    data_root = data_root.resolve()
    dataset.all_files = sorted(str(Path(path).resolve()) for path in dataset.all_files)
    if hasattr(dataset, "cached_file_idx"):
        dataset.cached_file_idx = -1
    by_relative: dict[str, tuple[int, Path]] = {}
    for index, path_string in enumerate(dataset.all_files):
        path = Path(path_string)
        try:
            relative = _canonical_source(str(path.relative_to(data_root)))
        except ValueError as exc:
            raise ValueError(f"dataset source lies outside data root: {path}") from exc
        by_relative[relative] = (index, path)

    if cache_path is not None and cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("schema_version") != 1:
            raise ValueError("source index schema is unsupported")
        rows = cached.get("sources")
        if not isinstance(rows, list) or len(rows) != len(by_relative):
            raise ValueError("source index inventory length is stale")
        records = []
        for row in rows:
            relative = _canonical_source(row["relative_path"])
            if relative not in by_relative:
                raise ValueError(f"source index path is absent: {relative}")
            file_index, path = by_relative[relative]
            size = int(row["size_bytes"])
            if path.stat().st_size != size:
                raise ValueError(f"source index size is stale: {relative}")
            records.append(
                SourceRecord(file_index, relative, int(row["block_count"]), size)
            )
        return records

    records = []
    for ordinal, (relative, (file_index, path)) in enumerate(sorted(by_relative.items()), 1):
        sample = dataset[file_index * int(dataset.layouts_per_file)]
        count = sum(float(value.item()) != -1.0 for value in sample["input"][0])
        records.append(SourceRecord(file_index, relative, count, path.stat().st_size))
        if ordinal % 250 == 0:
            print(f"indexed {ordinal}/{len(by_relative)} sources", flush=True)
    return records


def partition_sources(
    records: Iterable[SourceRecord],
    *,
    excluded: frozenset[str],
    min_blocks: int,
    max_blocks: int,
    validation_fraction: float,
    seed: int,
    max_sources: int | None,
) -> tuple[list[SourceRecord], list[SourceRecord]]:
    records = list(records)
    discovered = {row.relative_path for row in records}
    missing = excluded - discovered
    if missing:
        raise ValueError(
            f"{len(missing)} held-out sources are absent; first={sorted(missing)[0]}"
        )
    eligible = [
        row
        for row in records
        if min_blocks <= row.block_count <= max_blocks
        and row.relative_path not in excluded
    ]
    # The cap is applied after heavy filtering; this fixes the v4 undertraining bug.
    eligible.sort(key=lambda row: _source_order("select", seed, row.relative_path))
    if max_sources is not None:
        eligible = eligible[:max_sources]
    buckets: dict[int, list[SourceRecord]] = {}
    for row in eligible:
        buckets.setdefault(row.block_count, []).append(row)
    training = []
    validation = []
    for block_count, bucket in sorted(buckets.items()):
        bucket.sort(key=lambda row: _source_order("split", seed, row.relative_path))
        if len(bucket) < 2:
            training.extend(bucket)
            continue
        count = min(len(bucket) - 1, max(1, round(len(bucket) * validation_fraction)))
        validation.extend(bucket[:count])
        training.extend(bucket[count:])
    training.sort(key=lambda row: row.relative_path)
    validation.sort(key=lambda row: row.relative_path)
    if not training or not validation:
        raise ValueError("source partition must contain training and validation")
    assert not ({row.relative_path for row in training + validation} & excluded)
    return training, validation


def _fractional_ranks(values: list[float], tolerance: float = 1e-9) -> np.ndarray:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = np.zeros(len(values), dtype=np.float32)
    denominator = max(1, len(values) - 1)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and abs(values[order[end]] - values[order[start]]) <= tolerance:
            end += 1
        rank = ((start + end - 1) / 2.0) / denominator
        for ordinal in range(start, end):
            result[order[ordinal]] = rank
        start = end
    return result


def layout_arrays(sample: Any, *, message_steps: int) -> dict[str, np.ndarray]:
    area, b2b, p2b, pins, constraints = sample["input"]
    tree, fp_solution, _metrics = sample["label"]
    count = sum(float(value.item()) != -1.0 for value in area)
    golden = training_rectangles(fp_solution, count)
    labels = extract_oracle_labels(area, constraints, tree, golden)
    targets = hard_targets_from_golden(constraints, golden)
    features = extract_order_features(
        count,
        area,
        b2b,
        p2b,
        pins,
        constraints,
        targets,
        message_steps=message_steps,
    )
    pair_context = extract_pair_feature_context(
        count, area, b2b, p2b, constraints, features
    )
    pair_direct = np.asarray(
        [
            direct_pair_features(pair_context, child, parent)
            for child in range(count)
            for parent in range(count)
        ],
        dtype=np.float32,
    )
    centers_x = [x + width / 2.0 for x, _y, width, _height in golden]
    centers_y = [y + height / 2.0 for _x, y, _width, height in golden]
    horizontal_parent = np.full(count, -1, dtype=np.int16)
    horizontal_side = np.full(count, -1, dtype=np.int8)
    for relation in labels.horizontal:
        horizontal_parent[relation.child] = relation.parent
        horizontal_side[relation.child] = relation.side
    vertical_parent = np.asarray(
        [-1 if parent is None else parent for parent in labels.vertical_supports],
        dtype=np.int16,
    )
    constraint_rows = constraints.detach().cpu().tolist() if hasattr(constraints, "detach") else constraints
    inconsistent = set(labels.mib_inconsistent_groups)
    shape_mask = np.zeros(count, dtype=np.uint8)
    for index in range(count):
        row = constraint_rows[index]
        hard = row[0] != 0.0 or row[1] != 0.0
        group = int(row[2]) if len(row) > 2 else 0
        shape_mask[index] = int(not hard and group not in inconsistent)
    return {
        "features": np.asarray(features, dtype=np.float32),
        "shape_index": np.asarray(labels.selected_shape_indices, dtype=np.int8),
        "shape_count": np.asarray([len(row) for row in labels.shape_options], dtype=np.int8),
        "shape_mask": shape_mask,
        "x_rank": _fractional_ranks(centers_x),
        "y_rank": _fractional_ranks(centers_y),
        "horizontal_parent": horizontal_parent,
        "horizontal_side": horizontal_side,
        "vertical_parent": vertical_parent,
        "root": np.asarray([labels.root], dtype=np.int16),
        "mib_inconsistent": np.asarray([bool(inconsistent)], dtype=np.uint8),
        "pair_direct": pair_direct,
    }


class ShardBuilder:
    def __init__(self, cache_dir: Path, partition: str, shard_layouts: int):
        self.cache_dir = cache_dir
        self.partition = partition
        self.shard_layouts = shard_layouts
        self.layouts: list[dict[str, np.ndarray]] = []
        self.identities: list[dict[str, Any]] = []
        self.shards: list[dict[str, Any]] = []

    def add(self, arrays: dict[str, np.ndarray], identity: dict[str, Any]) -> None:
        self.layouts.append(arrays)
        self.identities.append(identity)
        if len(self.layouts) >= self.shard_layouts:
            self.flush()

    def flush(self) -> None:
        if not self.layouts:
            return
        offsets = [0]
        pair_offsets = [0]
        for row in self.layouts:
            offsets.append(offsets[-1] + row["features"].shape[0])
            pair_offsets.append(pair_offsets[-1] + row["pair_direct"].shape[0])
        block_fields = (
            "features", "shape_index", "shape_count", "shape_mask", "x_rank",
            "y_rank", "horizontal_parent", "horizontal_side", "vertical_parent",
        )
        arrays = {
            name: np.concatenate([row[name] for row in self.layouts], axis=0)
            for name in block_fields
        }
        arrays["layout_offsets"] = np.asarray(offsets, dtype=np.int32)
        arrays["pair_offsets"] = np.asarray(pair_offsets, dtype=np.int32)
        arrays["pair_direct"] = np.concatenate(
            [row["pair_direct"] for row in self.layouts], axis=0
        )
        arrays["roots"] = np.concatenate([row["root"] for row in self.layouts])
        arrays["mib_inconsistent"] = np.concatenate(
            [row["mib_inconsistent"] for row in self.layouts]
        )
        index = len(self.shards)
        relative = Path(f"{self.partition}-{index:04d}.npz")
        path = self.cache_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
        self.shards.append(
            {
                "path": relative.as_posix(),
                "partition": self.partition,
                "sha256": _sha256(path),
                "layouts": len(self.layouts),
                "blocks": offsets[-1],
                "identities": self.identities,
            }
        )
        self.layouts = []
        self.identities = []


def build_cache(
    dataset: Any,
    *,
    data_root: Path,
    cache_dir: Path,
    source_index_cache: Path | None,
    holdout_paths: list[Path],
    min_blocks: int,
    max_blocks: int,
    validation_fraction: float,
    seed: int,
    max_sources: int | None,
    layouts_per_source: int,
    message_steps: int,
    shard_layouts: int,
) -> dict[str, Any]:
    holdout = load_holdout_union(holdout_paths)
    records = load_source_index(dataset, data_root, source_index_cache)
    training, validation = partition_sources(
        records,
        excluded=holdout.sources,
        min_blocks=min_blocks,
        max_blocks=max_blocks,
        validation_fraction=validation_fraction,
        seed=seed,
        max_sources=max_sources,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    builders = {
        "train": ShardBuilder(cache_dir, "train", shard_layouts),
        "validation": ShardBuilder(cache_dir, "validation", shard_layouts),
    }
    stats: dict[str, Counter[str]] = {
        "train": Counter(), "validation": Counter()
    }
    source_content = hashlib.sha256(b"structured_source_content_v1\0")
    layout_count = int(dataset.layouts_per_file)
    for partition, selected in (("train", training), ("validation", validation)):
        for ordinal, source in enumerate(selected, 1):
            path = Path(dataset.all_files[source.file_index])
            source_content.update(partition.encode("utf-8") + b"\0")
            source_content.update(source.relative_path.encode("utf-8") + b"\0")
            source_content.update(bytes.fromhex(_sha256(path)))
            offsets = sorted(
                range(layout_count),
                key=lambda offset: _source_order(
                    "layout", seed, f"{source.relative_path}#{offset}"
                ),
            )[:layouts_per_source]
            for offset in sorted(offsets):
                stats[partition]["layouts_seen"] += 1
                sample = dataset[source.file_index * layout_count + offset]
                try:
                    arrays = layout_arrays(sample, message_steps=message_steps)
                except (DualParentError, IndexError, TypeError, ValueError, OverflowError) as exc:
                    reason = exc.code if isinstance(exc, DualParentError) else type(exc).__name__
                    stats[partition][f"rejected:{reason}"] += 1
                    continue
                builders[partition].add(
                    arrays,
                    {
                        "source_file": source.relative_path,
                        "file_offset": offset,
                        "block_count": source.block_count,
                    },
                )
                stats[partition]["layouts_accepted"] += 1
                stats[partition]["blocks_accepted"] += arrays["features"].shape[0]
                stats[partition]["mib_inconsistent_layouts"] += int(
                    arrays["mib_inconsistent"][0]
                )
            if ordinal % 50 == 0 or ordinal == len(selected):
                print(
                    f"cache {partition}: {ordinal}/{len(selected)} sources, "
                    f"{stats[partition]['layouts_accepted']} layouts",
                    flush=True,
                )
        builders[partition].flush()
    shards = builders["train"].shards + builders["validation"].shards
    selected_paths = {
        "train": [row.relative_path for row in training],
        "validation": [row.relative_path for row in validation],
    }
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_type": "dual_parent_supervision_shards",
        "feature_schema": {
            "version": FEATURE_VERSION,
            "names": list(FEATURE_NAMES),
            "message_steps": message_steps,
            "pair_direct_names": list(PAIR_DIRECT_FEATURE_NAMES),
            "forbidden_identity_features": list(FORBIDDEN_FEATURE_TOKENS),
        },
        "label_schema": {
            "shape": "legal_oriented_factor_pair_category",
            "horizontal": "parent_and_binary_side",
            "vertical": "support_parent_or_floor",
            "auxiliary": ["fractional_center_x_rank", "fractional_center_y_rank"],
            "corrupt_mib_policy": "mask_shape_head_for_inconsistent_group_members",
        },
        "selection": {
            "min_blocks": min_blocks,
            "max_blocks": max_blocks,
            "max_sources_after_heavy_filter": max_sources,
            "layouts_per_source": layouts_per_source,
            "validation_fraction": validation_fraction,
            "seed": seed,
            "split_unit": "source_file",
            "selected_source_counts": {
                partition: len(rows) for partition, rows in selected_paths.items()
            },
            "selected_source_sha256": {
                partition: _canonical_json_sha256(rows)
                for partition, rows in selected_paths.items()
            },
        },
        "holdouts": {
            "manifest_paths": [str(path) for path in holdout_paths],
            "manifest_sha256s": list(holdout.sha256s),
            "aggregate_sha256": holdout.aggregate_sha256,
            "excluded_source_count": len(holdout.sources),
        },
        "dataset": {
            "root": str(data_root.resolve()),
            "source_count": len(dataset.all_files),
            "layouts_per_file": layout_count,
            "selected_source_content_sha256": source_content.hexdigest(),
        },
        "stats": {key: dict(value) for key, value in stats.items()},
        "shards": shards,
    }
    manifest["payload_sha256"] = _canonical_json_sha256(manifest)
    _atomic_json(cache_dir / "manifest.json", manifest)
    return manifest


def load_cache(cache_dir: Path) -> dict[str, Any]:
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("structured cache schema is unsupported")
    expected = manifest.pop("payload_sha256", None)
    actual = _canonical_json_sha256(manifest)
    manifest["payload_sha256"] = expected
    if expected != actual:
        raise ValueError("structured cache manifest hash mismatch")
    feature_schema = manifest.get("feature_schema", {})
    if feature_schema.get("version") != FEATURE_VERSION or feature_schema.get("names") != list(FEATURE_NAMES):
        raise ValueError("structured cache feature schema is stale")
    if feature_schema.get("pair_direct_names") != list(PAIR_DIRECT_FEATURE_NAMES):
        raise ValueError("structured cache direct pair schema is stale")
    lowered = [name.lower() for name in FEATURE_NAMES]
    if any(token in name for token in FORBIDDEN_FEATURE_TOKENS for name in lowered):
        raise ValueError("structured cache exposes a prohibited identity feature")
    for shard in manifest.get("shards", []):
        path = cache_dir / shard["path"]
        if not path.is_file() or _sha256(path) != shard["sha256"]:
            raise ValueError(f"structured cache shard hash mismatch: {path}")
    return manifest


class WeightedRidgeMoments:
    def __init__(self, feature_count: int):
        self.feature_count = feature_count
        self.weight = 0.0
        self.x_sum = np.zeros(feature_count, dtype=np.float64)
        self.y_sum = 0.0
        self.xtx = np.zeros((feature_count, feature_count), dtype=np.float64)
        self.xty = np.zeros(feature_count, dtype=np.float64)
        self.yty = 0.0

    def update(self, features: np.ndarray, targets: np.ndarray, weights: np.ndarray | None = None):
        if features.ndim != 2 or features.shape[1] != self.feature_count:
            raise ValueError("ridge feature shape mismatch")
        if targets.shape != (features.shape[0],):
            raise ValueError("ridge target shape mismatch")
        weights = np.ones(features.shape[0]) if weights is None else weights
        if weights.shape != targets.shape or np.any(weights < 0.0):
            raise ValueError("ridge weight shape mismatch")
        keep = weights > 0.0
        if not np.any(keep):
            return
        x = np.asarray(features[keep], dtype=np.float64)
        y = np.asarray(targets[keep], dtype=np.float64)
        w = np.asarray(weights[keep], dtype=np.float64)
        self.weight += float(w.sum())
        self.x_sum += (x * w[:, None]).sum(axis=0)
        self.y_sum += float((y * w).sum())
        self.xtx += x.T @ (x * w[:, None])
        self.xty += x.T @ (y * w)
        self.yty += float((y * y * w).sum())


def _normalization(moment: WeightedRidgeMoments) -> tuple[np.ndarray, np.ndarray]:
    mean = moment.x_sum / moment.weight
    variance = np.diag(moment.xtx) / moment.weight - mean * mean
    center = mean.copy()
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale < 1e-12] = 1.0
    center[0] = 0.0
    scale[0] = 1.0
    return center, scale


def _fit(moment: WeightedRidgeMoments, center: np.ndarray, scale: np.ndarray, ridge: float) -> np.ndarray:
    numerator = (
        moment.xtx - np.outer(center, moment.x_sum) - np.outer(moment.x_sum, center)
        + moment.weight * np.outer(center, center)
    )
    ztz = numerator / np.outer(scale, scale)
    zty = (moment.xty - center * moment.y_sum) / scale
    system = ztz.copy()
    penalty = ridge * moment.weight
    system.flat[:: system.shape[0] + 1] += penalty
    system[0, 0] -= penalty
    try:
        weights = np.linalg.solve(system, zty)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(system, zty, rcond=None)[0]
    if not np.isfinite(weights).all():
        raise RuntimeError("ridge fit produced non-finite coefficients")
    return weights


def _shard_arrays(cache_dir: Path, manifest: dict[str, Any], partition: str):
    for shard in manifest["shards"]:
        if shard["partition"] == partition:
            with np.load(cache_dir / shard["path"], allow_pickle=False) as arrays:
                yield {name: arrays[name] for name in arrays.files}


def _node_targets(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    blocks = arrays["features"].shape[0]
    targets = np.zeros((blocks, len(NODE_TARGET_NAMES)), dtype=np.float64)
    masks = np.ones_like(targets)
    targets[:, 0] = arrays["x_rank"]
    targets[:, 1] = arrays["y_rank"]
    for option in range(MAX_SHAPE_OPTIONS):
        targets[:, 2 + option] = arrays["shape_index"] == option
        masks[:, 2 + option] = arrays["shape_mask"] * (arrays["shape_count"] > option)
    offsets = arrays["layout_offsets"]
    masks[:, -2:] = 1.0
    for layout, root in enumerate(arrays["roots"]):
        start, end = int(offsets[layout]), int(offsets[layout + 1])
        targets[start + int(root), -2] = 1.0
        floors = arrays["vertical_parent"][start:end] < 0
        targets[start:end, -1] = floors
    return targets, masks


def _pair_rows(
    hidden: np.ndarray,
    arrays: dict[str, np.ndarray],
    *,
    negative_count: int,
    max_children: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rows = [[] for _ in PAIR_HEAD_NAMES]
    targets = [[] for _ in PAIR_HEAD_NAMES]
    offsets = arrays["layout_offsets"]
    pair_offsets = arrays["pair_offsets"]
    for layout in range(len(offsets) - 1):
        start, end = int(offsets[layout]), int(offsets[layout + 1])
        count = end - start
        pair_start = int(pair_offsets[layout])

        def pair_row(child: int, parent: int) -> np.ndarray:
            direct = arrays["pair_direct"][pair_start + child * count + parent]
            return np.asarray(
                _pair_features(
                    hidden[start + child], hidden[start + parent], direct
                )
            )

        children = list(range(count))
        children.sort(key=lambda child: hashlib.sha256(f"{layout}:{child}".encode()).digest())
        children = children[:max_children]
        root = int(arrays["roots"][layout])
        for child in children:
            if child != root:
                parent = int(arrays["horizontal_parent"][start + child])
                side = int(arrays["horizontal_side"][start + child])
                positive = pair_row(child, parent)
                rows[side].append(positive)
                targets[side].append(1.0)
                rows[1 - side].append(positive)
                targets[1 - side].append(0.0)
                candidates = [value for value in range(count) if value not in (child, parent)]
                candidates.sort(key=lambda value: ((child + 1) * 65537 + (value + 1) * 257) % 104729)
                for candidate in candidates[:negative_count]:
                    rows[side].append(pair_row(child, candidate))
                    targets[side].append(0.0)
            support = int(arrays["vertical_parent"][start + child])
            if support >= 0:
                rows[2].append(pair_row(child, support))
                targets[2].append(1.0)
                candidates = [value for value in range(count) if value not in (child, support)]
                candidates.sort(key=lambda value: ((child + 1) * 8191 + (value + 1) * 131) % 65537)
                for candidate in candidates[:negative_count]:
                    rows[2].append(pair_row(child, candidate))
                    targets[2].append(0.0)
    return (
        [np.asarray(value, dtype=np.float64) for value in rows],
        [np.asarray(value, dtype=np.float64) for value in targets],
    )


def train_model(
    cache_dir: Path,
    manifest: dict[str, Any],
    *,
    hidden_size: int,
    ridge: float,
    pair_ridge: float,
    negative_count: int,
    max_relation_children: int,
) -> dict[str, Any]:
    feature_count = len(FEATURE_NAMES)
    feature_moment = WeightedRidgeMoments(feature_count)
    node_moments = [WeightedRidgeMoments(feature_count) for _ in NODE_TARGET_NAMES]
    for arrays in _shard_arrays(cache_dir, manifest, "train"):
        features = np.asarray(arrays["features"], dtype=np.float64)
        dummy = np.zeros(features.shape[0])
        feature_moment.update(features, dummy)
        targets, masks = _node_targets(arrays)
        for output, moment in enumerate(node_moments):
            moment.update(features, targets[:, output], masks[:, output])
    center, scale = _normalization(feature_moment)
    correlation = (
        feature_moment.xtx
        - np.outer(center, feature_moment.x_sum)
        - np.outer(feature_moment.x_sum, center)
        + feature_moment.weight * np.outer(center, center)
    ) / np.outer(scale, scale) / feature_moment.weight
    correlation[0, :] = 0.0
    correlation[:, 0] = 0.0
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    order = np.argsort(eigenvalues)[::-1][:hidden_size]
    projection = eigenvectors[:, order]
    for column in range(projection.shape[1]):
        pivot = int(np.argmax(np.abs(projection[:, column])))
        if projection[pivot, column] < 0.0:
            projection[:, column] *= -1.0
    node_coefficients = np.column_stack(
        [_fit(moment, center, scale, ridge) for moment in node_moments]
    )

    pair_feature_count = 1 + 5 * hidden_size + len(PAIR_DIRECT_FEATURE_NAMES)
    pair_moments = [WeightedRidgeMoments(pair_feature_count) for _ in PAIR_HEAD_NAMES]
    for arrays in _shard_arrays(cache_dir, manifest, "train"):
        normalized = (np.asarray(arrays["features"], dtype=np.float64) - center) / scale
        hidden = normalized @ projection
        rows, targets = _pair_rows(
            hidden,
            arrays,
            negative_count=negative_count,
            max_children=max_relation_children,
        )
        for head, moment in enumerate(pair_moments):
            if rows[head].size:
                moment.update(rows[head], targets[head])
    pair_center = np.zeros(pair_feature_count)
    pair_scale = np.ones(pair_feature_count)
    pair_coefficients = np.column_stack(
        [_fit(moment, pair_center, pair_scale, pair_ridge) for moment in pair_moments]
    )

    model = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_type": MODEL_TYPE,
        "feature_schema": manifest["feature_schema"],
        "structured_schema": {
            "node_targets": list(NODE_TARGET_NAMES),
            "pair_heads": list(PAIR_HEAD_NAMES),
            "pair_direct_features": list(PAIR_DIRECT_FEATURE_NAMES),
            "max_shape_options": MAX_SHAPE_OPTIONS,
            "hidden_size": hidden_size,
            "pair_feature_count": pair_feature_count,
        },
        "normalization": {"center": center.tolist(), "scale": scale.tolist()},
        "hidden_projection": projection.tolist(),
        "node_coefficients": node_coefficients.tolist(),
        "pair_coefficients": pair_coefficients.tolist(),
        # Threshold is intentionally fail-closed until validation proves exact layouts.
        "calibration": {
            "confidence_threshold": 1.0,
            "margin_scale": 1.0,
            "margin_bias": 0.0,
            "policy": "zero_false_accepts_on_internal_source_holdout",
        },
        "training": {
            "trainer_version": TRAINER_VERSION,
            "ridge_lambda_mean_loss": ridge,
            "pair_ridge_lambda_mean_loss": pair_ridge,
            "negative_pairs_per_positive": negative_count,
            "max_relation_children_per_layout": max_relation_children,
            "hidden_method": "top_eigenvectors_of_standardized_feature_correlation",
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
        },
        "provenance": {
            "cache_manifest_sha256": manifest["payload_sha256"],
            "cache_holdout_aggregate_sha256": manifest["holdouts"]["aggregate_sha256"],
            "excluded_source_count": manifest["holdouts"]["excluded_source_count"],
            "trainer_sha256": _sha256(Path(__file__)),
            "feature_implementation_sha256": _sha256(SOLUTION_DIR / "learned_order.py"),
            "decoder_sha256": _sha256(SOLUTION_DIR / "dual_parent_decoder.py"),
            "inference_sha256": _sha256(SOLUTION_DIR / "structured_predictor.py"),
            "prohibited_model_inputs": list(FORBIDDEN_FEATURE_TOKENS),
        },
    }
    return seal_artifact(model)


def validation_label_metrics(
    cache_dir: Path, manifest: dict[str, Any], model: dict[str, Any]
) -> dict[str, Any]:
    center = np.asarray(model["normalization"]["center"])
    scale = np.asarray(model["normalization"]["scale"])
    node_weights = np.asarray(model["node_coefficients"])
    projection = np.asarray(model["hidden_projection"])
    pair_weights = np.asarray(model["pair_coefficients"])
    counts = Counter()
    absolute_rank_error = np.zeros(2)
    margins_and_exact: list[tuple[float, bool]] = []
    for arrays in _shard_arrays(cache_dir, manifest, "validation"):
        features = (np.asarray(arrays["features"], dtype=np.float64) - center) / scale
        node = features @ node_weights
        hidden = features @ projection
        offsets = arrays["layout_offsets"]
        pair_offsets = arrays["pair_offsets"]
        for layout in range(len(offsets) - 1):
            start, end = int(offsets[layout]), int(offsets[layout + 1])
            count = end - start
            pair_start = int(pair_offsets[layout])
            node_layout = node[start:end]
            absolute_rank_error += np.abs(
                node_layout[:, :2]
                - np.column_stack((arrays["x_rank"][start:end], arrays["y_rank"][start:end]))
            ).sum(axis=0)
            counts["blocks"] += count
            shape_exact = True
            for block in range(count):
                if not arrays["shape_mask"][start + block]:
                    continue
                valid = int(arrays["shape_count"][start + block])
                predicted = int(np.argmax(node_layout[block, 2 : 2 + valid]))
                correct = predicted == int(arrays["shape_index"][start + block])
                counts["shape_labels"] += 1
                counts["shape_correct"] += int(correct)
                shape_exact &= correct
            predicted_root = int(np.argmax(node_layout[:, -2]))
            root_exact = predicted_root == int(arrays["roots"][layout])
            counts["root_correct"] += int(root_exact)

            pair_scores = []
            for child in range(count):
                child_rows = []
                for parent in range(count):
                    if child == parent:
                        child_rows.append([-math.inf] * len(PAIR_HEAD_NAMES))
                    else:
                        child_rows.append(
                            np.asarray(
                                _pair_features(
                                    hidden[start + child],
                                    hidden[start + parent],
                                    arrays["pair_direct"][
                                        pair_start + child * count + parent
                                    ],
                                )
                            )
                            @ pair_weights
                        )
                pair_scores.append(child_rows)
            root, horizontal, horizontal_margin = _project_horizontal(
                node_layout.tolist(), pair_scores
            )
            predicted_h = {
                relation.child: (relation.parent, relation.side) for relation in horizontal
            }
            horizontal_exact = root == int(arrays["roots"][layout])
            for child in range(count):
                parent = int(arrays["horizontal_parent"][start + child])
                if parent < 0:
                    continue
                correct = predicted_h.get(child) == (
                    parent, int(arrays["horizontal_side"][start + child])
                )
                counts["horizontal_edges"] += 1
                counts["horizontal_correct"] += int(correct)
                horizontal_exact &= correct

            vertical_exact = True
            vertical_margins = []
            for child in range(count):
                choices = [(node_layout[child, -1], -1)] + [
                    (pair_scores[child][parent][2], parent)
                    for parent in range(count) if parent != child
                ]
                choices.sort(key=lambda row: (-row[0], row[1]))
                correct = choices[0][1] == int(arrays["vertical_parent"][start + child])
                counts["vertical_labels"] += 1
                counts["vertical_correct"] += int(correct)
                vertical_exact &= correct
                vertical_margins.append(choices[0][0] - choices[1][0])
            full_exact = shape_exact and horizontal_exact and vertical_exact
            counts["layouts"] += 1
            counts["shape_exact_layouts"] += int(shape_exact)
            counts["horizontal_exact_layouts"] += int(horizontal_exact)
            counts["vertical_exact_layouts"] += int(vertical_exact)
            counts["full_label_exact_layouts"] += int(full_exact)
            margin = min(horizontal_margin, min(vertical_margins, default=20.0))
            margins_and_exact.append((margin, full_exact))

    # A deployable threshold may accept only observed exact layouts and must
    # have zero false accepts.  With no such evidence it remains 1.0.
    threshold = 1.0
    exact_confidences = [1.0 / (1.0 + math.exp(-margin)) for margin, exact in margins_and_exact if exact]
    if exact_confidences:
        candidate = min(exact_confidences)
        false_accepts = sum(
            not exact and 1.0 / (1.0 + math.exp(-margin)) >= candidate
            for margin, exact in margins_and_exact
        )
        if false_accepts == 0:
            threshold = candidate
    layouts = max(1, counts["layouts"])
    blocks = max(1, counts["blocks"])
    return {
        "layouts": counts["layouts"],
        "blocks": counts["blocks"],
        "rank_mae": (absolute_rank_error / blocks).tolist(),
        "shape_accuracy": counts["shape_correct"] / max(1, counts["shape_labels"]),
        "root_accuracy": counts["root_correct"] / layouts,
        "horizontal_edge_accuracy": counts["horizontal_correct"] / max(1, counts["horizontal_edges"]),
        "vertical_label_accuracy": counts["vertical_correct"] / max(1, counts["vertical_labels"]),
        "shape_exact_layout_rate": counts["shape_exact_layouts"] / layouts,
        "horizontal_exact_layout_rate": counts["horizontal_exact_layouts"] / layouts,
        "vertical_exact_layout_rate": counts["vertical_exact_layouts"] / layouts,
        "full_label_exact_layout_rate": counts["full_label_exact_layouts"] / layouts,
        "zero_false_accept_confidence_threshold": threshold,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--holdout-manifest", type=Path, action="append")
    parser.add_argument("--allow-test-holdouts", action="store_true")
    parser.add_argument("--source-index-cache", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "results" / "work" / "structured_cache_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "models" / "structured_linear_v1.json")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--min-blocks", type=int, default=100)
    parser.add_argument("--max-blocks", type=int, default=120)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--max-sources", type=int, default=0)
    parser.add_argument("--layouts-per-source", type=int, default=8)
    parser.add_argument("--message-steps", type=int, default=4)
    parser.add_argument("--shard-layouts", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--pair-ridge", type=float, default=1e-3)
    parser.add_argument("--negative-pairs", type=int, default=2)
    parser.add_argument("--max-relation-children", type=int, default=48)
    return parser


def main() -> int:
    args = _parser().parse_args()
    holdouts = args.holdout_manifest or list(DEFAULT_MANIFESTS)
    if len(holdouts) < 3 and not args.allow_test_holdouts:
        raise SystemExit("contest training requires clean-v1, raw-v1, and sealed-v2 holdouts")
    if any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS for name in FEATURE_NAMES):
        raise SystemExit("inference feature schema contains a prohibited identity feature")
    sys.path.insert(0, str(args.data_root.resolve()))
    from lite_dataset import FloorplanDatasetLite  # noqa: PLC0415

    if args.rebuild_cache or not (args.cache_dir / "manifest.json").is_file():
        dataset = FloorplanDatasetLite(str(args.data_root))
        build_cache(
            dataset,
            data_root=args.data_root,
            cache_dir=args.cache_dir,
            source_index_cache=args.source_index_cache,
            holdout_paths=holdouts,
            min_blocks=args.min_blocks,
            max_blocks=args.max_blocks,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            max_sources=args.max_sources or None,
            layouts_per_source=args.layouts_per_source,
            message_steps=args.message_steps,
            shard_layouts=args.shard_layouts,
        )
    manifest = load_cache(args.cache_dir)
    model = train_model(
        args.cache_dir,
        manifest,
        hidden_size=args.hidden_size,
        ridge=args.ridge,
        pair_ridge=args.pair_ridge,
        negative_count=args.negative_pairs,
        max_relation_children=args.max_relation_children,
    )
    metrics = validation_label_metrics(args.cache_dir, manifest, model)
    model["calibration"]["confidence_threshold"] = metrics[
        "zero_false_accept_confidence_threshold"
    ]
    model["validation"] = metrics
    model = seal_artifact(model)
    _atomic_json(args.output, model)
    print(json.dumps({"output": str(args.output), "validation": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
