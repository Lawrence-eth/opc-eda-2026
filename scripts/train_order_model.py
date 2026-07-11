#!/usr/bin/env python3
"""Train a compact, deterministic coordinate-order model on FloorSet labels.

This trainer deliberately has one feature implementation: every predictor row
is produced by :mod:`contest_solution.learned_order`.  Golden layouts are used
only to construct the supervised targets and the fixed/preplaced values that
are part of the contest input at inference time.

The immutable holdout manifest is enforced at the source ``.th`` file level.
The remaining files are deterministically divided into disjoint training and
validation partitions.  Sufficient statistics are accumulated one layout at a
time, so the full example matrix is never retained in memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOLUTION_DIR = ROOT / "contest_solution"
FLOORSET = ROOT / "external" / "FloorSet"
sys.path.insert(0, str(SOLUTION_DIR))
sys.path.insert(0, str(FLOORSET))

from learned_order import (  # noqa: E402
    FEATURE_NAMES,
    FEATURE_VERSION,
    extract_order_features,
)
from lite_dataset import FloorplanDatasetLite  # noqa: E402


TRAINER_VERSION = 3
MODEL_SCHEMA_VERSION = 1
TARGET_NAMES = ("golden_center_x_bbox_norm", "golden_center_y_bbox_norm")


def _scalar(value) -> float:
    return float(value.item()) if hasattr(value, "item") else float(value)


def _rows(value):
    if value is None:
        return []
    return value.tolist() if hasattr(value, "tolist") else value


def _canonical_source_path(value: str) -> str:
    """Return a safe, platform-independent manifest source path."""
    text = str(value).replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    path = PurePosixPath(text)
    if not text or str(path) == "." or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"invalid source_file path in holdout manifest: {value!r}")
    return str(path)


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class HoldoutSources:
    paths: frozenset[str]
    manifest_sha256: str
    manifest_sha256s: tuple[str, ...]
    manifest_schema_versions: tuple[int, ...]
    manifest_split_unit: str


def load_holdout_sources(paths: Path | list[Path] | tuple[Path, ...]) -> HoldoutSources:
    """Load the union of held-out source files from one or more manifests."""
    manifest_paths = [paths] if isinstance(paths, Path) else list(paths)
    if not manifest_paths:
        raise ValueError("at least one holdout manifest is required")
    sources = set()
    case_count = 0
    digests = []
    schema_versions = []
    for path in manifest_paths:
        raw = path.read_bytes()
        data = json.loads(raw)
        split_unit = str(data.get("split_unit", ""))
        if split_unit != "source_file":
            raise ValueError(
                f"holdout manifest {path} split_unit must be 'source_file', "
                f"got {split_unit!r}"
            )
        manifests = data.get("manifests")
        if manifests is None:
            manifests = [data]
        if not isinstance(manifests, list) or not manifests:
            raise ValueError(f"holdout manifest {path} contains no folds")
        for manifest in manifests:
            cases = manifest.get("cases", [])
            if not isinstance(cases, list):
                raise ValueError(
                    f"holdout manifest {path} has a non-list cases field"
                )
            for case in cases:
                if "source_file" not in case:
                    raise ValueError(
                        f"holdout manifest {path} case lacks source_file"
                    )
                sources.add(_canonical_source_path(case["source_file"]))
                case_count += 1
        digests.append(hashlib.sha256(raw).hexdigest())
        schema_versions.append(int(data.get("schema_version", 0)))
    if case_count == 0 or not sources:
        raise ValueError("holdout manifests contain no held-out sources")
    if len(digests) == 1:
        aggregate_digest = digests[0]
    else:
        aggregate = hashlib.sha256()
        aggregate.update(b"floorset_holdout_manifest_union_v1\0")
        for digest in digests:
            aggregate.update(bytes.fromhex(digest))
        aggregate_digest = aggregate.hexdigest()
    return HoldoutSources(
        paths=frozenset(sources),
        manifest_sha256=aggregate_digest,
        manifest_sha256s=tuple(digests),
        manifest_schema_versions=tuple(schema_versions),
        manifest_split_unit="source_file",
    )


@dataclass(frozen=True)
class SourceFile:
    file_index: int
    relative_path: str


def _hash_order(namespace: str, seed: int, relative_path: str) -> bytes:
    return hashlib.sha256(
        f"{namespace}:{seed}:{relative_path}".encode("utf-8")
    ).digest()


def partition_source_files(
    dataset,
    *,
    data_root: Path,
    excluded_sources: frozenset[str],
    seed: int,
    validation_fraction: float,
    max_files: int | None,
):
    """Return deterministic, exactly file-disjoint train and validation lists."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    if max_files is not None and max_files < 2:
        raise ValueError("max_files must be at least 2 when specified")

    data_root = data_root.resolve()
    dataset.all_files = sorted(str(Path(path).resolve()) for path in dataset.all_files)
    if hasattr(dataset, "cached_file_idx"):
        dataset.cached_file_idx = -1
    records = []
    discovered = set()
    for file_index, path_string in enumerate(dataset.all_files):
        path = Path(path_string)
        try:
            relative = _canonical_source_path(str(path.relative_to(data_root)))
        except ValueError as exc:
            raise ValueError(f"dataset source lies outside data root: {path}") from exc
        if relative in discovered:
            raise ValueError(f"dataset contains duplicate source path: {relative}")
        discovered.add(relative)
        if relative not in excluded_sources:
            records.append(SourceFile(file_index, relative))

    missing_exclusions = sorted(excluded_sources - discovered)
    if missing_exclusions:
        preview = ", ".join(missing_exclusions[:5])
        raise ValueError(
            f"{len(missing_exclusions)} holdout sources were not found under {data_root}: "
            f"{preview}"
        )
    if len(records) < 2:
        raise ValueError("fewer than two non-holdout source files remain")

    records.sort(key=lambda row: _hash_order("select", seed, row.relative_path))
    if max_files is not None:
        records = records[:max_files]
    # One source file has one fixed block count across its 112 configurations.
    # Split independently inside each n bucket so validation cannot silently
    # omit high-weight sizes (an unstratified 15% split did omit 9/21 heavy n's).
    buckets = {}
    for record in records:
        sample = dataset[record.file_index * int(dataset.layouts_per_file)]
        block_count = _block_count(sample["input"][0])
        buckets.setdefault(block_count, []).append(record)
    training = []
    validation = []
    for block_count in sorted(buckets):
        bucket = buckets[block_count]
        bucket.sort(key=lambda row: _hash_order("split", seed, row.relative_path))
        if len(bucket) < 2:
            training.extend(bucket)
            continue
        validation_count = int(round(len(bucket) * validation_fraction))
        validation_count = min(len(bucket) - 1, max(1, validation_count))
        validation.extend(bucket[:validation_count])
        training.extend(bucket[validation_count:])
    training.sort(key=lambda row: row.relative_path)
    validation.sort(key=lambda row: row.relative_path)

    assert not ({row.relative_path for row in training} & excluded_sources)
    assert not ({row.relative_path for row in validation} & excluded_sources)
    assert not (
        {row.relative_path for row in training}
        & {row.relative_path for row in validation}
    )
    return training, validation, len(discovered)


class LayoutRejected(ValueError):
    """A malformed or unusable supervised layout (with a stable reason code)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _block_count(area_targets) -> int:
    return sum(_scalar(value) != -1.0 for value in _rows(area_targets))


def layout_examples(sample, *, message_steps: int):
    """Build one layout's shared features and normalized center targets."""
    try:
        area_targets, b2b, p2b, pins, constraints = sample["input"]
        _tree, fp_sol, _metrics = sample["label"]
    except (KeyError, TypeError, ValueError) as exc:
        raise LayoutRejected("malformed_sample") from exc

    n = _block_count(area_targets)
    if n < 1:
        raise LayoutRejected("empty_layout")
    area_rows = _rows(area_targets)
    constraint_rows = _rows(constraints)
    golden_rows = _rows(fp_sol)
    if len(golden_rows) < n or any(len(row) < 4 for row in golden_rows[:n]):
        raise LayoutRejected("malformed_golden_positions")
    if constraint_rows and len(constraint_rows) < n:
        raise LayoutRejected("malformed_constraints")
    if any(not math.isfinite(_scalar(area_rows[i])) or _scalar(area_rows[i]) <= 0 for i in range(n)):
        raise LayoutRejected("invalid_area")

    golden = []
    for row in golden_rows[:n]:
        width, height, x, y = map(_scalar, row[:4])
        if not all(math.isfinite(value) for value in (width, height, x, y)):
            raise LayoutRejected("nonfinite_golden_positions")
        if width <= 0.0 or height <= 0.0:
            raise LayoutRejected("invalid_golden_shape")
        golden.append((width, height, x, y))

    min_x = min(x for width, height, x, y in golden)
    min_y = min(y for width, height, x, y in golden)
    max_x = max(x + width for width, height, x, y in golden)
    max_y = max(y + height for width, height, x, y in golden)
    span_x = max_x - min_x
    span_y = max_y - min_y
    if not math.isfinite(span_x) or not math.isfinite(span_y) or span_x <= 0 or span_y <= 0:
        raise LayoutRejected("degenerate_golden_bbox")

    target_positions = [[-1.0, -1.0, -1.0, -1.0] for _ in range(n)]
    constraint_columns = len(constraint_rows[0]) if constraint_rows else 0
    for i, (width, height, x, y) in enumerate(golden):
        fixed = constraint_columns > 0 and _scalar(constraint_rows[i][0]) != 0.0
        preplaced = constraint_columns > 1 and _scalar(constraint_rows[i][1]) != 0.0
        if preplaced:
            target_positions[i] = [x, y, width, height]
        elif fixed:
            target_positions[i][2:] = [width, height]

    try:
        features = extract_order_features(
            n,
            area_targets,
            b2b,
            p2b,
            pins,
            constraints,
            target_positions,
            message_steps=message_steps,
        )
    except (IndexError, TypeError, ValueError, OverflowError) as exc:
        raise LayoutRejected("feature_extraction_failed") from exc
    targets = [
        [
            (x + width / 2.0 - min_x) / span_x,
            (y + height / 2.0 - min_y) / span_y,
        ]
        for width, height, x, y in golden
    ]
    if len(features) != n or any(len(row) != len(FEATURE_NAMES) for row in features):
        raise LayoutRejected("feature_schema_mismatch")
    if any(not math.isfinite(value) for row in features for value in row):
        raise LayoutRejected("nonfinite_features")
    return features, targets, n


@dataclass
class ScanStats:
    source_files: int
    layouts_seen: int = 0
    layouts_accepted: int = 0
    blocks_accepted: int = 0
    rejections: Counter = field(default_factory=Counter)
    layouts_by_block_count: Counter = field(default_factory=Counter)
    accepted_sources: set[str] = field(default_factory=set)

    def as_json(self):
        return {
            "source_files_selected": self.source_files,
            "source_files_with_examples": len(self.accepted_sources),
            "layouts_seen": self.layouts_seen,
            "layouts_accepted": self.layouts_accepted,
            "blocks_accepted": self.blocks_accepted,
            "rejections": dict(sorted(self.rejections.items())),
            "layouts_by_block_count": {
                str(key): self.layouts_by_block_count[key]
                for key in sorted(self.layouts_by_block_count)
            },
        }


def stream_layouts(
    dataset,
    sources,
    *,
    min_blocks: int,
    max_blocks: int,
    max_layouts_per_file: int | None,
    message_steps: int,
    layout_seed: int,
    stats: ScanStats,
    progress_every_files: int = 0,
    partition_name: str = "partition",
):
    """Lazily yield one accepted layout matrix at a time."""
    layouts_per_file = int(dataset.layouts_per_file)
    for ordinal, source in enumerate(sources, 1):
        base = source.file_index * layouts_per_file
        accepted_in_source = False
        offsets = list(range(layouts_per_file))
        if max_layouts_per_file is not None:
            offsets.sort(
                key=lambda offset: _hash_order(
                    "layout", layout_seed, f"{source.relative_path}#{offset}"
                )
            )
            offsets = offsets[:max_layouts_per_file]
            offsets.sort()
        for offset in offsets:
            stats.layouts_seen += 1
            sample = dataset[base + offset]
            n = _block_count(sample["input"][0])
            if n < min_blocks or n > max_blocks:
                stats.rejections["outside_block_range"] += 1
                continue
            try:
                features, targets, n = layout_examples(
                    sample, message_steps=message_steps
                )
            except LayoutRejected as exc:
                stats.rejections[exc.reason] += 1
                continue
            stats.layouts_accepted += 1
            stats.blocks_accepted += n
            stats.layouts_by_block_count[n] += 1
            accepted_in_source = True
            yield np.asarray(features, dtype=np.float64), np.asarray(
                targets, dtype=np.float64
            )
        if accepted_in_source:
            stats.accepted_sources.add(source.relative_path)
        if progress_every_files and ordinal % progress_every_files == 0:
            print(
                f"{partition_name}: {ordinal}/{len(sources)} files, "
                f"{stats.layouts_accepted} layouts, {stats.blocks_accepted} blocks",
                flush=True,
            )


class RidgeMoments:
    """Streaming sufficient statistics for a two-output linear ridge model."""

    def __init__(self, feature_count: int, output_count: int = 2):
        self.feature_count = feature_count
        self.output_count = output_count
        self.count = 0
        self.x_sum = np.zeros(feature_count, dtype=np.float64)
        self.y_sum = np.zeros(output_count, dtype=np.float64)
        self.xtx = np.zeros((feature_count, feature_count), dtype=np.float64)
        self.xty = np.zeros((feature_count, output_count), dtype=np.float64)
        self.yty = np.zeros((output_count, output_count), dtype=np.float64)

    def update(self, features: np.ndarray, targets: np.ndarray):
        if features.ndim != 2 or features.shape[1] != self.feature_count:
            raise ValueError("unexpected feature matrix shape")
        if targets.shape != (features.shape[0], self.output_count):
            raise ValueError("unexpected target matrix shape")
        if not np.isfinite(features).all() or not np.isfinite(targets).all():
            raise ValueError("nonfinite training matrix")
        self.count += features.shape[0]
        self.x_sum += features.sum(axis=0)
        self.y_sum += targets.sum(axis=0)
        self.xtx += features.T @ features
        self.xty += features.T @ targets
        self.yty += targets.T @ targets


def fit_standardized_ridge(moments: RidgeMoments, ridge_lambda: float):
    """Fit mean-loss ridge; the explicit bias feature is never penalized."""
    if moments.count < 2:
        raise ValueError("at least two training examples are required")
    if not math.isfinite(ridge_lambda) or ridge_lambda < 0.0:
        raise ValueError("ridge_lambda must be finite and nonnegative")
    count = float(moments.count)
    mean = moments.x_sum / count
    variance = np.diag(moments.xtx) / count - mean * mean
    variance = np.maximum(variance, 0.0)
    center = mean.copy()
    scale = np.sqrt(variance)
    constant = scale < 1e-12
    scale[constant] = 1.0
    # FEATURE_NAMES[0] is the explicit constant-one predictor.
    center[0] = 0.0
    scale[0] = 1.0

    numerator = (
        moments.xtx
        - np.outer(center, moments.x_sum)
        - np.outer(moments.x_sum, center)
        + count * np.outer(center, center)
    )
    ztz = numerator / np.outer(scale, scale)
    zty = (moments.xty - np.outer(center, moments.y_sum)) / scale[:, None]
    system = ztz.copy()
    penalty = ridge_lambda * count
    system.flat[:: system.shape[0] + 1] += penalty
    system[0, 0] -= penalty
    try:
        weights = np.linalg.solve(system, zty)
        solver = "solve"
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(system, zty, rcond=None)[0]
        solver = "lstsq"
    if not np.isfinite(weights).all():
        raise RuntimeError("ridge fit produced nonfinite coefficients")

    residual_squares = np.diag(
        moments.yty - 2.0 * weights.T @ zty + weights.T @ ztz @ weights
    )
    residual_squares = np.maximum(residual_squares, 0.0)
    total_squares = np.diag(moments.yty) - moments.y_sum * moments.y_sum / count
    train_rmse = np.sqrt(residual_squares / count)
    train_r2 = np.zeros_like(total_squares)
    np.divide(
        residual_squares,
        total_squares,
        out=train_r2,
        where=total_squares > 0.0,
    )
    train_r2 = np.where(total_squares > 0.0, 1.0 - train_r2, 0.0)
    condition_number = float(np.linalg.cond(system))
    return {
        "center": center,
        "scale": scale,
        "weights": weights,
        "solver": solver,
        "condition_number": condition_number if math.isfinite(condition_number) else None,
        "train_rmse": train_rmse,
        "train_r2": train_r2,
    }


class PredictionMetrics:
    def __init__(self, output_count: int = 2):
        self.count = 0
        self.layouts = 0
        self.absolute_error = np.zeros(output_count, dtype=np.float64)
        self.square_error = np.zeros(output_count, dtype=np.float64)
        self.target_sum = np.zeros(output_count, dtype=np.float64)
        self.target_square_sum = np.zeros(output_count, dtype=np.float64)
        # Prediction ties carry half an inversion. This is the usual expected
        # error under an unbiased deterministic tie-break and avoids reporting
        # a constant predictor as having perfect ordering.
        self.discordant_pairs = np.zeros(output_count, dtype=np.float64)
        self.comparable_pairs = np.zeros(output_count, dtype=np.int64)

    def update(self, predictions: np.ndarray, targets: np.ndarray):
        residual = predictions - targets
        self.count += targets.shape[0]
        self.absolute_error += np.abs(residual).sum(axis=0)
        self.square_error += (residual * residual).sum(axis=0)
        self.target_sum += targets.sum(axis=0)
        self.target_square_sum += (targets * targets).sum(axis=0)
        self.layouts += 1
        upper = np.triu(np.ones((targets.shape[0], targets.shape[0]), dtype=bool), 1)
        for output in range(targets.shape[1]):
            target_delta = targets[:, output, None] - targets[None, :, output]
            prediction_delta = (
                predictions[:, output, None] - predictions[None, :, output]
            )
            comparable = upper & (target_delta != 0.0)
            self.comparable_pairs[output] += int(comparable.sum())
            inverse = comparable & (target_delta * prediction_delta < 0.0)
            prediction_tie = comparable & (prediction_delta == 0.0)
            self.discordant_pairs[output] += float(inverse.sum())
            self.discordant_pairs[output] += 0.5 * float(prediction_tie.sum())

    def as_json(self):
        if self.count == 0:
            raise ValueError("validation partition contains no usable examples")
        count = float(self.count)
        total_squares = self.target_square_sum - self.target_sum * self.target_sum / count
        r2 = np.zeros_like(total_squares)
        np.divide(
            self.square_error,
            total_squares,
            out=r2,
            where=total_squares > 0.0,
        )
        r2 = np.where(total_squares > 0.0, 1.0 - r2, 0.0)
        return {
            "examples": self.count,
            "layouts": self.layouts,
            "mae": (self.absolute_error / count).tolist(),
            "rmse": np.sqrt(self.square_error / count).tolist(),
            "r2": r2.tolist(),
            "pairwise_inversion_fraction": np.divide(
                self.discordant_pairs,
                self.comparable_pairs,
                out=np.zeros_like(self.discordant_pairs, dtype=np.float64),
                where=self.comparable_pairs > 0,
            ).tolist(),
        }


def _selection_sha256(training, validation) -> str:
    digest = hashlib.sha256()
    for partition, rows in (("train", training), ("validation", validation)):
        for row in rows:
            digest.update(f"{partition}\0{row.relative_path}\n".encode("utf-8"))
    return digest.hexdigest()


def _partition_content_sha256(dataset, training, validation) -> str:
    """Bind the exact bytes of every selected training/validation source."""
    digest = hashlib.sha256()
    for partition, rows in (("train", training), ("validation", validation)):
        for row in rows:
            source = Path(dataset.all_files[row.file_index])
            digest.update(f"{partition}\0{row.relative_path}\0".encode("utf-8"))
            with source.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            digest.update(b"\xff")
    return digest.hexdigest()


def train_order_model(
    dataset,
    *,
    data_root: Path,
    holdout_manifest: Path | list[Path] | tuple[Path, ...],
    seed: int,
    validation_fraction: float,
    ridge_lambda: float,
    min_blocks: int,
    max_blocks: int,
    max_files: int | None,
    max_layouts_per_file: int | None,
    message_steps: int,
    progress_every_files: int = 0,
):
    if min_blocks < 1 or max_blocks < min_blocks:
        raise ValueError("invalid block-count range")
    if max_layouts_per_file is not None and max_layouts_per_file < 1:
        raise ValueError("max_layouts_per_file must be positive")
    if message_steps < 0:
        raise ValueError("message_steps must be nonnegative")

    holdout_paths = (
        [holdout_manifest]
        if isinstance(holdout_manifest, Path)
        else list(holdout_manifest)
    )
    holdout = load_holdout_sources(holdout_paths)
    training, validation, discovered_count = partition_source_files(
        dataset,
        data_root=data_root,
        excluded_sources=holdout.paths,
        seed=seed,
        validation_fraction=validation_fraction,
        max_files=max_files,
    )

    train_stats = ScanStats(source_files=len(training))
    moments = RidgeMoments(len(FEATURE_NAMES), len(TARGET_NAMES))
    for features, targets in stream_layouts(
        dataset,
        training,
        min_blocks=min_blocks,
        max_blocks=max_blocks,
        max_layouts_per_file=max_layouts_per_file,
        message_steps=message_steps,
        layout_seed=seed,
        stats=train_stats,
        progress_every_files=progress_every_files,
        partition_name="train",
    ):
        moments.update(features, targets)
    if moments.count == 0:
        raise ValueError("training partition contains no usable examples")
    fitted = fit_standardized_ridge(moments, ridge_lambda)

    validation_stats = ScanStats(source_files=len(validation))
    validation_metrics = PredictionMetrics(len(TARGET_NAMES))
    for features, targets in stream_layouts(
        dataset,
        validation,
        min_blocks=min_blocks,
        max_blocks=max_blocks,
        max_layouts_per_file=max_layouts_per_file,
        message_steps=message_steps,
        layout_seed=seed,
        stats=validation_stats,
        progress_every_files=progress_every_files,
        partition_name="validation",
    ):
        normalized = (features - fitted["center"]) / fitted["scale"]
        validation_metrics.update(normalized @ fitted["weights"], targets)

    model = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_type": "standardized_linear_ridge",
        "feature_schema": {
            "version": FEATURE_VERSION,
            "names": list(FEATURE_NAMES),
            "message_steps": message_steps,
        },
        "target_schema": {
            "names": list(TARGET_NAMES),
            "description": "golden block centers normalized independently to the golden layout bbox",
        },
        "normalization": {
            "center": fitted["center"].tolist(),
            "scale": fitted["scale"].tolist(),
        },
        "coefficients": fitted["weights"].tolist(),
        "training": {
            "trainer_version": TRAINER_VERSION,
            "ridge_lambda_mean_loss": ridge_lambda,
            "linear_solver": fitted["solver"],
            "condition_number": fitted["condition_number"],
            "seed": seed,
            "validation_fraction": validation_fraction,
            "block_count_range": [min_blocks, max_blocks],
            "max_files": max_files,
            "max_layouts_per_file": max_layouts_per_file,
            "example_weighting": "one_per_block",
            "source_split": "file_disjoint_stratified_by_block_count",
        },
        "provenance": {
            "trainer_sha256": _file_sha256(Path(__file__)),
            "feature_implementation_sha256": _file_sha256(
                SOLUTION_DIR / "learned_order.py"
            ),
            "dataset": "FloorSet-Lite",
            "data_root": _portable_path(data_root),
            "source_files_discovered": discovered_count,
            "source_files_excluded": len(holdout.paths),
            "holdout_manifests": [_portable_path(path) for path in holdout_paths],
            "holdout_manifest_sha256": holdout.manifest_sha256,
            "holdout_manifest_sha256s": list(holdout.manifest_sha256s),
            "holdout_manifest_schema_versions": list(
                holdout.manifest_schema_versions
            ),
            "holdout_split_unit": holdout.manifest_split_unit,
            "source_partition_sha256": _selection_sha256(training, validation),
            "source_partition_content_sha256": _partition_content_sha256(
                dataset, training, validation
            ),
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
        },
        "stats": {
            "train_scan": train_stats.as_json(),
            "train_fit": {
                "examples": moments.count,
                "rmse": fitted["train_rmse"].tolist(),
                "r2": fitted["train_r2"].tolist(),
            },
            "validation_scan": validation_stats.as_json(),
            "validation_fit": validation_metrics.as_json(),
        },
    }
    payload = json.dumps(model, sort_keys=True, separators=(",", ":"), allow_nan=False)
    model["payload_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return model


def _atomic_write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=FLOORSET)
    parser.add_argument(
        "--fold-manifest",
        type=Path,
        action="append",
        help=(
            "source-file holdout manifest; repeat to exclude the union "
            "(defaults to the clean and raw heavy panels)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "models" / "order_ridge_v4.json",
    )
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--ridge-lambda", type=float, default=1e-4)
    parser.add_argument("--min-blocks", type=int, default=100)
    parser.add_argument("--max-blocks", type=int, default=120)
    parser.add_argument(
        "--max-files",
        type=int,
        default=1024,
        help="deterministically select this many non-holdout files; 0 uses all",
    )
    parser.add_argument(
        "--max-layouts-per-file",
        type=int,
        default=24,
        help="layouts read from each selected source file; 0 uses the dataset value",
    )
    parser.add_argument("--message-steps", type=int, default=4)
    parser.add_argument("--progress-every-files", type=int, default=25)
    args = parser.parse_args()
    max_files = args.max_files or None
    max_layouts = args.max_layouts_per_file or None
    holdout_manifests = args.fold_manifest or [
        ROOT / "results" / "folds" / "heavy_clean_v1.json",
        ROOT / "results" / "folds" / "heavy_raw_hash_v1.json",
    ]

    dataset = FloorplanDatasetLite(str(args.data_root))
    model = train_order_model(
        dataset,
        data_root=args.data_root,
        holdout_manifest=holdout_manifests,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        ridge_lambda=args.ridge_lambda,
        min_blocks=args.min_blocks,
        max_blocks=args.max_blocks,
        max_files=max_files,
        max_layouts_per_file=max_layouts,
        message_steps=args.message_steps,
        progress_every_files=args.progress_every_files,
    )
    _atomic_write_json(args.output, model)
    validation = model["stats"]["validation_fit"]
    print(
        f"wrote {args.output}: {model['stats']['train_fit']['examples']} train blocks, "
        f"{validation['examples']} validation blocks, validation MAE={validation['mae']}, "
        f"pairwise inversion={validation['pairwise_inversion_fraction']}"
    )


if __name__ == "__main__":
    main()
