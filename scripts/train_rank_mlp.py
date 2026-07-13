#!/usr/bin/env python3
"""Train and gate a compact nonlinear coordinate-rank prior on cached labels.

The cache is opened read-only.  Source/file identities and structured labels
other than x/y fractional ranks and the per-layout MIB inconsistency mask are
never loaded into a model matrix.  The exact same validation features are used
to score embedded v5b and the MLP.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
SOLUTION_DIR = ROOT / "contest_solution"
sys.path.insert(0, str(SOLUTION_DIR))

from learned_order import (  # noqa: E402
    FEATURE_NAMES,
    FEATURE_VERSION,
    MIB_FEATURE_INDICES as INFERENCE_MIB_FEATURE_INDICES,
)
from order_model_v5b import MODEL as EMBEDDED_V5B  # noqa: E402
from rank_mlp import (  # noqa: E402
    ARCHITECTURE,
    MODEL_TYPE,
    SCHEMA_VERSION,
    TARGET_NAMES,
    seal_artifact,
)


TRAINER_VERSION = 1
CACHE_SCHEMA_VERSION = 2
MIB_FEATURE_INDICES = INFERENCE_MIB_FEATURE_INDICES
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
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


def load_cache_manifest(cache_dir: Path) -> dict[str, Any]:
    path = cache_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = manifest.pop("payload_sha256", None)
    actual = _canonical_sha256(manifest)
    manifest["payload_sha256"] = expected
    if expected != actual:
        raise ValueError("rank cache manifest payload hash mismatch")
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("rank cache schema is unsupported")
    if manifest.get("cache_type") != "dual_parent_supervision_shards":
        raise ValueError("rank cache type is unsupported")
    feature_schema = manifest.get("feature_schema", {})
    if feature_schema.get("version") != FEATURE_VERSION:
        raise ValueError("rank cache feature version is stale")
    if feature_schema.get("names") != list(FEATURE_NAMES):
        raise ValueError("rank cache feature names are stale")
    if feature_schema.get("message_steps") != 4:
        raise ValueError("rank cache message step count is unsupported")
    selection = manifest.get("selection", {})
    if selection.get("split_unit") != "source_file":
        raise ValueError("rank cache must be source-file-disjoint")
    if int(manifest.get("holdouts", {}).get("excluded_source_count", 0)) != 741:
        raise ValueError("rank cache does not bind the complete holdout source union")
    lowered = [name.lower() for name in FEATURE_NAMES]
    if any(token in name for token in FORBIDDEN_FEATURE_TOKENS for name in lowered):
        raise ValueError("rank cache exposes a prohibited identity feature")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("rank cache shard inventory is empty")
    selected_sources: dict[str, list[str]] = {"train": [], "validation": []}
    seen_sources: dict[str, set[str]] = {"train": set(), "validation": set()}
    for shard in shards:
        if not isinstance(shard, dict) or shard.get("partition") not in selected_sources:
            raise ValueError("rank cache shard partition is unsupported")
        partition = shard["partition"]
        identities = shard.get("identities")
        if not isinstance(identities, list) or len(identities) != shard.get("layouts"):
            raise ValueError("rank cache shard identity count is inconsistent")
        for identity in identities:
            source = identity.get("source_file") if isinstance(identity, dict) else None
            if not isinstance(source, str) or not source:
                raise ValueError("rank cache shard source identity is invalid")
            if source not in seen_sources[partition]:
                seen_sources[partition].add(source)
                selected_sources[partition].append(source)
        shard_path = cache_dir / shard["path"]
        if not shard_path.is_file() or _sha256(shard_path) != shard["sha256"]:
            raise ValueError(f"rank cache shard hash mismatch: {shard_path}")
    if seen_sources["train"] & seen_sources["validation"]:
        raise ValueError("rank cache train and validation sources overlap")
    expected_counts = selection.get("selected_source_counts", {})
    expected_hashes = selection.get("selected_source_sha256", {})
    for partition in selected_sources:
        if len(selected_sources[partition]) != expected_counts.get(partition):
            raise ValueError(f"rank cache {partition} source count is inconsistent")
        if _canonical_sha256(selected_sources[partition]) != expected_hashes.get(partition):
            raise ValueError(f"rank cache {partition} source hash is inconsistent")
    return manifest


class Partition:
    def __init__(self, features, targets, layouts, inconsistent_count):
        self.features = features
        self.targets = targets
        self.layouts = layouts
        self.inconsistent_count = inconsistent_count


def load_partition(
    cache_dir: Path, manifest: dict[str, Any], partition: str
) -> Partition:
    features = []
    targets = []
    layouts: list[tuple[int, int]] = []
    base = 0
    inconsistent_count = 0
    for shard in manifest["shards"]:
        if shard["partition"] != partition:
            continue
        with np.load(cache_dir / shard["path"], allow_pickle=False) as arrays:
            x = np.asarray(arrays["features"], dtype=np.float32).copy()
            y = np.column_stack((arrays["x_rank"], arrays["y_rank"])).astype(
                np.float32
            )
            offsets = arrays["layout_offsets"]
            inconsistent = arrays["mib_inconsistent"]
            if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
                raise ValueError("rank cache feature matrix shape is invalid")
            if y.shape != (len(x), len(TARGET_NAMES)):
                raise ValueError("rank cache target matrix shape is invalid")
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError("rank cache features and targets must be finite")
            if np.any(y < 0.0) or np.any(y > 1.0):
                raise ValueError("rank cache fractional ranks are out of range")
            if (
                offsets.ndim != 1
                or len(offsets) < 2
                or int(offsets[0]) != 0
                or int(offsets[-1]) != len(x)
                or np.any(np.diff(offsets) <= 0)
            ):
                raise ValueError("rank cache layout offsets are invalid")
            if len(offsets) - 1 != shard["layouts"] or len(inconsistent) != len(offsets) - 1:
                raise ValueError("rank cache layout metadata is inconsistent")
            if len(x) != shard["blocks"] or np.any((inconsistent != 0) & (inconsistent != 1)):
                raise ValueError("rank cache block or MIB metadata is inconsistent")
            for layout in range(len(offsets) - 1):
                start, end = int(offsets[layout]), int(offsets[layout + 1])
                if inconsistent[layout]:
                    x[start:end, MIB_FEATURE_INDICES] = 0.0
                    inconsistent_count += 1
                layouts.append((base + start, base + end))
            features.append(x)
            targets.append(y)
            base += x.shape[0]
    if not features or not layouts:
        raise ValueError(f"rank cache partition is empty: {partition}")
    return Partition(
        np.concatenate(features, axis=0),
        np.concatenate(targets, axis=0),
        layouts,
        inconsistent_count,
    )


def fractional_ranks(values: np.ndarray, tolerance: float = 1e-12) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=np.float64)
    denominator = max(1, len(values) - 1)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and abs(values[order[end]] - values[order[start]]) <= tolerance:
            end += 1
        rank = ((start + end - 1) / 2.0) / denominator
        result[order[start:end]] = rank
        start = end
    return result


def rank_metrics(
    predictions: np.ndarray, targets: np.ndarray, layouts: list[tuple[int, int]]
) -> dict[str, Any]:
    raw_error = np.abs(predictions - targets).sum(axis=0)
    rank_error = np.zeros(2, dtype=np.float64)
    discordant = np.zeros(2, dtype=np.float64)
    comparable = np.zeros(2, dtype=np.int64)
    for start, end in layouts:
        predicted_rank = np.column_stack(
            [fractional_ranks(predictions[start:end, axis]) for axis in range(2)]
        )
        target = targets[start:end]
        rank_error += np.abs(predicted_rank - target).sum(axis=0)
        upper = np.triu(np.ones((end - start, end - start), dtype=bool), 1)
        for axis in range(2):
            target_delta = target[:, axis, None] - target[None, :, axis]
            prediction_delta = (
                predicted_rank[:, axis, None] - predicted_rank[None, :, axis]
            )
            valid = upper & (target_delta != 0.0)
            comparable[axis] += int(valid.sum())
            discordant[axis] += float(
                (valid & (target_delta * prediction_delta < 0.0)).sum()
            )
            discordant[axis] += 0.5 * float(
                (valid & (prediction_delta == 0.0)).sum()
            )
    count = len(targets)
    return {
        "blocks": count,
        "layouts": len(layouts),
        "raw_mae": (raw_error / count).tolist(),
        "rank_mae": (rank_error / count).tolist(),
        "pairwise_inversion_fraction": (discordant / comparable).tolist(),
        "comparable_pairs": comparable.tolist(),
    }


def v5b_predictions(features: np.ndarray) -> np.ndarray:
    center = np.asarray(EMBEDDED_V5B["normalization"]["center"], dtype=np.float64)
    scale = np.asarray(EMBEDDED_V5B["normalization"]["scale"], dtype=np.float64)
    weights = np.asarray(EMBEDDED_V5B["coefficients"], dtype=np.float64)
    return ((features.astype(np.float64) - center) / scale) @ weights


class ResidualRankMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(ARCHITECTURE[0], ARCHITECTURE[1])
        self.hidden2 = nn.Linear(ARCHITECTURE[1], ARCHITECTURE[2])
        self.output = nn.Linear(ARCHITECTURE[2], ARCHITECTURE[3])
        self.skip = nn.Linear(ARCHITECTURE[0], ARCHITECTURE[3])

    def forward(self, features):
        hidden = F.relu(self.hidden1(features))
        hidden = F.relu(self.hidden2(hidden))
        return self.output(hidden) + self.skip(features)


def _normalization(features: np.ndarray):
    center = features.mean(axis=0, dtype=np.float64)
    scale = features.std(axis=0, dtype=np.float64)
    scale[scale < 1e-12] = 1.0
    center[0] = 0.0
    scale[0] = 1.0
    return center, scale


def _initialize_linear_skip(
    model: ResidualRankMLP,
    normalized: np.ndarray,
    targets: np.ndarray,
    ridge: float,
):
    xtx = normalized.T @ normalized
    xty = normalized.T @ targets
    system = xtx.copy()
    penalty = ridge * len(normalized)
    system.flat[:: system.shape[0] + 1] += penalty
    system[0, 0] -= penalty
    weights = np.linalg.solve(system, xty)
    with torch.no_grad():
        model.skip.weight.copy_(torch.from_numpy(weights.T.astype(np.float32)))
        model.skip.bias.zero_()
        model.output.weight.zero_()
        model.output.bias.zero_()


def _pair_indices(
    layout_lengths: list[int], epoch: int, batch_ordinal: int, pairs_per_block: int
):
    left_all = []
    right_all = []
    base = 0
    for layout_ordinal, count in enumerate(layout_lengths):
        if count < 2:
            raise ValueError("pairwise rank training requires at least two blocks per layout")
        generator = np.random.default_rng(
            20260713 + epoch * 1000003 + batch_ordinal * 1009 + layout_ordinal
        )
        pair_count = max(count, pairs_per_block * count)
        left = generator.integers(0, count, size=pair_count)
        right = generator.integers(0, count - 1, size=pair_count)
        right += right >= left
        left_all.append(left + base)
        right_all.append(right + base)
        base += count
    return np.concatenate(left_all), np.concatenate(right_all)


def _predict(model, features: torch.Tensor, chunk: int = 65536) -> np.ndarray:
    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), chunk):
            rows.append(model(features[start : start + chunk]).cpu().numpy())
    return np.concatenate(rows, axis=0)


def _gate(candidate: dict[str, Any], baseline: dict[str, Any]):
    candidate_mae = np.asarray(candidate["rank_mae"])
    baseline_mae = np.asarray(baseline["rank_mae"])
    candidate_inversion = np.asarray(candidate["pairwise_inversion_fraction"])
    baseline_inversion = np.asarray(baseline["pairwise_inversion_fraction"])
    mae_relative = 1.0 - float(candidate_mae.mean() / baseline_mae.mean())
    inversion_relative = 1.0 - float(
        candidate_inversion.mean() / baseline_inversion.mean()
    )
    passed = (
        np.all(candidate_mae <= baseline_mae + 0.0005)
        and np.all(candidate_inversion <= baseline_inversion + 0.0005)
        and mae_relative >= 0.03
        and inversion_relative >= 0.03
    )
    return {
        "passed": bool(passed),
        "policy": {
            "max_axis_regression": 0.0005,
            "minimum_mean_rank_mae_relative_improvement": 0.03,
            "minimum_mean_inversion_relative_improvement": 0.03,
        },
        "mean_rank_mae_relative_improvement": mae_relative,
        "mean_inversion_relative_improvement": inversion_relative,
    }


def train(args: argparse.Namespace):
    _validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    cache_dir = args.cache_dir.resolve()
    manifest = load_cache_manifest(cache_dir)
    train_partition = load_partition(cache_dir, manifest, "train")
    validation = load_partition(cache_dir, manifest, "validation")
    v5b_metrics = rank_metrics(
        v5b_predictions(validation.features),
        validation.targets,
        validation.layouts,
    )
    print("embedded v5b:", json.dumps(v5b_metrics, sort_keys=True), flush=True)

    center, scale = _normalization(train_partition.features)
    train_x_np = (
        (train_partition.features.astype(np.float64) - center) / scale
    ).astype(np.float32)
    validation_x_np = (
        (validation.features.astype(np.float64) - center) / scale
    ).astype(np.float32)
    train_x = torch.from_numpy(train_x_np)
    train_y = torch.from_numpy(train_partition.targets)
    validation_x = torch.from_numpy(validation_x_np)

    model = ResidualRankMLP()
    _initialize_linear_skip(
        model, train_x_np, train_partition.targets.astype(np.float64), args.ridge
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05
    )

    best_state = copy.deepcopy(model.state_dict())
    initial_predictions = _predict(model, validation_x)
    best_metrics = rank_metrics(
        initial_predictions, validation.targets, validation.layouts
    )
    best_objective = float(
        np.mean(best_metrics["rank_mae"])
        + np.mean(best_metrics["pairwise_inversion_fraction"])
    )
    best_epoch = 0
    history = []
    stale = 0
    for epoch in range(1, args.epochs + 1):
        order = np.random.default_rng(args.seed + epoch).permutation(
            len(train_partition.layouts)
        )
        model.train()
        losses = []
        for batch_ordinal, batch_start in enumerate(
            range(0, len(order), args.batch_layouts)
        ):
            layout_ids = order[batch_start : batch_start + args.batch_layouts]
            node_indices = []
            lengths = []
            for layout_id in layout_ids:
                start, end = train_partition.layouts[int(layout_id)]
                node_indices.extend(range(start, end))
                lengths.append(end - start)
            node_indices_tensor = torch.tensor(node_indices, dtype=torch.long)
            features = train_x[node_indices_tensor]
            targets = train_y[node_indices_tensor]
            predictions = model(features)
            regression = F.mse_loss(predictions, targets)
            left, right = _pair_indices(
                lengths, epoch, batch_ordinal, args.pairs_per_block
            )
            left_tensor = torch.from_numpy(left.astype(np.int64))
            right_tensor = torch.from_numpy(right.astype(np.int64))
            target_delta = targets[left_tensor] - targets[right_tensor]
            prediction_delta = predictions[left_tensor] - predictions[right_tensor]
            valid = target_delta.abs() > 1e-7
            signed_margin = torch.sign(target_delta[valid]) * prediction_delta[valid]
            pairwise = (
                F.softplus(-signed_margin / args.pair_temperature).mean()
                if signed_margin.numel()
                else predictions.sum() * 0.0
            )
            loss = regression + args.pair_weight * pairwise
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        predictions = _predict(model, validation_x)
        metrics = rank_metrics(predictions, validation.targets, validation.layouts)
        objective = float(
            np.mean(metrics["rank_mae"])
            + np.mean(metrics["pairwise_inversion_fraction"])
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "objective": objective,
            "rank_mae": metrics["rank_mae"],
            "pairwise_inversion_fraction": metrics[
                "pairwise_inversion_fraction"
            ],
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if objective < best_objective - 1e-7:
            best_objective = objective
            best_metrics = metrics
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch >= args.minimum_epochs and stale >= args.patience:
            break

    model.load_state_dict(best_state)
    final_predictions = _predict(model, validation_x)
    best_metrics = rank_metrics(
        final_predictions, validation.targets, validation.layouts
    )
    gate = _gate(best_metrics, v5b_metrics)

    def layer(linear: nn.Linear):
        return {
            "weights": linear.weight.detach().cpu().numpy().T.tolist(),
            "bias": linear.bias.detach().cpu().numpy().tolist(),
        }

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "model_type": MODEL_TYPE,
        "feature_schema": {
            "version": FEATURE_VERSION,
            "names": list(FEATURE_NAMES),
            "message_steps": 4,
            "mib_policy": "mask_incompatible",
        },
        "target_schema": {
            "names": list(TARGET_NAMES),
            "description": "fractional within-layout golden center ranks",
        },
        "architecture": list(ARCHITECTURE),
        "normalization": {"center": center.tolist(), "scale": scale.tolist()},
        "layers": [layer(model.hidden1), layer(model.hidden2), layer(model.output)],
        "linear_skip": layer(model.skip),
        "training": {
            "trainer_version": TRAINER_VERSION,
            "seed": args.seed,
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "ridge_skip_initialization": args.ridge,
            "rank_regression_loss": "mean_squared_error",
            "pairwise_loss": "sampled_softplus_signed_order",
            "pair_weight": args.pair_weight,
            "pair_temperature": args.pair_temperature,
            "pairs_per_block": args.pairs_per_block,
            "batch_layouts": args.batch_layouts,
            "epochs_requested": args.epochs,
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "mib_feature_policy": "zero_mib_channels_on_cache_inconsistent_layouts",
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "deterministic_algorithms": True,
        },
        "validation": {
            "partition": "same_exact_cache_for_candidate_and_embedded_v5b",
            "candidate": best_metrics,
            "embedded_v5b": v5b_metrics,
            "gate": gate,
            "history": history,
        },
        "provenance": {
            "cache_manifest_payload_sha256": manifest["payload_sha256"],
            "cache_holdout_aggregate_sha256": manifest["holdouts"][
                "aggregate_sha256"
            ],
            "cache_excluded_sources": manifest["holdouts"][
                "excluded_source_count"
            ],
            "cache_train_sources": manifest["selection"][
                "selected_source_counts"
            ]["train"],
            "cache_validation_sources": manifest["selection"][
                "selected_source_counts"
            ]["validation"],
            "cache_train_layouts": manifest["stats"]["train"][
                "layouts_accepted"
            ],
            "cache_validation_layouts": manifest["stats"]["validation"][
                "layouts_accepted"
            ],
            "cache_access": "read_only_no_sealed_labels_or_scores",
            "fold4_access": "none",
            "embedded_v5b_payload_sha256": EMBEDDED_V5B["payload_sha256"],
            "embedded_v5b_module_sha256": _sha256(
                SOLUTION_DIR / "order_model_v5b.py"
            ),
            "trainer_sha256": _sha256(Path(__file__)),
            "inference_sha256": _sha256(SOLUTION_DIR / "rank_mlp.py"),
            "prohibited_model_inputs": list(FORBIDDEN_FEATURE_TOKENS),
        },
    }
    artifact = seal_artifact(artifact)
    return artifact


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(
            "/tmp/opc-structured-predictor/results/work/structured_cache_v2"
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "models" / "rank_mlp_v6.json"
    )
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--minimum-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-layouts", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--pair-weight", type=float, default=0.05)
    parser.add_argument("--pair-temperature", type=float, default=0.12)
    parser.add_argument("--pairs-per-block", type=int, default=2)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if not 1 <= args.minimum_epochs <= args.epochs:
        raise ValueError("minimum_epochs must be within requested epochs")
    if args.patience < 1 or args.batch_layouts < 1 or args.pairs_per_block < 1:
        raise ValueError("patience, batch_layouts, and pairs_per_block must be positive")
    positive = {
        "learning_rate": args.learning_rate,
        "pair_temperature": args.pair_temperature,
    }
    nonnegative = {
        "weight_decay": args.weight_decay,
        "ridge": args.ridge,
        "pair_weight": args.pair_weight,
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in positive.values()):
        raise ValueError("learning_rate and pair_temperature must be finite and positive")
    if any(not math.isfinite(value) or value < 0.0 for value in nonnegative.values()):
        raise ValueError("weight_decay, ridge, and pair_weight must be finite and nonnegative")


def main():
    args = _parser().parse_args()
    artifact = train(args)
    _atomic_json(args.output, artifact)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "best_epoch": artifact["training"]["best_epoch"],
                "candidate": artifact["validation"]["candidate"],
                "embedded_v5b": artifact["validation"]["embedded_v5b"],
                "gate": artifact["validation"]["gate"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
