"""Pure-stdlib inference for a compact nonlinear coordinate-rank prior."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from pathlib import Path
from typing import Any

try:
    from .learned_order import (
        FEATURE_NAMES,
        FEATURE_VERSION,
        apply_mib_feature_policy,
        extract_order_features,
    )
except ImportError:  # flat packaged-solver imports
    from learned_order import (  # type: ignore
        FEATURE_NAMES,
        FEATURE_VERSION,
        apply_mib_feature_policy,
        extract_order_features,
    )


SCHEMA_VERSION = 1
MODEL_TYPE = "standardized_residual_relu_rank_mlp"
ARCHITECTURE = (len(FEATURE_NAMES), 24, 12, 2)
TARGET_NAMES = ("fractional_center_x_rank", "fractional_center_y_rank")
MESSAGE_STEPS = 4


def _number(value: Any, name: str) -> float:
    if hasattr(value, "item"):
        value = value.item()
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must contain finite values")
    return result


def _matrix(value: Any, rows: int, columns: int, name: str):
    if not isinstance(value, (list, tuple)) or len(value) != rows:
        raise ValueError(f"{name} must have {rows} rows")
    result = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != columns:
            raise ValueError(f"{name} must have {columns} columns")
        result.append(tuple(_number(item, name) for item in row))
    return tuple(result)


def _vector(value: Any, count: int, name: str):
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(f"{name} must have {count} entries")
    return tuple(_number(item, name) for item in value)


def payload_sha256(model: dict[str, Any]) -> str:
    payload = dict(model)
    payload.pop("payload_sha256", None)
    payload.pop("_validated", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def seal_artifact(model: dict[str, Any]) -> dict[str, Any]:
    sealed = dict(model)
    sealed.pop("payload_sha256", None)
    sealed.pop("_validated", None)
    sealed["payload_sha256"] = payload_sha256(sealed)
    return sealed


def validate_artifact(model: Any) -> dict[str, Any]:
    if not isinstance(model, dict):
        raise ValueError("rank MLP artifact must be an object")
    if model.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("rank MLP schema_version is unsupported")
    if model.get("model_type") != MODEL_TYPE:
        raise ValueError("rank MLP model_type is unsupported")
    schema = model.get("feature_schema")
    if not isinstance(schema, dict):
        raise ValueError("rank MLP feature_schema must be an object")
    if schema.get("version") != FEATURE_VERSION:
        raise ValueError("rank MLP feature version is unsupported")
    if schema.get("names") != list(FEATURE_NAMES):
        raise ValueError("rank MLP feature names do not match inference")
    if schema.get("mib_policy") != "mask_incompatible":
        raise ValueError("rank MLP MIB policy is unsupported")
    if schema.get("message_steps") != MESSAGE_STEPS:
        raise ValueError("rank MLP message_steps is unsupported")
    target = model.get("target_schema")
    if not isinstance(target, dict) or target.get("names") != list(TARGET_NAMES):
        raise ValueError("rank MLP targets are unsupported")
    if model.get("architecture") != list(ARCHITECTURE):
        raise ValueError("rank MLP architecture is unsupported")

    normalization = model.get("normalization")
    if not isinstance(normalization, dict):
        raise ValueError("rank MLP normalization must be an object")
    center = _vector(normalization.get("center"), ARCHITECTURE[0], "center")
    scale = _vector(normalization.get("scale"), ARCHITECTURE[0], "scale")
    if any(value <= 0.0 for value in scale):
        raise ValueError("rank MLP scale must be positive")

    layers = model.get("layers")
    if not isinstance(layers, list) or len(layers) != 3:
        raise ValueError("rank MLP must contain three layers")
    parsed_layers = []
    for index, (inputs, outputs) in enumerate(zip(ARCHITECTURE, ARCHITECTURE[1:])):
        layer = layers[index]
        if not isinstance(layer, dict):
            raise ValueError("rank MLP layer must be an object")
        parsed_layers.append(
            (
                _matrix(layer.get("weights"), inputs, outputs, f"layer {index} weights"),
                _vector(layer.get("bias"), outputs, f"layer {index} bias"),
            )
        )
    skip = model.get("linear_skip")
    if not isinstance(skip, dict):
        raise ValueError("rank MLP linear_skip must be an object")
    skip_weights = _matrix(
        skip.get("weights"), ARCHITECTURE[0], ARCHITECTURE[-1], "skip weights"
    )
    skip_bias = _vector(skip.get("bias"), ARCHITECTURE[-1], "skip bias")

    expected = model.get("payload_sha256")
    if not (
        isinstance(expected, str)
        and len(expected) == 64
        and all(character in "0123456789abcdef" for character in expected)
    ):
        raise ValueError("rank MLP payload_sha256 is invalid")
    actual = payload_sha256(model)
    if not hmac.compare_digest(expected, actual):
        raise ValueError("rank MLP payload_sha256 integrity check failed")

    checked = dict(model)
    checked["_validated"] = {
        "center": center,
        "scale": scale,
        "layers": tuple(parsed_layers),
        "skip_weights": skip_weights,
        "skip_bias": skip_bias,
    }
    return checked


def load_artifact(path: str | Path) -> dict[str, Any]:
    return validate_artifact(json.loads(Path(path).read_text(encoding="utf-8")))


def compile_artifact(model: dict[str, Any]) -> dict[str, Any]:
    """Validate/hash an artifact once and retain only immutable inference data."""
    checked = validate_artifact(model)
    parsed = checked["_validated"]
    return {
        "compiled_type": MODEL_TYPE,
        "message_steps": checked["feature_schema"]["message_steps"],
        "mib_policy": checked["feature_schema"]["mib_policy"],
        "center": parsed["center"],
        "inverse_scale": tuple(1.0 / value for value in parsed["scale"]),
        "layers": parsed["layers"],
        "skip_weights": parsed["skip_weights"],
        "skip_bias": parsed["skip_bias"],
    }


def _dense(row, weights, bias, *, relu):
    outputs = []
    for output in range(len(bias)):
        value = bias[output] + sum(
            row[index] * weights[index][output] for index in range(len(row))
        )
        outputs.append(max(0.0, value) if relu else value)
    return outputs


def compiled_predictions(features: Any, compiled_model: Any, *, message_steps: int):
    """Apply a once-validated artifact without reparsing or rehashing JSON."""
    if not isinstance(compiled_model, dict):
        raise ValueError("compiled rank MLP must be an object")
    try:
        compiled_type = compiled_model["compiled_type"]
        artifact_steps = compiled_model["message_steps"]
        center = compiled_model["center"]
        inverse_scale = compiled_model["inverse_scale"]
        layers = compiled_model["layers"]
        skip_weights = compiled_model["skip_weights"]
        skip_bias = compiled_model["skip_bias"]
    except (KeyError, TypeError) as exc:
        raise ValueError("compiled rank MLP is malformed") from exc
    if compiled_type != MODEL_TYPE:
        raise ValueError("compiled rank MLP type is unsupported")
    if artifact_steps != message_steps:
        raise ValueError("rank MLP message_steps does not match inference")
    if (
        len(center) != ARCHITECTURE[0]
        or len(inverse_scale) != ARCHITECTURE[0]
        or len(layers) != 3
        or len(skip_weights) != ARCHITECTURE[0]
        or len(skip_bias) != ARCHITECTURE[-1]
    ):
        raise ValueError("compiled rank MLP dimensions do not match inference")
    for layer, inputs, outputs in zip(layers, ARCHITECTURE, ARCHITECTURE[1:]):
        if (
            not isinstance(layer, (list, tuple))
            or len(layer) != 2
            or len(layer[0]) != inputs
            or any(len(row) != outputs for row in layer[0])
            or len(layer[1]) != outputs
        ):
            raise ValueError("compiled rank MLP layer dimensions do not match inference")
    if any(len(row) != ARCHITECTURE[-1] for row in skip_weights):
        raise ValueError("compiled rank MLP skip dimensions do not match inference")
    predictions = []
    for feature_row in features:
        if len(feature_row) != ARCHITECTURE[0]:
            raise ValueError("rank MLP feature width is invalid")
        row = [
            (_number(value, "features") - center[index]) * inverse_scale[index]
            for index, value in enumerate(feature_row)
        ]
        hidden = row
        for index, (weights, bias) in enumerate(layers):
            hidden = _dense(hidden, weights, bias, relu=index < 2)
        skip = _dense(
            row, skip_weights, skip_bias, relu=False
        )
        output = [hidden[index] + skip[index] for index in range(ARCHITECTURE[-1])]
        if not all(math.isfinite(value) for value in output):
            raise ValueError("rank MLP prediction is non-finite")
        predictions.append(output)
    return predictions


def artifact_predictions(features: Any, model: dict[str, Any], *, message_steps: int):
    # Never trust a caller-provided private parse cache. Compile from the
    # content-bound artifact before entering the fast inference path.
    return compiled_predictions(
        features, compile_artifact(model), message_steps=message_steps
    )


def extract_compiled_rank_predictions(
    block_count,
    area_targets,
    b2b_connectivity,
    p2b_connectivity,
    pins_pos,
    constraints,
    target_positions,
    compiled_model,
):
    if not isinstance(compiled_model, dict):
        raise ValueError("compiled rank MLP must be an object")
    try:
        message_steps = compiled_model["message_steps"]
        mib_policy = compiled_model["mib_policy"]
    except KeyError as exc:
        raise ValueError("compiled rank MLP is malformed") from exc
    features = extract_order_features(
        block_count,
        area_targets,
        b2b_connectivity,
        p2b_connectivity,
        pins_pos,
        constraints,
        target_positions,
        message_steps=message_steps,
    )
    features, metadata = apply_mib_feature_policy(
        features,
        policy=mib_policy,
        block_count=block_count,
        area_targets=area_targets,
        constraints=constraints,
        target_positions=target_positions,
    )
    return compiled_predictions(
        features, compiled_model, message_steps=message_steps
    ), metadata


def extract_rank_predictions(
    block_count,
    area_targets,
    b2b_connectivity,
    p2b_connectivity,
    pins_pos,
    constraints,
    target_positions,
    model,
):
    return extract_compiled_rank_predictions(
        block_count,
        area_targets,
        b2b_connectivity,
        p2b_connectivity,
        pins_pos,
        constraints,
        target_positions,
        compile_artifact(model),
    )
