import copy
from pathlib import Path

import pytest

from contest_solution import rank_mlp


def _artifact():
    dimensions = rank_mlp.ARCHITECTURE
    model = {
        "schema_version": rank_mlp.SCHEMA_VERSION,
        "model_type": rank_mlp.MODEL_TYPE,
        "feature_schema": {
            "version": rank_mlp.FEATURE_VERSION,
            "names": list(rank_mlp.FEATURE_NAMES),
            "message_steps": 4,
            "mib_policy": "mask_incompatible",
        },
        "target_schema": {"names": list(rank_mlp.TARGET_NAMES)},
        "architecture": list(dimensions),
        "normalization": {
            "center": [0.0] * dimensions[0],
            "scale": [1.0] * dimensions[0],
        },
        "layers": [
            {
                "weights": [[0.0] * output for _ in range(inputs)],
                "bias": [0.0] * output,
            }
            for inputs, output in zip(dimensions, dimensions[1:])
        ],
        "linear_skip": {
            "weights": [[0.0, 0.0] for _ in range(dimensions[0])],
            "bias": [0.25, 0.75],
        },
        "training": {"fixture": True},
        "provenance": {},
    }
    return rank_mlp.seal_artifact(model)


def test_artifact_is_hash_bound_and_predicts_without_numpy():
    artifact = _artifact()
    checked = rank_mlp.validate_artifact(artifact)
    features = [[1.0] + [0.0] * (len(rank_mlp.FEATURE_NAMES) - 1)]
    assert rank_mlp.artifact_predictions(
        features, checked, message_steps=4
    ) == [[0.25, 0.75]]

    corrupted = copy.deepcopy(artifact)
    corrupted["linear_skip"]["bias"][0] = 0.5
    with pytest.raises(ValueError, match="integrity"):
        rank_mlp.validate_artifact(corrupted)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda model: model.update(schema_version=99), "schema_version"),
        (
            lambda model: model["feature_schema"].update(mib_policy="unmasked"),
            "MIB policy",
        ),
        (lambda model: model.update(architecture=[60, 8, 2]), "architecture"),
        (
            lambda model: model["normalization"]["scale"].__setitem__(2, 0.0),
            "scale",
        ),
    ],
)
def test_schema_drift_fails_closed(mutation, message):
    artifact = _artifact()
    mutation(artifact)
    artifact = rank_mlp.seal_artifact(artifact)
    with pytest.raises(ValueError, match=message):
        rank_mlp.validate_artifact(artifact)


def test_inference_module_has_no_training_framework_dependency():
    source = Path(rank_mlp.__file__).read_text()
    assert "import numpy" not in source
    assert "import torch" not in source


def test_forged_validation_cache_cannot_bypass_integrity_check():
    artifact = _artifact()
    artifact["_validated"] = {"forged": True}
    artifact["linear_skip"]["bias"][0] = 9.0
    with pytest.raises(ValueError, match="integrity"):
        rank_mlp.artifact_predictions(
            [[0.0] * rank_mlp.ARCHITECTURE[0]], artifact, message_steps=4
        )


def test_message_depth_is_fixed_to_training_schema():
    artifact = _artifact()
    artifact["feature_schema"]["message_steps"] = 3
    artifact = rank_mlp.seal_artifact(artifact)
    with pytest.raises(ValueError, match="message_steps"):
        rank_mlp.validate_artifact(artifact)
