import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "learned_order", ROOT / "contest_solution" / "learned_order.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _case():
    return {
        "block_count": 5,
        "area_targets": [4.0, 9.0, 16.0, 25.0, 36.0],
        "b2b_connectivity": [
            (0, 1, 2.0),
            (1, 2, 1.0),
            (2, 3, 3.0),
            (3, 4, 1.5),
            (0, 4, 0.5),
            (-1, -1, -1.0),
        ],
        "p2b_connectivity": [
            (0, 0, 1.0),
            (1, 0, 3.0),
            (2, 2, 2.0),
            (3, 4, 1.0),
            (-1, -1, 0.0),
        ],
        "pins_pos": [(10.0, 20.0), (40.0, 5.0), (20.0, 30.0), (18.0, 16.0)],
        "constraints": [
            [0, 0, 7, 5, 1],
            [1, 0, 7, 0, 0],
            [0, 1, 0, 5, 8],
            [0, 1, 11, 9, 0],
            [0, 0, 11, 9, 4],
        ],
        "target_positions": [
            [-1, -1, -1, -1],
            [-1, -1, 3, 3],
            [5, 6, 4, 4],
            [15, 8, 5, 5],
            [-1, -1, -1, -1],
        ],
        "message_steps": 4,
    }


def _extract(case):
    return MODULE.extract_order_features(**case)


def _column(features, name):
    index = MODULE.FEATURE_NAMES.index(name)
    return [row[index] for row in features]


def _seal_artifact(model):
    model.pop("payload_sha256", None)
    payload = json.dumps(
        model, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    model["payload_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return model


def _artifact():
    feature_count = len(MODULE.FEATURE_NAMES)
    return _seal_artifact(
        {
            "schema_version": MODULE.MODEL_SCHEMA_VERSION,
            "model_type": MODULE.MODEL_TYPE,
            "feature_schema": {
                "version": MODULE.FEATURE_VERSION,
                "names": list(MODULE.FEATURE_NAMES),
                "message_steps": 4,
            },
            "target_schema": {"names": list(MODULE.TARGET_NAMES)},
            "normalization": {
                "center": [0.0] * feature_count,
                "scale": [1.0] * feature_count,
            },
            "coefficients": [[2.0, -1.0]]
            + [[0.0, 0.0] for _ in range(feature_count - 1)],
        }
    )


def test_features_are_finite_and_have_stable_v3_schema():
    features = _extract(_case())
    assert MODULE.FEATURE_VERSION == 3
    assert len(MODULE.FEATURE_NAMES) == 60
    assert len(set(MODULE.FEATURE_NAMES)) == len(MODULE.FEATURE_NAMES)
    assert len(features) == 5
    assert all(len(row) == len(MODULE.FEATURE_NAMES) for row in features)
    assert _column(features, "boundary_left")[0] == 1.0
    assert _column(features, "preplaced")[2] == 1.0
    assert _column(features, "hard_width_norm")[2] > 0.0
    assert all(math.isfinite(value) for row in features for value in row)
    assert not any("_id" in name for name in MODULE.FEATURE_NAMES)


def test_weighted_pin_statistics_and_hyperedge_aggregates():
    features = _extract(_case())
    row = dict(zip(MODULE.FEATURE_NAMES, features[0]))
    assert row["pin_centroid_x_norm"] == pytest.approx(32.5 / 40.0)
    assert row["pin_centroid_y_norm"] == pytest.approx(8.75 / 30.0)
    assert row["pin_median_x_norm"] == pytest.approx(40.0 / 40.0)
    assert row["pin_median_y_norm"] == pytest.approx(5.0 / 30.0)
    assert row["pin_mad_x_norm"] == pytest.approx(7.5 / 40.0)
    assert row["pin_mad_y_norm"] == pytest.approx(3.75 / 30.0)

    assert row["mib_group_size_fraction"] == pytest.approx(2.0 / 5.0)
    assert row["mib_group_area_fraction"] == pytest.approx(13.0 / 90.0)
    assert row["mib_member_area_share"] == pytest.approx(4.0 / 13.0)
    assert row["mib_group_boundary_fraction"] == pytest.approx(0.5)
    assert row["mib_internal_degree_fraction"] == pytest.approx(2.0 / 2.5)
    assert row["cluster_group_size_fraction"] == pytest.approx(2.0 / 5.0)


def test_block_permutation_is_equivariant():
    case = _case()
    original = _extract(case)
    # new block i contains old block permutation[i]
    permutation = [2, 4, 0, 3, 1]
    inverse = {old: new for new, old in enumerate(permutation)}
    permuted = copy.deepcopy(case)
    permuted["area_targets"] = [case["area_targets"][old] for old in permutation]
    permuted["constraints"] = [case["constraints"][old] for old in permutation]
    permuted["target_positions"] = [
        case["target_positions"][old] for old in permutation
    ]
    permuted["b2b_connectivity"] = [
        (inverse[first], inverse[second], weight)
        if weight > 0.0
        else (first, second, weight)
        for first, second, weight in case["b2b_connectivity"]
    ]
    permuted["p2b_connectivity"] = [
        (pin, inverse[block], weight)
        if weight > 0.0
        else (pin, block, weight)
        for pin, block, weight in case["p2b_connectivity"]
    ]

    transformed = _extract(permuted)
    for new, old in enumerate(permutation):
        assert transformed[new] == pytest.approx(original[old], abs=1e-12)


def test_group_identifier_renaming_is_invariant():
    case = _case()
    renamed = copy.deepcopy(case)
    mib_names = {0: 0, 7: 101, 11: 3}
    cluster_names = {0: 0, 5: 9001, 9: 2}
    for row in renamed["constraints"]:
        row[2] = mib_names[row[2]]
        row[3] = cluster_names[row[3]]
    for renamed_row, original_row in zip(_extract(renamed), _extract(case)):
        assert renamed_row == pytest.approx(original_row, abs=1e-12)


def test_mib_compatibility_and_mask_policy_use_only_input_visible_targets():
    case = _case()
    case["constraints"][1][0] = 0
    case["constraints"][4][2] = 0
    case["area_targets"][0] = 4.0
    case["area_targets"][1] = 4.02
    assert MODULE.mib_is_input_compatible(
        case["block_count"],
        case["area_targets"],
        case["constraints"],
        case["target_positions"],
    )
    case["area_targets"][1] = 9.0
    assert not MODULE.mib_is_input_compatible(
        case["block_count"],
        case["area_targets"],
        case["constraints"],
        case["target_positions"],
    )
    features = [[1.0] * len(MODULE.FEATURE_NAMES) for _ in range(case["block_count"])]
    masked, metadata = MODULE.apply_mib_feature_policy(
        features,
        policy="mask_incompatible",
        block_count=case["block_count"],
        area_targets=case["area_targets"],
        constraints=case["constraints"],
        target_positions=case["target_positions"],
    )
    assert metadata == {"input_compatible": False, "masked": True}
    assert all(
        row[index] == 0.0
        for row in masked
        for index in MODULE.MIB_FEATURE_INDICES
    )
    assert features[0][MODULE.MIB_FEATURE_INDICES[0]] == 1.0


def test_uniform_geometric_scale_is_invariant():
    case = _case()
    scale = 7.25
    scaled = copy.deepcopy(case)
    scaled["area_targets"] = [area * scale * scale for area in case["area_targets"]]
    scaled["pins_pos"] = [
        (x * scale, y * scale) for x, y in case["pins_pos"]
    ]
    scaled["target_positions"] = [
        [value * scale if value >= 0.0 else value for value in row]
        for row in case["target_positions"]
    ]
    for scaled_row, original_row in zip(_extract(scaled), _extract(case)):
        assert scaled_row == pytest.approx(original_row, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda case: case.update(block_count=0), "block_count"),
        (lambda case: case["area_targets"].__setitem__(0, 0.0), "positive"),
        (lambda case: case["area_targets"].__setitem__(0, math.nan), "finite"),
        (lambda case: case["constraints"][0].__setitem__(2, 1.5), "integer"),
        (lambda case: case["constraints"][0].__setitem__(4, 16), "low four"),
        (lambda case: case["constraints"].__setitem__(0, [0, 0, 0]), "width"),
        (lambda case: case["target_positions"][2].__setitem__(0, -1), "non-negative"),
        (
            lambda case: case["b2b_connectivity"].append((0, 99, 1.0)),
            "invalid endpoint",
        ),
        (lambda case: case["pins_pos"].append((1.0,)), "two values"),
        (lambda case: case.update(message_steps=17), "message_steps"),
    ],
)
def test_input_validation_rejects_malformed_cases(mutation, message):
    case = _case()
    mutation(case)
    with pytest.raises(ValueError, match=message):
        _extract(case)


def test_linear_predictions_match_hand_computation():
    feature_count = len(MODULE.FEATURE_NAMES)
    features = [
        [1.0] + [0.0] * (feature_count - 1),
        [1.0, 2.0] + [0.0] * (feature_count - 2),
    ]
    weights = [[3.0, -1.0], [4.0, 2.0]] + [
        [0.0, 0.0] for _ in range(feature_count - 2)
    ]
    assert MODULE.linear_predictions(features, weights) == [[3.0, -1.0], [11.0, 3.0]]


def test_linear_predictions_reject_ragged_inputs():
    feature_count = len(MODULE.FEATURE_NAMES)
    features = [[1.0] + [0.0] * (feature_count - 1)]
    weights = [[1.0, 2.0] for _ in range(feature_count)]
    weights[-1] = [1.0]
    with pytest.raises(ValueError, match="uniform"):
        MODULE.linear_predictions(features, weights)


def test_standardized_predictions_apply_center_and_scale():
    feature_count = len(MODULE.FEATURE_NAMES)
    features = [[3.0, 5.0] + [0.0] * (feature_count - 2)]
    center = [1.0, 1.0] + [0.0] * (feature_count - 2)
    scale = [2.0, 4.0] + [1.0] * (feature_count - 2)
    weights = [[2.0], [3.0]] + [[0.0] for _ in range(feature_count - 2)]
    assert MODULE.standardized_linear_predictions(
        features, center, scale, weights
    ) == [[5.0]]


def test_artifact_predictions_bind_message_steps_and_schema():
    feature_count = len(MODULE.FEATURE_NAMES)
    features = [[1.0] + [0.0] * (feature_count - 1)]
    model = _artifact()
    assert MODULE.artifact_predictions(features, model, message_steps=4) == [
        [2.0, -1.0]
    ]
    with pytest.raises(ValueError, match="message_steps"):
        MODULE.artifact_predictions(features, model, message_steps=3)


def test_extract_artifact_predictions_applies_declared_mib_policy():
    case = _case()
    case["constraints"][1][0] = 0
    case["area_targets"][0] = 4.0
    case["area_targets"][1] = 9.0
    model = _artifact()
    model["feature_schema"]["mib_policy"] = "mask_incompatible"
    _seal_artifact(model)
    predictions, metadata = MODULE.extract_artifact_predictions(
        case["block_count"],
        case["area_targets"],
        case["b2b_connectivity"],
        case["p2b_connectivity"],
        case["pins_pos"],
        case["constraints"],
        case["target_positions"],
        model,
    )
    assert len(predictions) == case["block_count"]
    assert metadata == {"input_compatible": False, "masked": True}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda model: model.update(schema_version=99), "schema_version"),
        (lambda model: model.update(model_type="unknown"), "model_type"),
        (
            lambda model: model["feature_schema"].update(
                version=MODULE.FEATURE_VERSION + 1
            ),
            "feature version",
        ),
        (
            lambda model: model["feature_schema"]["names"].__setitem__(0, "wrong"),
            "feature names",
        ),
        (
            lambda model: model["target_schema"].update(names=["only_x"]),
            "target names",
        ),
        (
            lambda model: model["coefficients"][0].pop(),
            "output width",
        ),
        (
            lambda model: model["coefficients"].pop(),
            "feature count",
        ),
    ],
)
def test_artifact_predictions_reject_schema_and_width_mismatches(mutation, message):
    feature_count = len(MODULE.FEATURE_NAMES)
    features = [[1.0] + [0.0] * (feature_count - 1)]
    model = _artifact()
    mutation(model)
    _seal_artifact(model)
    with pytest.raises(ValueError, match=message):
        MODULE.artifact_predictions(features, model, message_steps=4)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda model: model["normalization"]["center"].__setitem__(0, math.nan),
            "finite",
        ),
        (
            lambda model: model["normalization"]["scale"].__setitem__(0, math.inf),
            "finite",
        ),
        (
            lambda model: model["normalization"]["scale"].__setitem__(0, 0.0),
            "positive",
        ),
        (
            lambda model: model["coefficients"][0].__setitem__(0, math.inf),
            "finite",
        ),
    ],
)
def test_artifact_predictions_reject_nonfinite_parameters(mutation, message):
    feature_count = len(MODULE.FEATURE_NAMES)
    features = [[1.0] + [0.0] * (feature_count - 1)]
    model = _artifact()
    mutation(model)
    with pytest.raises(ValueError, match=message):
        MODULE.artifact_predictions(features, model, message_steps=4)


def test_artifact_predictions_verify_payload_integrity():
    feature_count = len(MODULE.FEATURE_NAMES)
    features = [[1.0] + [0.0] * (feature_count - 1)]
    model = _artifact()
    model["coefficients"][0][0] = 3.0
    with pytest.raises(ValueError, match="integrity"):
        MODULE.artifact_predictions(features, model, message_steps=4)

    model = _artifact()
    model["payload_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="payload_sha256"):
        MODULE.artifact_predictions(features, model, message_steps=4)
