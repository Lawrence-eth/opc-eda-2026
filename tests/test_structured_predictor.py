import copy

import pytest

from contest_solution import structured_predictor as module


def _artifact(*, threshold=0.4):
    features = len(module.FEATURE_NAMES)
    hidden = 2
    model = {
        "schema_version": module.ARTIFACT_SCHEMA_VERSION,
        "model_type": module.MODEL_TYPE,
        "feature_schema": {
            "version": module.FEATURE_VERSION,
            "names": list(module.FEATURE_NAMES),
            "message_steps": 1,
        },
        "structured_schema": {
            "node_targets": list(module.NODE_TARGET_NAMES),
            "pair_heads": list(module.PAIR_HEAD_NAMES),
            "pair_direct_features": list(module.PAIR_DIRECT_FEATURE_NAMES),
            "max_shape_options": module.MAX_SHAPE_OPTIONS,
            "hidden_size": hidden,
            "pair_feature_count": 1 + 5 * hidden + len(module.PAIR_DIRECT_FEATURE_NAMES),
        },
        "normalization": {
            "center": [0.0] * features,
            "scale": [1.0] * features,
        },
        "hidden_projection": [[0.0] * hidden for _ in range(features)],
        "node_coefficients": [
            [0.0] * len(module.NODE_TARGET_NAMES) for _ in range(features)
        ],
        "pair_coefficients": [
            [0.0] * len(module.PAIR_HEAD_NAMES)
            for _ in range(1 + 5 * hidden + len(module.PAIR_DIRECT_FEATURE_NAMES))
        ],
        "calibration": {
            "confidence_threshold": threshold,
            "margin_scale": 1.0,
            "margin_bias": 0.0,
        },
        "provenance": {"fixture": True},
    }
    return module.seal_artifact(model)


def _case():
    return {
        "block_count": 2,
        "area_targets": [4.0, 4.0],
        "b2b_connectivity": [(0, 1, 1.0)],
        "p2b_connectivity": [],
        "pins_pos": [],
        "constraints": [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        "target_positions": [[-1, -1, -1, -1], [-1, -1, -1, -1]],
    }


def test_artifact_hash_and_schema_are_fail_closed():
    artifact = _artifact()
    checked = module.validate_artifact(artifact)
    assert checked["_validated"]["projection"][0] == (0.0, 0.0)

    corrupted = copy.deepcopy(artifact)
    corrupted["node_coefficients"][0][0] = 1.0
    with pytest.raises(ValueError, match="integrity"):
        module.validate_artifact(corrupted)

    unsupported = copy.deepcopy(artifact)
    unsupported["feature_schema"]["names"][0] = "source_file_id"
    unsupported = module.seal_artifact(unsupported)
    with pytest.raises(ValueError, match="feature names"):
        module.validate_artifact(unsupported)


def test_zero_model_projects_a_valid_tree_and_support_forest():
    prediction = module.predict_candidate(_artifact(), **_case())
    assert prediction.reason == "candidate"
    assert prediction.hard_feasible
    assert prediction.root == 0
    assert prediction.positions == (
        (0.0, 0.0, 2.0, 2.0),
        (2.0, 0.0, 2.0, 2.0),
    )
    assert prediction.confidence == pytest.approx(0.5)


def test_direct_pair_features_use_only_input_visible_relations():
    case = _case()
    case["constraints"] = [[0, 0, 7, 9, 1], [0, 0, 7, 9, 1]]
    features = module.extract_order_features(
        **case, message_steps=1
    )
    context = module.extract_pair_feature_context(
        case["block_count"],
        case["area_targets"],
        case["b2b_connectivity"],
        case["p2b_connectivity"],
        case["constraints"],
        features,
    )
    direct = dict(
        zip(module.PAIR_DIRECT_FEATURE_NAMES, module.direct_pair_features(context, 0, 1))
    )
    assert direct["has_b2b_edge"] == 1.0
    assert direct["log_b2b_weight_norm"] == pytest.approx(1.0)
    assert direct["same_mib_group"] == 1.0
    assert direct["same_cluster_group"] == 1.0
    assert direct["log_area_ratio"] == pytest.approx(0.0)
    assert direct["boundary_bit_compatibility"] == pytest.approx(1.0)


def test_confidence_and_validator_both_fail_closed_to_fallback():
    case = _case()
    fallback = [(0.0, 0.0, 4.0, 1.0), (0.0, 1.0, 4.0, 1.0)]

    result, prediction, accepted = module.predict_or_fallback(
        _artifact(threshold=0.6), fallback, **case
    )
    assert not accepted
    assert prediction.hard_feasible
    assert result == fallback

    result, _prediction, accepted = module.predict_or_fallback(
        _artifact(threshold=0.4), fallback, **case, validator=lambda _rows: False
    )
    assert not accepted
    assert result == fallback


def test_skyline_support_projection_prevents_horizontal_tree_collisions():
    case = {
        "block_count": 4,
        "area_targets": [4.0] * 4,
        "b2b_connectivity": [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)],
        "p2b_connectivity": [],
        "pins_pos": [],
        "constraints": [[0, 0, 0, 0, 0] for _ in range(4)],
        "target_positions": [[-1, -1, -1, -1] for _ in range(4)],
    }
    skyline = module.predict_candidate(_artifact(), **case, vertical_mode="skyline")
    learned = module.predict_candidate(_artifact(), **case, vertical_mode="learned")
    assert skyline.hard_feasible
    assert skyline.positions is not None
    assert learned.reason == "hard_infeasible"
    assert learned.positions is None

def test_hard_feasible_rejects_overlap_and_bad_preplaced_geometry():
    case = _case()
    assert not module.hard_feasible(
        [(0, 0, 2, 2), (1, 0, 2, 2)],
        case["area_targets"],
        case["constraints"],
        case["target_positions"],
    )
    constraints = [[0, 1, 0, 0, 0], [0, 0, 0, 0, 0]]
    targets = [[5, 6, 2, 2], [-1, -1, -1, -1]]
    assert not module.hard_feasible(
        [(0, 0, 2, 2), (2, 0, 2, 2)],
        case["area_targets"],
        constraints,
        targets,
    )
