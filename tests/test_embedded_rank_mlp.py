import hashlib
import importlib.util
import json
from pathlib import Path

from contest_solution import rank_mlp


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "results" / "models" / "rank_mlp_v6_validation.json"
MODULE_PATH = ROOT / "contest_solution" / "rank_model_v6_validation.py"


def _load_generated_model():
    spec = importlib.util.spec_from_file_location("rank_model_v6_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MODEL


def test_embedded_v6_exactly_matches_content_bound_validation_artifact():
    artifact = json.loads(MODEL_PATH.read_bytes())
    embedded = _load_generated_model()
    assert embedded == artifact
    checked = rank_mlp.validate_artifact(embedded)
    assert checked["validation"]["gate"]["passed"] is True
    assert checked["training"]["best_epoch"] == 30
    assert checked["provenance"]["cache_excluded_sources"] == 741

    payload = dict(embedded)
    expected = payload.pop("payload_sha256")
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    assert hashlib.sha256(canonical.encode()).hexdigest() == expected


def test_embedded_v6_predicts_with_finite_stdlib_outputs():
    embedded = _load_generated_model()
    features = [[1.0] + [0.0] * (rank_mlp.ARCHITECTURE[0] - 1)]
    prediction = rank_mlp.artifact_predictions(features, embedded, message_steps=4)
    assert len(prediction) == 1
    assert len(prediction[0]) == 2
    assert all(rank_mlp.math.isfinite(value) for value in prediction[0])
