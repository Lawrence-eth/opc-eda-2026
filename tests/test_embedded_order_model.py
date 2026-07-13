import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "results" / "models" / "order_ridge_v5b_clean_raw.json"
MODULE_PATH = ROOT / "contest_solution" / "order_model_v5b.py"


def _load_generated_model():
    spec = importlib.util.spec_from_file_location("order_model_v5b_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MODEL


def test_embedded_v5b_model_exactly_matches_content_bound_json_artifact():
    artifact = json.loads(MODEL_PATH.read_bytes())
    embedded = _load_generated_model()
    assert embedded == artifact
    payload = dict(embedded)
    expected = payload.pop("payload_sha256")
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    assert hashlib.sha256(canonical.encode()).hexdigest() == expected
    assert embedded["training"]["layout_selection"] == "clean_plus_hash_raw"
    assert embedded["provenance"]["source_files_excluded"] == 741
