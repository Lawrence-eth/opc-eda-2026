import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.solver_components import LIVE_SOLVER_COMPONENTS


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODEL_SHA256 = (
    "c94b4af92a7088f04206a5fa20dfbf807f945d9bdd80d9ffcbdc0b8b45f18beb"
)


def _canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def test_source_fallback_live_module_selftest_exercises_learned_and_mib(tmp_path):
    for component in LIVE_SOLVER_COMPONENTS:
        shutil.copy2(ROOT / "contest_solution" / component, tmp_path / component)
    shutil.copy2(ROOT / "packaging" / "torch_stub.py", tmp_path / "torch.py")
    shutil.copy2(
        ROOT / "packaging" / "eval_stub.py",
        tmp_path / "iccad2026_evaluate.py",
    )
    shutil.copy2(ROOT / "packaging" / "solver_main.py", tmp_path / "solver_main.py")

    completed = subprocess.run(
        [sys.executable, str(tmp_path / "solver_main.py"), "--self-test-live-modules"],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
        env={**os.environ, "PYTHONHASHSEED": "0"},
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    learned = payload["learned"]
    safe_mib = payload["safe_mib"]
    assert payload["schema_version"] == 1
    assert learned["model_payload_sha256"] == EXPECTED_MODEL_SHA256
    assert learned["candidate_attempted"] is True
    assert 100 <= learned["production_eligible_block_count"] <= 120
    assert 100 <= learned["abstention_block_count"] <= 120
    assert learned["abstention_verified"] is True
    assert len(learned["final_positions"]) == learned["production_eligible_block_count"]
    assert all(
        len(learned[name]) == 64
        for name in (
            "compiled_model_sha256",
            "prior_sha256",
            "raw_candidate_sha256",
        )
    )
    assert safe_mib["repaired"] is True
    assert safe_mib["positions_sha256"] == _canonical_sha256(
        safe_mib["positions"]
    )
    assert all(row[2:] == [4.0, 10.0] for row in safe_mib["positions"][:3])


def test_source_fallback_default_mode_report_is_read_only_json(tmp_path):
    for component in LIVE_SOLVER_COMPONENTS:
        shutil.copy2(ROOT / "contest_solution" / component, tmp_path / component)
    shutil.copy2(ROOT / "packaging" / "torch_stub.py", tmp_path / "torch.py")
    shutil.copy2(
        ROOT / "packaging" / "eval_stub.py",
        tmp_path / "iccad2026_evaluate.py",
    )
    shutil.copy2(ROOT / "packaging" / "solver_main.py", tmp_path / "solver_main.py")

    completed = subprocess.run(
        [sys.executable, str(tmp_path / "solver_main.py"), "--report-default-mode"],
        input="this is deliberately not a solve payload",
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
        env={**os.environ, "PYTHONHASHSEED": "0"},
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "schema_version": 1,
        "learned_order_mode": "replacement",
    }


@pytest.mark.parametrize("default_mode", ["off", "additive", "additive_first_pass"])
def test_live_module_selftest_is_independent_of_promoted_mode(
    tmp_path, default_mode
):
    for component in LIVE_SOLVER_COMPONENTS:
        shutil.copy2(ROOT / "contest_solution" / component, tmp_path / component)
    optimizer_path = tmp_path / "my_optimizer.py"
    optimizer_source = optimizer_path.read_text(encoding="utf-8")
    frozen_default = 'self._learned_order_mode = "replacement"'
    assert optimizer_source.count(frozen_default) == 1
    optimizer_path.write_text(
        optimizer_source.replace(
            frozen_default,
            f'self._learned_order_mode = "{default_mode}"',
        ),
        encoding="utf-8",
    )
    shutil.copy2(ROOT / "packaging" / "torch_stub.py", tmp_path / "torch.py")
    shutil.copy2(
        ROOT / "packaging" / "eval_stub.py",
        tmp_path / "iccad2026_evaluate.py",
    )
    shutil.copy2(ROOT / "packaging" / "solver_main.py", tmp_path / "solver_main.py")

    completed = subprocess.run(
        [sys.executable, str(tmp_path / "solver_main.py"), "--self-test-live-modules"],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
        env={**os.environ, "PYTHONHASHSEED": "0"},
    )

    assert completed.returncode == 0, completed.stderr
    learned = json.loads(completed.stdout)["learned"]
    assert learned["candidate_attempted"] is True
    assert learned["abstention_verified"] is True
