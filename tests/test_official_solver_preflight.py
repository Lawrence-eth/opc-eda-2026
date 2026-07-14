import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from contest_solution import golden_plus_repair, learned_order, order_model_v5b
from scripts.preflight_official_solver import (
    EXPECTED_V5B_PAYLOAD_SHA256,
    PreflightError,
    check_replacement_policy,
    check_v5b_integrity,
    exercise_safe_mib_repair,
)
from scripts.solver_components import LIVE_SOLVER_COMPONENTS


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "scripts" / "preflight_official_solver.py"
SOLVER_DIR = ROOT / "contest_solution"


def test_setup_runs_preflight_after_registry_copy_before_official_evaluator():
    setup = (ROOT / "scripts" / "setup_and_evaluate.sh").read_text()

    registry_copy = setup.index('cp "$ROOT/contest_solution/$component"')
    preflight = setup.index('scripts/preflight_official_solver.py')
    official_validation = setup.index('iccad2026_evaluate.py --validate')

    assert registry_copy < preflight < official_validation


def test_cli_imports_registry_copy_and_passes_every_live_contract(tmp_path):
    pytest.importorskip("torch")
    solver_dir = tmp_path / "iccad2026contest"
    solver_dir.mkdir()
    for component in LIVE_SOLVER_COMPONENTS:
        shutil.copy2(SOLVER_DIR / component, solver_dir / component)
    (solver_dir / "iccad2026_evaluate.py").write_text(
        """\
class FloorplanOptimizer:
    def __init__(self, verbose=False):
        self.verbose = verbose

def calculate_bbox_area(positions):
    return 0.0

def calculate_hpwl_b2b(positions, connectivity):
    return 0.0

def calculate_hpwl_p2b(positions, connectivity, pins):
    return 0.0
"""
    )
    completed = subprocess.run(
        [sys.executable, str(PREFLIGHT), "--solver-dir", str(solver_dir)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(completed.stdout)
    assert result["status"] == "PASS"
    assert result["v5b"]["payload_sha256"] == EXPECTED_V5B_PAYLOAD_SHA256
    assert result["replacement_policy"]["mode"] == "replacement"
    assert result["replacement_policy"]["abstain_sizes"] == [
        101, 105, 109, 112, 113, 118, 119, 120
    ]
    assert result["safe_mib_repair"] == {
        "accepted_groups": 1,
        "changed": True,
        "common_shape": [4.0, 10.0],
    }
    expected_imports = {Path(name).stem for name in LIVE_SOLVER_COMPONENTS}
    expected_imports.add("iccad2026_evaluate")
    assert set(result["imported_modules"]) == expected_imports
    assert all(
        Path(origin).parent == solver_dir.resolve()
        for origin in result["imported_modules"].values()
    )


def test_v5b_check_rejects_a_matching_digest_field_on_tampered_payload():
    tampered = copy.deepcopy(order_model_v5b.MODEL)
    assert tampered["payload_sha256"] == EXPECTED_V5B_PAYLOAD_SHA256
    tampered["coefficients"][0][0] += 0.25
    optimizer = SimpleNamespace(_compiled_learned_model=lambda: None)

    with pytest.raises(PreflightError, match="canonical integrity/compilation failed"):
        check_v5b_integrity(
            learned_order,
            SimpleNamespace(MODEL=tampered),
            optimizer,
        )


def test_v5b_check_rejects_wrong_frozen_payload_sha_before_compilation():
    optimizer = SimpleNamespace(_compiled_learned_model=lambda: {})

    with pytest.raises(PreflightError, match="v5b payload SHA mismatch"):
        check_v5b_integrity(
            learned_order,
            SimpleNamespace(MODEL={"payload_sha256": "0" * 64}),
            optimizer,
        )


class _ProductionOptimizer:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self._learned_order_enabled = True
        self._learned_order_mode = "replacement"


def test_policy_check_accepts_configured_complete_mapping_and_abstention():
    module = SimpleNamespace(
        _LEARNED_REPLACEMENT_WF={100: 0.8, 102: 1.2},
        MyOptimizer=_ProductionOptimizer,
    )

    result = check_replacement_policy(
        module,
        minimum_blocks=100,
        maximum_blocks=102,
        expected_abstain_sizes=(101,),
    )

    assert result["mapped_sizes"] == 2
    assert result["abstain_sizes"] == [101]


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        ({}, "missing or empty"),
        ({100: 0.8}, "coverage mismatch"),
        ({100: 0.8, 101: 0.9, 102: 1.2}, "coverage mismatch"),
        ({100: 0.75, 102: 1.2}, "invalid replacement width"),
    ],
)
def test_policy_check_fails_closed_on_bad_coverage_or_width(policy, message):
    module = SimpleNamespace(
        _LEARNED_REPLACEMENT_WF=policy,
        MyOptimizer=_ProductionOptimizer,
    )

    with pytest.raises(PreflightError, match=message):
        check_replacement_policy(
            module,
            minimum_blocks=100,
            maximum_blocks=102,
            expected_abstain_sizes=(101,),
        )


def test_safe_mib_probe_changes_to_expected_common_shape_with_real_torch():
    torch = pytest.importorskip("torch")

    result = exercise_safe_mib_repair(golden_plus_repair, torch)

    assert result == {
        "changed": True,
        "common_shape": [4.0, 10.0],
        "accepted_groups": 1,
    }
