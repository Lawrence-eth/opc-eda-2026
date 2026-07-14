import json
import subprocess
import sys
from pathlib import Path

from scripts.check_position_parity import check_position_parity


ROOT = Path(__file__).resolve().parents[1]


def _result(*, second_x=1.0, include_second=True):
    cases = [
        {
            "test_id": 0,
            "block_count": 1,
            "is_feasible": True,
            "cost": 1.0,
            "hpwl_gap": 0.0,
            "area_gap": 0.0,
            "violations_relative": 0.0,
            "runtime_seconds": 0.1,
            "positions": [[0.0, 0.0, 1.0, 1.0]],
        }
    ]
    if include_second:
        cases.append(
            {
                "test_id": 1,
                "block_count": 1,
                "is_feasible": True,
                "cost": 1.0,
                "hpwl_gap": 0.0,
                "area_gap": 0.0,
                "violations_relative": 0.0,
                "runtime_seconds": 0.2,
                "positions": [[second_x, 0.0, 1.0, 1.0]],
            }
        )
    return {
        "total_score": 1.0,
        "summary": {
            "num_tests": len(cases),
            "num_feasible": len(cases),
            "avg_cost": 1.0,
            "avg_runtime": sum(case["runtime_seconds"] for case in cases) / len(cases),
        },
        "test_results": cases,
    }


def test_exact_position_parity_passes():
    ok, messages = check_position_parity(
        _result(), _result(), expected_cases=2
    )

    assert ok
    assert "2 cases" in messages[0]
    assert "8 coordinates" in messages[0]
    assert "position_sha256=" in messages[0]


def test_exact_position_parity_reports_coordinate_drift():
    ok, messages = check_position_parity(
        _result(), _result(second_x=1.0000000000001), expected_cases=2
    )

    assert not ok
    assert any("case 1" in message and "x differs" in message for message in messages)


def test_exact_position_parity_rejects_missing_case():
    ok, messages = check_position_parity(
        _result(), _result(include_second=False), expected_cases=2
    )

    assert not ok
    assert any("missing case IDs" in message for message in messages)


def test_exact_position_parity_distinguishes_signed_zero():
    reference = _result()
    candidate = _result()
    candidate["test_results"][0]["positions"][0][0] = -0.0

    ok, messages = check_position_parity(
        reference, candidate, expected_cases=2
    )

    assert not ok
    assert any("x differs" in message for message in messages)


def test_exact_position_parity_rejects_quality_drift():
    candidate = _result()
    candidate["test_results"][1]["hpwl_gap"] = 0.25

    ok, messages = check_position_parity(
        _result(), candidate, expected_cases=2
    )

    assert not ok
    assert any("non-runtime field 'hpwl_gap' differs" in message for message in messages)


def test_position_parity_cli_is_a_fail_closed_gate(tmp_path):
    reference = tmp_path / "reference.json"
    candidate = tmp_path / "candidate.json"
    reference.write_text(json.dumps(_result()), encoding="utf-8")
    candidate.write_text(json.dumps(_result(second_x=2.0)), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_position_parity.py"),
            str(reference),
            str(candidate),
            "--expected-cases",
            "2",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "Position parity: FAIL" in completed.stdout
