import copy
import math

from scripts import audit_results


def _valid_result():
    cases = []
    for test_id, block_count in [(0, 21), (1, 22)]:
        cost = 2.0 + test_id
        runtime = 0.02 + test_id * 0.02
        cases.append(
            {
                "test_id": test_id,
                "block_count": block_count,
                "is_feasible": True,
                "cost": cost,
                "hpwl_gap": 0.5,
                "area_gap": 0.4,
                "violations_relative": 0.1,
                "runtime_seconds": runtime,
                "error": None,
                "positions": [[float(i), 0.0, 1.0, 1.0] for i in range(block_count)],
            }
        )
    max_blocks = max(case["block_count"] for case in cases)
    weights = [math.exp((case["block_count"] - max_blocks) / 12) for case in cases]
    total_weight = sum(weights)
    return {
        "total_score": sum(case["cost"] * weight / total_weight for case, weight in zip(cases, weights)),
        "summary": {
            "num_tests": 2,
            "num_feasible": 2,
            "avg_cost": sum(case["cost"] for case in cases) / len(cases),
            "avg_runtime": sum(case["runtime_seconds"] for case in cases) / len(cases),
        },
        "test_results": cases,
    }


def test_audit_accepts_complete_fully_feasible_result():
    ok, errors, warnings = audit_results.audit_result(_valid_result(), expected_cases=2, require_positions=True)

    assert ok
    assert errors == []
    assert warnings == []


def test_audit_rejects_duplicate_case_ids_and_summary_mismatch():
    data = _valid_result()
    data["test_results"][1]["test_id"] = 0
    data["summary"]["num_tests"] = 3

    ok, errors, _ = audit_results.audit_result(data, expected_cases=2)

    assert not ok
    assert any("duplicate test_id 0" in error for error in errors)
    assert any("summary.num_tests=3" in error for error in errors)


def test_audit_rejects_nonfinite_metric_and_bad_rectangle():
    data = _valid_result()
    data["test_results"][0]["cost"] = float("nan")
    data["test_results"][0]["positions"][0][2] = 0.0

    ok, errors, _ = audit_results.audit_result(data, require_positions=True)

    assert not ok
    assert any("cost must be finite" in error for error in errors)
    assert any("non-positive width/height" in error for error in errors)


def test_audit_rejects_overlapping_saved_rectangles():
    data = _valid_result()
    data["test_results"][0]["positions"][1] = [0.5, 0.0, 1.0, 1.0]

    ok, errors, _ = audit_results.audit_result(data, require_positions=True)

    assert not ok
    assert any("positions 0 and 1 overlap" in error for error in errors)


def test_audit_warns_when_positions_absent_but_not_required():
    data = _valid_result()
    for case in data["test_results"]:
        case.pop("positions")

    ok, errors, warnings = audit_results.audit_result(data)

    assert ok
    assert errors == []
    assert warnings == []


def test_audit_rejects_missing_positions_when_required():
    data = _valid_result()
    data["test_results"][0].pop("positions")

    ok, errors, warnings = audit_results.audit_result(data, require_positions=True)

    assert not ok
    assert any("positions are absent" in error for error in errors)
    assert warnings == []


def test_audit_enforces_full_feasibility_and_score_ceiling():
    data = _valid_result()
    data["total_score"] = 2.1
    data["summary"]["num_feasible"] = 1
    data["test_results"][1]["is_feasible"] = False

    ok, errors, _ = audit_results.audit_result(data, max_score=2.0)

    assert not ok
    assert any("exceeds maximum allowed score" in error for error in errors)
    assert any("is_feasible is not true" in error for error in errors)
    assert any("not fully feasible" in error for error in errors)


def test_audit_can_allow_infeasible_for_debug_artifacts():
    data = _valid_result()
    data["summary"]["num_feasible"] = 1
    data["test_results"][1]["is_feasible"] = False

    ok, errors, _ = audit_results.audit_result(copy.deepcopy(data), require_full_feasible=False)

    assert ok
    assert errors == []


def test_audit_rejects_total_score_that_does_not_match_cases():
    data = _valid_result()
    data["total_score"] += 0.01

    ok, errors, _ = audit_results.audit_result(data)

    assert not ok
    assert any("reconstructed weighted score" in error for error in errors)


def test_audit_rejects_summary_average_mismatches():
    data = _valid_result()
    data["summary"]["avg_cost"] += 0.01
    data["summary"]["avg_runtime"] += 0.01

    ok, errors, _ = audit_results.audit_result(data)

    assert not ok
    assert any("summary.avg_cost" in error for error in errors)
    assert any("summary.avg_runtime" in error for error in errors)
