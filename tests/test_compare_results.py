import copy
import math

from scripts import compare_results


def _reconstructed_score(data):
    cases = data["test_results"]
    max_blocks = max(case["block_count"] for case in cases)
    weights = [math.exp((case["block_count"] - max_blocks) / 12) for case in cases]
    total_weight = sum(weights)
    return sum(case["cost"] * weight / total_weight for case, weight in zip(cases, weights))


def _result(score=2.0, feasible=True, cases=2):
    test_results = [
        {
            "test_id": i,
            "block_count": 100 + i,
            "is_feasible": feasible,
            "runtime_seconds": 0.1 + i,
            "cost": score,
        }
        for i in range(cases)
    ]
    return {
        "total_score": score,
        "test_results": test_results,
        "summary": {
            "num_feasible": cases if feasible else cases - 1,
            "avg_runtime": 0.2,
        },
    }


def test_compare_accepts_strictly_lower_fully_feasible_score():
    ok, messages = compare_results.compare(_result(score=2.0), _result(score=1.9))

    assert ok
    assert any("PASS" in message for message in messages)


def test_compare_rejects_equal_score_by_default():
    ok, messages = compare_results.compare(_result(score=2.0), _result(score=2.0))

    assert not ok
    assert any("candidate score is equal to baseline" in message for message in messages)


def test_compare_can_allow_equal_for_reproducibility_checks():
    ok, _ = compare_results.compare(_result(score=2.0), _result(score=2.0), allow_equal=True)

    assert ok


def test_compare_rejects_infeasible_candidate_even_when_score_is_lower():
    candidate = _result(score=1.5, feasible=False)

    ok, messages = compare_results.compare(_result(score=2.0), candidate)

    assert not ok
    assert any("not fully feasible" in message for message in messages)


def test_compare_rejects_stale_feasibility_summary_even_when_score_is_lower():
    candidate = _result(score=1.5, feasible=True)
    candidate["test_results"][1]["is_feasible"] = False

    ok, messages = compare_results.compare(_result(score=2.0), candidate)

    assert not ok
    assert any("summary.num_feasible=2 does not match per-case feasible count 1" in message for message in messages)
    assert any("not fully feasible" in message for message in messages)


def test_compare_rejects_truncated_candidate_result():
    baseline = _result(score=2.0, cases=3)
    candidate = copy.deepcopy(_result(score=1.5, cases=2))

    ok, messages = compare_results.compare(baseline, candidate)

    assert not ok
    assert any("expected at least 3" in message for message in messages)


def test_compare_rejects_missing_baseline_test_ids_even_with_same_case_count():
    baseline = _result(score=2.0, cases=3)
    candidate = copy.deepcopy(_result(score=1.5, cases=3))
    candidate["test_results"][1]["test_id"] = 99

    ok, messages = compare_results.compare(baseline, candidate)

    assert not ok
    assert any("missing baseline test_id values: 1" in message for message in messages)


def test_compare_rejects_duplicate_candidate_test_ids():
    baseline = _result(score=2.0, cases=3)
    candidate = copy.deepcopy(_result(score=1.5, cases=3))
    candidate["test_results"][1]["test_id"] = 0

    ok, messages = compare_results.compare(baseline, candidate)

    assert not ok
    assert any("duplicate test_id values: 0" in message for message in messages)


def test_compare_rejects_stale_candidate_total_score_even_when_declared_lower():
    baseline = _result(score=2.0, cases=3)
    candidate = copy.deepcopy(baseline)
    candidate["total_score"] = 1.5
    candidate["test_results"][2]["cost"] = 2.5

    ok, messages = compare_results.compare(baseline, candidate)

    assert not ok
    assert any("candidate total_score" in message and "reconstructed score" in message for message in messages)


def test_compare_rejects_stale_baseline_total_score():
    baseline = _result(score=2.0, cases=3)
    baseline["total_score"] = 2.5
    candidate = _result(score=1.9, cases=3)

    ok, messages = compare_results.compare(baseline, candidate)

    assert not ok
    assert any("baseline total_score" in message and "reconstructed score" in message for message in messages)


def test_compare_reports_top_weighted_case_deltas():
    baseline = _result(score=2.0, cases=3)
    candidate = copy.deepcopy(baseline)
    candidate["test_results"][0]["cost"] = 1.5
    candidate["test_results"][1]["cost"] = 2.5
    candidate["test_results"][2]["cost"] = 1.2
    candidate["total_score"] = _reconstructed_score(candidate)
    candidate["test_results"][2]["hpwl_gap"] = 0.3
    baseline["test_results"][2]["hpwl_gap"] = 0.5

    ok, messages = compare_results.compare(baseline, candidate)
    joined = "\n".join(messages)

    assert ok
    assert "Top weighted regressions:" in joined
    assert "Top weighted improvements:" in joined
    assert "test_id=1" in joined
    assert "test_id=2" in joined
    assert "hpwl_delta=-0.2000" in joined


def test_weighted_case_deltas_uses_high_block_cases_first():
    baseline = _result(score=2.0, cases=3)
    candidate = copy.deepcopy(baseline)
    candidate["test_results"][0]["cost"] = 2.2
    candidate["test_results"][2]["cost"] = 2.5

    regressions, _ = compare_results.weighted_case_deltas(baseline, candidate, top=2)

    assert regressions[0].startswith("test_id=2")
