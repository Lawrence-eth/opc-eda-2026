import importlib.util
import json
import math
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "select_policy_tournament", ROOT / "scripts" / "select_policy_tournament.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _head_commit():
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _metric_row(manifest_case, cost, *, runtime=0.1):
    hpwl_gap = 2.0 * (cost - 1.0)
    row = {
        "sample_index": manifest_case["sample_index"],
        "block_count": manifest_case["block_count"],
        "is_feasible": True,
        "cost": cost,
        "hpwl_gap": hpwl_gap,
        "hpwl_gap_clamped": max(0.0, hpwl_gap),
        "area_gap": 0.0,
        "area_gap_clamped": 0.0,
        "violations_relative": 0.0,
        "boundary_violations": 0,
        "grouping_violations": 0,
        "mib_violations": 0,
        "golden_mib_violations": 0,
        "runtime_seconds": runtime,
        "case_id": manifest_case["case_id"],
        "source_file": manifest_case["source_file"],
        "file_offset": manifest_case["file_offset"],
        "input_sha256": manifest_case["input_sha256"],
        "optimizer_target_sha256": manifest_case["optimizer_target_sha256"],
        "scoring_label_sha256": manifest_case["scoring_label_sha256"],
    }
    assert math.isclose(MODULE._official_cost(row), cost, rel_tol=0.0, abs_tol=1e-12)
    return row


def _holdout_payload(campaign, panel, mode, fold, cost):
    manifest = campaign["manifests"][panel]
    fold_row = MODULE._manifest_fold(manifest["payload"], fold, f"{panel} manifest")
    rows = [_metric_row(case, cost) for case in fold_row["cases"]]
    golden_rows = [_metric_row(case, 2.0, runtime=0.0) for case in fold_row["cases"]]
    by_size = {
        str(size): MODULE._summary([row for row in rows if row["block_count"] == size])
        for size in MODULE.EXPECTED_BLOCK_COUNTS
    }
    manifest_provenance = MODULE._expected_manifest_provenance(
        manifest["raw"], manifest["payload"], fold_row, fold
    )
    generation = manifest_provenance["generation"]
    return {
        "config": {
            "min_blocks": generation["min_blocks"],
            "max_blocks": generation["max_blocks"],
            "per_size": generation["per_size"],
            "seed": generation["seed"],
            "scanned_files": 0,
            "require_golden_mib_clean": None,
            "solver_dir": "contest_solution",
            "learned_mode": mode,
            "indices_from": None,
            "fold_manifest": MODULE.PANEL_MANIFEST_PATHS[panel],
            "fold": fold,
            "manifest": manifest_provenance,
            "oracle_baseline_selector": False,
            "runtime_factor_mode": "neutral_rf_1",
        },
        "provenance": {
            "evaluation_harness_sha256": campaign["evaluation_harness_sha256"],
            "solver_source_sha256": campaign["solver_source_sha256"],
            "solver_component_sha256": campaign["solver_component_sha256"],
            "solver_git": {
                "commit": campaign["expected_commit"],
                "dirty": False,
                "tracked_dirty": False,
                "has_untracked": False,
            },
            "evaluator_sha256": campaign["evaluator_sha256"],
            "official_floorset_git": {
                "commit": campaign["official_floorset_commit"],
                "dirty": True,
                "tracked_dirty": False,
                "has_untracked": True,
            },
        },
        "solver_all": MODULE._summary(rows),
        "golden_all": MODULE._summary(golden_rows),
        "solver_golden_mib_clean": MODULE._summary(rows),
        "golden_mib_clean": MODULE._summary(golden_rows),
        "golden_mib_violation_cases": 0,
        "solver_by_size": by_size,
        "cases": rows,
    }


class TournamentFixture:
    def __init__(self, root, campaign, costs, *, include_calibration=False):
        self.root = root
        self.campaign = campaign
        self.holdouts = {}
        self.dev = {mode: {} for mode in MODULE.CHALLENGERS}
        self.calibration = None
        folds = MODULE.DEV_FOLDS + ((MODULE.CALIBRATION_FOLD,) if include_calibration else ())
        for panel in MODULE.PANELS:
            for mode in MODULE.MODES:
                for fold in folds:
                    cost = costs.get((panel, mode, fold), costs.get((panel, mode), 2.0))
                    path = root / "holdouts" / f"{panel}-{mode}-f{fold}.json"
                    _write(path, _holdout_payload(campaign, panel, mode, fold, cost))
                    self.holdouts[(panel, mode, fold)] = path
        for mode in MODULE.CHALLENGERS:
            for panel in MODULE.PANELS:
                path = root / "compare" / f"{panel}-{mode}-dev.json"
                self._write_comparison(path, panel, mode, MODULE.DEV_FOLDS)
                self.dev[mode][panel] = path
        if include_calibration:
            self.calibration = {}
            # Tests predeclare replacement as the only fold-3 finalist.
            for panel in MODULE.PANELS:
                path = root / "compare" / f"{panel}-replacement-fold3.json"
                self._write_comparison(path, panel, "replacement", (MODULE.CALIBRATION_FOLD,))
                self.calibration[panel] = path

    def _write_comparison(self, path, panel, mode, folds):
        manifest_path = ROOT / MODULE.PANEL_MANIFEST_PATHS[panel]
        baseline = [self.holdouts[(panel, "off", fold)] for fold in folds]
        candidate = [self.holdouts[(panel, mode, fold)] for fold in folds]
        result = MODULE.fold_compare.compare_result_pairs(
            baseline,
            candidate,
            source_by_sample=MODULE.fold_compare._source_map(manifest_path),
            source_manifest_path=manifest_path,
            bootstrap_samples=MODULE.BOOTSTRAP_SAMPLES,
            seed=MODULE.BOOTSTRAP_SEED,
        )
        _write(path, result)

    def rebuild(self, panel, mode, folds=MODULE.DEV_FOLDS):
        path = (
            self.dev[mode][panel]
            if folds == MODULE.DEV_FOLDS
            else self.calibration[panel]
        )
        self._write_comparison(path, panel, mode, folds)

    def select(self, *, calibration=False):
        return MODULE.select_policy_tournament(
            development_paths=self.dev,
            expected_commit=self.campaign["expected_commit"],
            artifact_root=self.root,
            calibration_paths=self.calibration if calibration else None,
        )


@pytest.fixture(autouse=True)
def _bounded_bootstrap(monkeypatch):
    # Production remains frozen at 30,000; synthetic tests need only exercise
    # deterministic recomputation and gate boundaries.
    monkeypatch.setattr(MODULE, "BOOTSTRAP_SAMPLES", 100)


@pytest.fixture
def campaign():
    return MODULE._campaign_contract(_head_commit())


def _default_costs():
    return {
        ("clean", "off"): 2.0,
        ("raw", "off"): 2.0,
        ("clean", "replacement"): 1.99,
        ("raw", "replacement"): 1.999,
        ("clean", "additive"): 2.0,
        ("raw", "additive"): 2.0,
        ("clean", "additive_first_pass"): 2.001,
        ("raw", "additive_first_pass"): 2.0,
    }


def test_complete_matrix_selects_one_finalist_and_binds_every_artifact(tmp_path, campaign):
    fixture = TournamentFixture(tmp_path, campaign, _default_costs())

    first = fixture.select()
    second = fixture.select()

    assert first == second
    assert first["status"] == "requires_calibration"
    assert first["dev_finalist"] == "replacement"
    assert first["final_mode"] is None
    assert first["development"]["challengers"]["replacement"]["passed"] is True
    assert first["development"]["challengers"]["additive"]["passed"] is False
    assert first["thresholds"] == MODULE.THRESHOLDS
    artifacts = first["development"]["holdout_artifacts"]
    assert sum(
        len(artifacts[panel][mode])
        for panel in MODULE.PANELS
        for mode in MODULE.MODES
    ) == 24


def test_no_passing_challenger_selects_off(tmp_path, campaign):
    costs = {(panel, mode): 2.0 for panel in MODULE.PANELS for mode in MODULE.MODES}
    fixture = TournamentFixture(tmp_path, campaign, costs)

    result = fixture.select()

    assert result["dev_finalist"] == "off"
    assert result["final_mode"] == "off"
    assert result["status"] == "off_selected"
    assert result["development"]["selection"]["passing_challengers"] == []


def test_statistical_tie_uses_predeclared_complexity_order(tmp_path, campaign):
    costs = _default_costs()
    costs[("clean", "additive")] = 1.9899
    costs[("raw", "additive")] = 1.999
    fixture = TournamentFixture(tmp_path, campaign, costs)

    result = fixture.select()

    assert result["development"]["selection"]["ranking"][0] == "additive"
    assert result["dev_finalist"] == "replacement"
    assert result["development"]["selection"]["statistically_tied"] == [
        "replacement",
        "additive",
    ]


def test_fold3_pass_promotes_the_predeclared_finalist(tmp_path, campaign):
    costs = _default_costs()
    costs[("clean", "replacement", 3)] = 1.99
    costs[("raw", "replacement", 3)] = 2.0001
    fixture = TournamentFixture(tmp_path, campaign, costs, include_calibration=True)

    result = fixture.select(calibration=True)

    assert result["dev_finalist"] == "replacement"
    assert result["calibration"]["passed"] is True
    assert result["final_mode"] == "replacement"
    assert result["status"] == "calibration_passed"


def test_fold3_failure_falls_back_to_off_without_trying_another_mode(tmp_path, campaign):
    costs = _default_costs()
    costs[("clean", "replacement", 3)] = 2.001
    costs[("raw", "replacement", 3)] = 2.0
    fixture = TournamentFixture(tmp_path, campaign, costs, include_calibration=True)

    result = fixture.select(calibration=True)

    assert result["dev_finalist"] == "replacement"
    assert result["calibration"]["passed"] is False
    assert "clean.pooled_delta" in result["calibration"]["reasons"]
    assert result["final_mode"] == "off"
    assert result["status"] == "calibration_failed_fallback_off"


def test_tampered_comparison_statistics_are_recomputed_and_rejected(tmp_path, campaign):
    fixture = TournamentFixture(tmp_path, campaign, _default_costs())
    path = fixture.dev["replacement"]["clean"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["bootstrap"]["probability_candidate_improves"] = 0.0
    _write(path, payload)

    with pytest.raises(ValueError, match="does not match recomputation"):
        fixture.select()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload["config"].__setitem__("learned_mode", "additive"), "learned_mode"),
        (
            lambda payload: payload["provenance"]["solver_git"].__setitem__(
                "dirty", True
            ),
            "Git state",
        ),
        (
            lambda payload: payload["cases"][0].__setitem__(
                "optimizer_target_sha256", "f" * 64
            ),
            "optimizer_target_sha256",
        ),
    ],
)
def test_raw_holdout_role_and_provenance_tampering_is_rejected(
    tmp_path, campaign, mutation, message
):
    fixture = TournamentFixture(tmp_path, campaign, _default_costs())
    holdout = fixture.holdouts[("clean", "replacement", 0)]
    payload = json.loads(holdout.read_text(encoding="utf-8"))
    mutation(payload)
    _write(holdout, payload)
    fixture.rebuild("clean", "replacement")

    with pytest.raises(ValueError, match=message):
        fixture.select()


def test_strict_json_rejects_duplicate_keys_and_nonfinite_values():
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        MODULE._decode_json(b'{"value": 1, "value": 2}', "fixture")
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        MODULE._decode_json(b'{"value": NaN}', "fixture")


def test_output_writer_is_atomic_and_refuses_replacement(tmp_path):
    output = tmp_path / "ledger.json"
    MODULE._write_json_exclusive(output, {"status": "first"})

    with pytest.raises(FileExistsError, match="already exists"):
        MODULE._write_json_exclusive(output, {"status": "second"})

    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "first"}
    assert not list(tmp_path.glob("*.tmp"))
