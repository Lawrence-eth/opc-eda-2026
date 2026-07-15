import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import check_public_release
from scripts.solver_components import (
    LIVE_SOLVER_COMPONENTS,
    PACKAGE_SUPPORT_SOURCE_BINDINGS,
    SOLVER_ENTRYPOINT,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _live_solver_tree(root: Path, text: str = "x = 1\n") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for component in LIVE_SOLVER_COMPONENTS:
        (root / component).write_text(text, encoding="utf-8")
    return root / SOLVER_ENTRYPOINT


def _release_fixture(tmp_path: Path) -> tuple[dict, Path, Path, Path]:
    source_dir = tmp_path / "contest_solution"
    source_dir.mkdir()
    sources = {}
    for component in LIVE_SOLVER_COMPONENTS:
        component_path = source_dir / component
        component_path.write_text(
            f"# fixture: {component}\n", encoding="utf-8"
        )
        sources[f"contest_solution/{component}"] = _sha256(component_path)
    for source_name in PACKAGE_SUPPORT_SOURCE_BINDINGS:
        support_path = tmp_path / source_name
        support_path.parent.mkdir(parents=True, exist_ok=True)
        support_path.write_text(f"# fixture: {source_name}\n", encoding="utf-8")
        sources[source_name] = _sha256(support_path)
    source = source_dir / SOLVER_ENTRYPOINT

    result = tmp_path / "results" / "incumbent.json"
    result.parent.mkdir()
    result.write_text(
        json.dumps(
            {
                "total_score": 2.0,
                "summary": {"num_tests": 1, "num_feasible": 1},
                "test_results": [{"is_feasible": True}],
            }
        ),
        encoding="utf-8",
    )
    package = tmp_path / "submission" / "submission.tar.gz"
    package.parent.mkdir()
    package.write_bytes(b"release package")

    sealed_selector = tmp_path / check_public_release.SEALED_SELECTOR_PATH
    sealed_selector.parent.mkdir(parents=True)
    sealed_selector.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "sealed_confirmation_failed_fallback_off",
                "final_mode": "off",
            }
        ),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": check_public_release.RELEASE_MANIFEST_SCHEMA_VERSION,
        "release": "incumbent_test",
        "verified_on": "2026-07-10",
        "solver": {
            "version": "test",
            "commit": "1" * 40,
            "entrypoint": "contest_solution/my_optimizer.py",
            "learned_order_mode": "replacement",
            "sources": sources,
        },
        "public_result": {
            "path": "results/incumbent.json",
            "sha256": _sha256(result),
            "total_score": 2.0,
            "num_cases": 1,
            "num_feasible": 1,
        },
        "submission_package": {
            "path": "submission/submission.tar.gz",
            "sha256": _sha256(package),
        },
        "floorset": {
            "repository": "https://github.com/IntelLabs/FloorSet.git",
            "commit": "2" * 40,
        },
        "decision_evidence": {
            "native_tournament": {
                "run_id": 123456789,
                "run_attempt": 1,
                "run_url": (
                    "https://github.com/Lawrence-eth/opc-eda-2026/"
                    "actions/runs/123456789"
                ),
                "head_sha": "1" * 40,
                "conclusion": "success",
                "artifact_id": 987654321,
                "artifact_name": f"native-package-tournament-{'1' * 40}-1",
                "artifact_size_bytes": 123456,
                "artifact_digest_sha256": "3" * 64,
                "preserved_asset_name": (
                    f"native-package-tournament-{'1' * 40}-1.zip"
                ),
                "preserved_asset_size_bytes": 120000,
                "preserved_asset_sha256": "7" * 64,
                "build_manifest_schema_version": 2,
                "timing_manifest_schema_version": 3,
                "build_manifest_sha256": "4" * 64,
                "timing_manifest_sha256": "5" * 64,
                "evidence_bundle_sha256": "6" * 64,
                "selected_mode": "replacement",
                "selected_package_sha256": _sha256(package),
                "selected_source_patch_changed": False,
            },
            "sealed_selector": {
                "path": check_public_release.SEALED_SELECTOR_PATH,
                "sha256": _sha256(sealed_selector),
                "status": "sealed_confirmation_failed_fallback_off",
            },
            "sealed_policy_overridden": True,
            "rationale": (
                "Native package evidence supports the explicit expected-score choice."
            ),
        },
    }
    return manifest, source, result, package


def test_public_safe_scan_uses_word_boundaries(tmp_path):
    safe = tmp_path / "README.md"
    safe.write_text("Use PYTHONPATH for imports.\n", encoding="utf-8")

    ok, findings = check_public_release.scan_public_safe([safe])

    assert ok
    assert findings == []


def test_public_safe_scan_rejects_blocked_phrase_and_sensitive_word(tmp_path):
    unsafe = tmp_path / "PROJECT_STATUS.md"
    unsafe.write_text("after review, do not include a token here\n", encoding="utf-8")

    ok, findings = check_public_release.scan_public_safe([unsafe])

    assert not ok
    assert any("after review" in finding for finding in findings)
    assert any("token" in finding for finding in findings)


def test_optimizer_sync_detects_mismatch(tmp_path):
    public = _live_solver_tree(tmp_path / "public")
    contest = _live_solver_tree(tmp_path / "contest")
    contest.write_text("x = 2\n", encoding="utf-8")

    ok, messages = check_public_release.check_optimizer_sync(public, contest)

    assert not ok
    assert any("optimizer copies differ" in message for message in messages)


def test_optimizer_sync_detects_missing_live_dependency(tmp_path):
    public = _live_solver_tree(tmp_path / "public")
    contest = _live_solver_tree(tmp_path / "contest")
    (contest.parent / "golden_plus_repair.py").unlink()

    ok, messages = check_public_release.check_optimizer_sync(public, contest)

    assert not ok
    assert any(
        "contest optimizer dependency is missing" in message
        and "golden_plus_repair.py" in message
        for message in messages
    )


def test_optimizer_sync_detects_live_artifact_drift(tmp_path):
    public = _live_solver_tree(tmp_path / "public")
    contest = _live_solver_tree(tmp_path / "contest")
    (contest.parent / "order_model_v5b.py").write_text(
        "MODEL = {'drifted': True}\n", encoding="utf-8"
    )

    ok, messages = check_public_release.check_optimizer_sync(public, contest)

    assert not ok
    assert any(
        "optimizer copies differ" in message and "order_model_v5b.py" in message
        for message in messages
    )


def test_release_manifest_validates_artifact_bindings_and_supplies_defaults(tmp_path):
    manifest, _, result, _ = _release_fixture(tmp_path)

    ok, errors = check_public_release.validate_release_manifest(
        manifest,
        tmp_path,
        verify_solver_commit=False,
    )
    defaults = check_public_release.release_manifest_defaults(manifest, tmp_path)

    assert ok
    assert errors == []
    assert defaults == (result, 1, 2.0, tmp_path / "contest_solution" / "my_optimizer.py")


def test_release_manifest_accepts_non_override_matching_sealed_policy(tmp_path):
    manifest, _, _, _ = _release_fixture(tmp_path)
    manifest["solver"]["learned_order_mode"] = "off"
    manifest["decision_evidence"]["native_tournament"]["selected_mode"] = "off"
    manifest["decision_evidence"]["sealed_policy_overridden"] = False

    ok, errors = check_public_release.validate_release_manifest(
        manifest,
        tmp_path,
        verify_solver_commit=False,
    )

    assert ok
    assert errors == []


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.__setitem__("schema_version", 1), "schema_version"),
        (
            lambda value: value["solver"].__setitem__("learned_order_mode", "unknown"),
            "solver.learned_order_mode",
        ),
        (
            lambda value: value["decision_evidence"]["native_tournament"].__setitem__(
                "run_id", True
            ),
            "run_id",
        ),
        (
            lambda value: value["decision_evidence"]["native_tournament"].__setitem__(
                "run_attempt", False
            ),
            "run_attempt",
        ),
        (
            lambda value: value["decision_evidence"]["native_tournament"].__setitem__(
                "conclusion", "cancelled"
            ),
            "conclusion",
        ),
        (
            lambda value: value["decision_evidence"]["native_tournament"].__setitem__(
                "timing_manifest_schema_version", 2
            ),
            "timing_manifest_schema_version",
        ),
        (
            lambda value: value["decision_evidence"]["native_tournament"].__setitem__(
                "artifact_name", "unbound-artifact"
            ),
            "artifact_name",
        ),
        (
            lambda value: value["decision_evidence"]["native_tournament"].__setitem__(
                "preserved_asset_name", "unbound.zip"
            ),
            "preserved_asset_name",
        ),
        (
            lambda value: value["decision_evidence"]["native_tournament"].__setitem__(
                "preserved_asset_sha256", "not-a-digest"
            ),
            "preserved_asset_sha256",
        ),
        (
            lambda value: value["decision_evidence"]["native_tournament"].__setitem__(
                "run_url",
                "https://github.com/Lawrence-eth/opc-eda-2026/actions/runs/987654321",
            ),
            "run URL and run ID disagree",
        ),
        (
            lambda value: value["decision_evidence"]["native_tournament"].__setitem__(
                "run_url",
                "https://github.com/unrelated/project/actions/runs/123456789",
            ),
            "canonical GitHub Actions run URL",
        ),
        (
            lambda value: value["decision_evidence"]["native_tournament"].pop(
                "build_manifest_sha256"
            ),
            "must contain exactly",
        ),
        (
            lambda value: value["decision_evidence"]["native_tournament"].__setitem__(
                "head_sha", "A" * 40
            ),
            "head_sha",
        ),
        (
            lambda value: value["decision_evidence"]["native_tournament"].__setitem__(
                "head_sha", "2" * 40
            ),
            "head_sha must equal solver.commit",
        ),
        (
            lambda value: value["decision_evidence"]["native_tournament"].__setitem__(
                "timing_manifest_sha256", "not-a-digest"
            ),
            "timing_manifest_sha256",
        ),
        (
            lambda value: value["decision_evidence"]["native_tournament"].__setitem__(
                "selected_mode", "off"
            ),
            "selected_mode",
        ),
        (
            lambda value: value["decision_evidence"]["native_tournament"].__setitem__(
                "selected_package_sha256", "7" * 64
            ),
            "selected package",
        ),
        (
            lambda value: value["decision_evidence"]["native_tournament"].__setitem__(
                "selected_source_patch_changed", True
            ),
            "selected_source_patch_changed",
        ),
        (
            lambda value: value["decision_evidence"]["sealed_selector"].__setitem__(
                "path", "results/research/other-selector.json"
            ),
            "canonical selector",
        ),
        (
            lambda value: value["decision_evidence"]["sealed_selector"].__setitem__(
                "status", "invented_status"
            ),
            "status is unsupported",
        ),
        (
            lambda value: value["decision_evidence"].__setitem__(
                "sealed_policy_overridden", False
            ),
            "sealed_policy_overridden disagrees",
        ),
        (
            lambda value: value["decision_evidence"].__setitem__("rationale", "short"),
            "rationale",
        ),
        (
            lambda value: value["decision_evidence"].__setitem__("extra", True),
            "must contain exactly",
        ),
    ],
)
def test_release_manifest_v2_rejects_malformed_decision_authority(
    tmp_path, mutate, message
):
    manifest, _, _, _ = _release_fixture(tmp_path)
    malformed = copy.deepcopy(manifest)
    mutate(malformed)

    ok, errors = check_public_release.validate_release_manifest(
        malformed,
        tmp_path,
        verify_solver_commit=False,
    )

    assert not ok
    assert any(message in error for error in errors), errors


def test_release_manifest_v2_rejects_sealed_selector_byte_or_status_drift(tmp_path):
    manifest, _, _, _ = _release_fixture(tmp_path)
    selector = tmp_path / check_public_release.SEALED_SELECTOR_PATH
    selector.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "sealed_confirmation_passed",
                "final_mode": "replacement",
            }
        ),
        encoding="utf-8",
    )

    ok, errors = check_public_release.validate_release_manifest(
        manifest,
        tmp_path,
        verify_solver_commit=False,
    )

    assert not ok
    assert any("sealed_selector.sha256 hash mismatch" in error for error in errors)
    assert any("sealed-selector status does not match" in error for error in errors)


def test_release_manifest_v2_rejects_unsupported_selector_final_mode(tmp_path):
    manifest, _, _, _ = _release_fixture(tmp_path)
    selector = tmp_path / check_public_release.SEALED_SELECTOR_PATH
    selector.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "sealed_confirmation_failed_fallback_off",
                "final_mode": "invented_mode",
            }
        ),
        encoding="utf-8",
    )
    manifest["decision_evidence"]["sealed_selector"]["sha256"] = _sha256(selector)

    ok, errors = check_public_release.validate_release_manifest(
        manifest,
        tmp_path,
        verify_solver_commit=False,
    )

    assert not ok
    assert any("unsupported final_mode" in error for error in errors)


def test_release_manifest_v2_rejects_unavailable_native_head_commit(tmp_path):
    manifest, _, _, _ = _release_fixture(tmp_path)

    ok, errors = check_public_release.validate_release_manifest(
        manifest,
        tmp_path,
        verify_solver_commit=True,
    )

    assert not ok
    assert any("native tournament head commit is unavailable" in error for error in errors)


def test_release_manifest_loader_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "release_manifest.json"
    path.write_text('{"schema_version":2,"schema_version":2}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        check_public_release.load_release_manifest(path)


def test_release_manifest_rejects_artifact_drift(tmp_path):
    manifest, source, _, package = _release_fixture(tmp_path)
    source.write_text("def solve():\n    return [1]\n", encoding="utf-8")
    package.write_bytes(b"stale package")

    ok, errors = check_public_release.validate_release_manifest(
        manifest,
        tmp_path,
        verify_solver_commit=False,
    )

    assert not ok
    assert any("solver.sources" in error and "hash mismatch" in error for error in errors)
    assert any("submission_package.sha256 hash mismatch" in error for error in errors)


def test_release_manifest_requires_every_live_solver_component(tmp_path):
    manifest, _, _, _ = _release_fixture(tmp_path)
    missing = "contest_solution/golden_plus_repair.py"
    del manifest["solver"]["sources"][missing]

    ok, errors = check_public_release.validate_release_manifest(
        manifest,
        tmp_path,
        verify_solver_commit=False,
    )

    assert not ok
    assert any(
        "missing live solver components" in error and missing in error
        for error in errors
    )


def test_release_manifest_requires_package_support_sources(tmp_path):
    manifest, _, _, _ = _release_fixture(tmp_path)
    missing = "packaging/solver_main.py"
    del manifest["solver"]["sources"][missing]

    ok, errors = check_public_release.validate_release_manifest(
        manifest,
        tmp_path,
        verify_solver_commit=False,
    )

    assert not ok
    assert any(
        "missing package support sources" in error and missing in error
        for error in errors
    )


def test_release_manifest_rejects_result_metadata_drift(tmp_path):
    manifest, _, _, _ = _release_fixture(tmp_path)
    manifest["public_result"]["total_score"] = 1.5
    manifest["public_result"]["num_feasible"] = 0

    ok, errors = check_public_release.validate_release_manifest(
        manifest,
        tmp_path,
        verify_solver_commit=False,
    )

    assert not ok
    assert any("public_result.total_score does not match" in error for error in errors)
    assert any("public_result.num_feasible does not match" in error for error in errors)


def test_run_checks_combines_audit_scan_and_sync(tmp_path, monkeypatch):
    result = tmp_path / "result.json"
    cases = [
        {
            "test_id": 0,
            "block_count": 21,
            "is_feasible": True,
            "cost": 2.0,
            "hpwl_gap": 0.5,
            "area_gap": 0.4,
            "violations_relative": 0.1,
            "runtime_seconds": 0.02,
            "error": None,
            "positions": [[float(i), 0.0, 1.0, 1.0] for i in range(21)],
        }
    ]
    result.write_text(
        json.dumps(
            {
                "total_score": 2.0,
                "summary": {"num_tests": 1, "num_feasible": 1, "avg_cost": 2.0, "avg_runtime": 0.02},
                "test_results": cases,
            }
        ),
        encoding="utf-8",
    )
    optimizer = _live_solver_tree(tmp_path)
    monkeypatch.setattr(check_public_release, "DEFAULT_SCAN_PATHS", (tmp_path,))

    ok, messages = check_public_release.run_checks(
        result_json=result,
        expected_cases=1,
        max_score=2.0,
        require_positions=True,
        public_optimizer=optimizer,
        contest_optimizer=optimizer,
        candidate_json=None,
    )

    assert ok
    assert "result_audit=PASS" in messages
    assert "public_safe_scan=PASS" in messages
    assert "optimizer_sync=PASS" in messages


def test_run_checks_audits_candidate_before_compare(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    cases = [
        {
            "test_id": 0,
            "block_count": 21,
            "is_feasible": True,
            "cost": 2.0,
            "hpwl_gap": 0.5,
            "area_gap": 0.4,
            "violations_relative": 0.1,
            "runtime_seconds": 0.02,
            "error": None,
            "positions": [[float(i), 0.0, 1.0, 1.0] for i in range(21)],
        }
    ]
    baseline.write_text(
        json.dumps(
            {
                "total_score": 2.0,
                "summary": {"num_tests": 1, "num_feasible": 1, "avg_cost": 2.0, "avg_runtime": 0.02},
                "test_results": cases,
            }
        ),
        encoding="utf-8",
    )
    stale_candidate = json.loads(baseline.read_text(encoding="utf-8"))
    stale_candidate["total_score"] = 1.5
    candidate.write_text(json.dumps(stale_candidate), encoding="utf-8")
    optimizer = _live_solver_tree(tmp_path)
    monkeypatch.setattr(check_public_release, "DEFAULT_SCAN_PATHS", (tmp_path,))

    ok, messages = check_public_release.run_checks(
        result_json=baseline,
        expected_cases=1,
        max_score=2.0,
        require_positions=True,
        public_optimizer=optimizer,
        contest_optimizer=optimizer,
        candidate_json=candidate,
    )

    joined = "\n".join(messages)
    assert not ok
    assert "candidate_result_audit=FAIL" in messages
    assert "candidate_compare=FAIL" in messages
    assert "reconstructed score" in joined


def test_tracked_official_source_manifest_passes_offline():
    ok, errors, notes = check_public_release.check_official_sources.verify_official_sources(
        check_public_release.check_official_sources.DEFAULT_MANIFEST,
        floorset_path=None,
        materials_dir=None,
        release_manifest_path=check_public_release.DEFAULT_MANIFEST,
    )

    assert ok, errors
    assert any("Drive downloads not supplied" in note for note in notes)
