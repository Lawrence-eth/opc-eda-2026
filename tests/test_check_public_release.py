import json
import hashlib
from pathlib import Path

from scripts import check_public_release
from scripts.solver_components import LIVE_SOLVER_COMPONENTS, SOLVER_ENTRYPOINT


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _live_solver_tree(root: Path, text: str = "x = 1\n") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for component in LIVE_SOLVER_COMPONENTS:
        (root / component).write_text(text, encoding="utf-8")
    return root / SOLVER_ENTRYPOINT


def _release_fixture(tmp_path: Path) -> tuple[dict, Path, Path, Path]:
    source = tmp_path / "contest_solution" / "my_optimizer.py"
    source.parent.mkdir()
    source.write_text("def solve():\n    return []\n", encoding="utf-8")

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

    manifest = {
        "schema_version": 1,
        "release": "incumbent_test",
        "verified_on": "2026-07-10",
        "solver": {
            "version": "test",
            "commit": "1" * 40,
            "entrypoint": "contest_solution/my_optimizer.py",
            "sources": {"contest_solution/my_optimizer.py": _sha256(source)},
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
