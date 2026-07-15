#!/usr/bin/env python3
"""Run public-release checks for FloorSet result and documentation updates.

This combines the cheap guards that should pass before publishing repository
changes: release-manifest integrity, result-artifact audit, public-safe wording
scan, and optional optimizer copy synchronization against an official contest
checkout.

Examples:
    python scripts/check_public_release.py
    python scripts/check_public_release.py --contest-optimizer external/FloorSet/iccad2026contest/my_optimizer.py
    python scripts/check_public_release.py --candidate candidate_full.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import filecmp
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import audit_results, check_official_sources, compare_results
from scripts.solver_components import (
    LIVE_SOLVER_COMPONENTS,
    PACKAGE_SUPPORT_SOURCE_BINDINGS,
    SOLVER_ENTRYPOINT,
)

DEFAULT_MANIFEST = ROOT / "results" / "release_manifest.json"
DEFAULT_SCAN_PATHS = (
    ROOT / "README.md",
    ROOT / "PROJECT_STATUS.md",
    ROOT / "docs" / "SUMMARY.md",
)

BLOCKED_PHRASES = (
    "autonomous workspace",
    "provided PDFs",
    "after review",
    "external-agent",
    "Hermes",
)

SENSITIVE_WORDS = (
    "token",
    "pat",
    "credential",
    "credentials",
    "secret",
    "secrets",
)

SECRET_LIKE_PATTERNS = (
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b"),
)

TEXT_SUFFIXES = {
    ".md",
    ".py",
    ".sh",
    ".txt",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
}

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
RELEASE_GITHUB_REPOSITORY = "Lawrence-eth/opc-eda-2026"
GITHUB_ACTIONS_RUN_RE = re.compile(
    rf"^https://github\.com/{re.escape(RELEASE_GITHUB_REPOSITORY)}"
    r"/actions/runs/([1-9][0-9]*)$"
)
RELEASE_MANIFEST_SCHEMA_VERSION = 2
LEARNED_ORDER_MODES = {
    "off",
    "replacement",
    "additive",
    "additive_first_pass",
}
SEALED_SELECTOR_PATH = (
    "results/research/policy_tournament_v1/sealed_selector.json"
)
SEALED_SELECTOR_STATUSES = {
    "sealed_confirmation_passed",
    "sealed_confirmation_failed_fallback_off",
}


def _reject_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        value[key] = item
    return value


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle, object_pairs_hook=_reject_duplicate_object)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"cannot load {description} {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{description} {path} must contain a JSON object")
    return data


def load_release_manifest(path: Path) -> dict[str, Any]:
    """Load a release manifest with a useful error for malformed input."""

    return _load_json_object(path, "release manifest")


def _resolve_repo_path(root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty repository-relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"{field} must be repository-relative, got {value!r}")
    root = root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} escapes the repository root: {value!r}") from exc
    return resolved


def _manifest_section(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    section = manifest.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"release manifest field {name!r} must be an object")
    return section


def release_manifest_defaults(
    manifest: dict[str, Any], root: Path = ROOT
) -> tuple[Path, int, float, Path]:
    """Return CLI defaults supplied by a structurally valid release manifest."""

    result = _manifest_section(manifest, "public_result")
    solver = _manifest_section(manifest, "solver")
    result_path = _resolve_repo_path(root, result.get("path"), "public_result.path")
    optimizer_path = _resolve_repo_path(root, solver.get("entrypoint"), "solver.entrypoint")
    num_cases = result.get("num_cases")
    score = result.get("total_score")
    if isinstance(num_cases, bool) or not isinstance(num_cases, int) or num_cases <= 0:
        raise ValueError("public_result.num_cases must be a positive integer")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        raise ValueError("public_result.total_score must be a finite number")
    return result_path, num_cases, float(score), optimizer_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_file_hash(path: Path, expected: Any, field: str, errors: list[str]) -> None:
    if not isinstance(expected, str) or SHA256_RE.fullmatch(expected) is None:
        errors.append(f"{field} must be a lowercase SHA-256 digest")
        return
    if not path.is_file():
        errors.append(f"{field} file is missing: {path}")
        return
    actual = _sha256_file(path)
    if actual != expected:
        errors.append(f"{field} hash mismatch for {path}: expected {expected}, got {actual}")


def _require_exact_keys(
    value: Any,
    expected: set[str],
    field: str,
    errors: list[str],
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        errors.append(f"{field} must be an object")
        return None
    observed = set(value)
    if observed != expected:
        errors.append(
            f"{field} must contain exactly {sorted(expected)}; "
            f"observed {sorted(observed)}"
        )
        return None
    return value


def _validate_decision_evidence(
    manifest: dict[str, Any],
    root: Path,
    selected_mode: Any,
    errors: list[str],
    *,
    verify_native_commit: bool,
) -> None:
    decision = _require_exact_keys(
        manifest.get("decision_evidence"),
        {
            "native_tournament",
            "sealed_selector",
            "sealed_policy_overridden",
            "rationale",
        },
        "decision_evidence",
        errors,
    )
    if decision is None:
        return

    native = _require_exact_keys(
        decision.get("native_tournament"),
        {
            "run_id",
            "run_url",
            "head_sha",
            "build_manifest_sha256",
            "timing_manifest_sha256",
            "evidence_bundle_sha256",
        },
        "decision_evidence.native_tournament",
        errors,
    )
    if native is not None:
        run_id = native.get("run_id")
        if isinstance(run_id, bool) or not isinstance(run_id, int) or run_id < 1:
            errors.append(
                "decision_evidence.native_tournament.run_id must be a positive integer"
            )
        run_url = native.get("run_url")
        match = (
            GITHUB_ACTIONS_RUN_RE.fullmatch(run_url)
            if isinstance(run_url, str)
            else None
        )
        if match is None:
            errors.append(
                "decision_evidence.native_tournament.run_url must be a canonical "
                f"GitHub Actions run URL for {RELEASE_GITHUB_REPOSITORY}"
            )
        elif isinstance(run_id, int) and not isinstance(run_id, bool):
            if int(match.group(1)) != run_id:
                errors.append(
                    "decision_evidence native tournament run URL and run ID disagree"
                )

        head_sha = native.get("head_sha")
        head_valid = (
            isinstance(head_sha, str)
            and GIT_COMMIT_RE.fullmatch(head_sha) is not None
        )
        if not head_valid:
            errors.append(
                "decision_evidence.native_tournament.head_sha must be a lowercase "
                "40-character Git commit"
            )
        elif verify_native_commit:
            completed = subprocess.run(
                ["git", "-C", str(root), "cat-file", "-e", f"{head_sha}^{{commit}}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if completed.returncode != 0:
                errors.append(
                    "decision_evidence native tournament head commit is unavailable: "
                    f"{head_sha}"
                )

        for field in (
            "build_manifest_sha256",
            "timing_manifest_sha256",
            "evidence_bundle_sha256",
        ):
            if (
                not isinstance(native.get(field), str)
                or SHA256_RE.fullmatch(native[field]) is None
            ):
                errors.append(
                    f"decision_evidence.native_tournament.{field} must be a "
                    "lowercase SHA-256 digest"
                )

    sealed = _require_exact_keys(
        decision.get("sealed_selector"),
        {"path", "sha256", "status"},
        "decision_evidence.sealed_selector",
        errors,
    )
    sealed_final_mode = None
    if sealed is not None:
        declared_path = sealed.get("path")
        if declared_path != SEALED_SELECTOR_PATH:
            errors.append(
                "decision_evidence.sealed_selector.path must name the canonical "
                f"selector {SEALED_SELECTOR_PATH!r}"
            )
            selector_path = None
        else:
            try:
                selector_path = _resolve_repo_path(
                    root,
                    declared_path,
                    "decision_evidence.sealed_selector.path",
                )
            except ValueError as exc:
                errors.append(str(exc))
                selector_path = None

        declared_status = sealed.get("status")
        if declared_status not in SEALED_SELECTOR_STATUSES:
            errors.append(
                "decision_evidence.sealed_selector.status is unsupported"
            )
        declared_sha = sealed.get("sha256")
        if not isinstance(declared_sha, str) or SHA256_RE.fullmatch(declared_sha) is None:
            errors.append(
                "decision_evidence.sealed_selector.sha256 must be a lowercase "
                "SHA-256 digest"
            )

        if selector_path is not None:
            _check_file_hash(
                selector_path,
                declared_sha,
                "decision_evidence.sealed_selector.sha256",
                errors,
            )
            if selector_path.is_file():
                try:
                    selector = _load_json_object(selector_path, "sealed selector")
                except ValueError as exc:
                    errors.append(str(exc))
                else:
                    actual_status = selector.get("status")
                    if actual_status != declared_status:
                        errors.append(
                            "decision_evidence sealed-selector status does not match "
                            f"{selector_path}: expected {declared_status!r}, "
                            f"got {actual_status!r}"
                        )
                    sealed_final_mode = selector.get("final_mode")
                    if sealed_final_mode not in LEARNED_ORDER_MODES:
                        errors.append(
                            "canonical sealed selector has an unsupported final_mode"
                        )

    overridden = decision.get("sealed_policy_overridden")
    if not isinstance(overridden, bool):
        errors.append("decision_evidence.sealed_policy_overridden must be a boolean")
    elif selected_mode in LEARNED_ORDER_MODES and sealed_final_mode in LEARNED_ORDER_MODES:
        expected_override = selected_mode != sealed_final_mode
        if overridden is not expected_override:
            errors.append(
                "decision_evidence.sealed_policy_overridden disagrees with the "
                "selected mode and canonical sealed-selector final_mode"
            )

    rationale = decision.get("rationale")
    if (
        not isinstance(rationale, str)
        or len(rationale.strip()) < 20
        or "\x00" in rationale
    ):
        errors.append(
            "decision_evidence.rationale must be a substantive string of at least "
            "20 non-padding characters"
        )


def validate_release_manifest(
    manifest: dict[str, Any],
    root: Path = ROOT,
    *,
    verify_solver_commit: bool = True,
) -> tuple[bool, list[str]]:
    """Validate release metadata and every incumbent artifact binding."""

    errors: list[str] = []
    if manifest.get("schema_version") != RELEASE_MANIFEST_SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {RELEASE_MANIFEST_SCHEMA_VERSION}"
        )
    if not isinstance(manifest.get("release"), str) or not manifest.get("release"):
        errors.append("release must be a non-empty string")
    verified_on = manifest.get("verified_on")
    try:
        if not isinstance(verified_on, str):
            raise ValueError
        dt.date.fromisoformat(verified_on)
    except ValueError:
        errors.append("verified_on must be an ISO 8601 calendar date")

    try:
        solver = _manifest_section(manifest, "solver")
    except ValueError as exc:
        errors.append(str(exc))
        solver = {}
    commit = solver.get("commit")
    if not isinstance(commit, str) or GIT_COMMIT_RE.fullmatch(commit) is None:
        errors.append("solver.commit must be a lowercase 40-character Git commit")
    if not isinstance(solver.get("version"), str) or not solver.get("version"):
        errors.append("solver.version must be a non-empty string")
    learned_order_mode = solver.get("learned_order_mode")
    if learned_order_mode not in LEARNED_ORDER_MODES:
        errors.append(
            "solver.learned_order_mode must be one of "
            + ", ".join(sorted(LEARNED_ORDER_MODES))
        )
    sources = solver.get("sources")
    if not isinstance(sources, dict) or not sources:
        errors.append("solver.sources must be a non-empty path-to-SHA-256 object")
        sources = {}
    required_solver_sources = {
        f"contest_solution/{component}" for component in LIVE_SOLVER_COMPONENTS
    }
    missing_solver_sources = sorted(required_solver_sources - sources.keys())
    if missing_solver_sources:
        errors.append(
            "solver.sources missing live solver components: "
            + ", ".join(missing_solver_sources)
        )
    missing_package_sources = sorted(
        set(PACKAGE_SUPPORT_SOURCE_BINDINGS) - sources.keys()
    )
    if missing_package_sources:
        errors.append(
            "solver.sources missing package support sources: "
            + ", ".join(missing_package_sources)
        )
    entrypoint = solver.get("entrypoint")
    if not isinstance(entrypoint, str) or entrypoint not in sources:
        errors.append("solver.entrypoint must name one of solver.sources")

    commit_valid = isinstance(commit, str) and GIT_COMMIT_RE.fullmatch(commit) is not None
    for source_name, expected_hash in sources.items():
        try:
            source_path = _resolve_repo_path(root, source_name, f"solver.sources[{source_name!r}]")
        except ValueError as exc:
            errors.append(str(exc))
            continue
        _check_file_hash(source_path, expected_hash, f"solver.sources[{source_name!r}]", errors)
        if verify_solver_commit and commit_valid and isinstance(expected_hash, str) and SHA256_RE.fullmatch(expected_hash):
            completed = subprocess.run(
                ["git", "-C", str(root), "show", f"{commit}:{source_name}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if completed.returncode != 0:
                errors.append(f"solver commit does not contain {source_name!r} at {commit}")
            else:
                committed_hash = hashlib.sha256(completed.stdout).hexdigest()
                if committed_hash != expected_hash:
                    errors.append(
                        f"solver commit hash mismatch for {source_name}: "
                        f"expected {expected_hash}, got {committed_hash}"
                    )

    try:
        public_result = _manifest_section(manifest, "public_result")
        result_path = _resolve_repo_path(root, public_result.get("path"), "public_result.path")
    except ValueError as exc:
        errors.append(str(exc))
        public_result = {}
        result_path = None
    if result_path is not None:
        _check_file_hash(result_path, public_result.get("sha256"), "public_result.sha256", errors)
        if result_path.is_file():
            try:
                result_data = audit_results.load_result(result_path)
            except (OSError, json.JSONDecodeError, SystemExit) as exc:
                errors.append(f"cannot inspect public result {result_path}: {exc}")
            else:
                actual_score = result_data.get("total_score")
                expected_score = public_result.get("total_score")
                if (
                    isinstance(actual_score, bool)
                    or not isinstance(actual_score, (int, float))
                    or isinstance(expected_score, bool)
                    or not isinstance(expected_score, (int, float))
                    or not math.isclose(float(actual_score), float(expected_score), rel_tol=0.0, abs_tol=1e-12)
                ):
                    errors.append(
                        f"public_result.total_score does not match {result_path}: "
                        f"expected {expected_score!r}, got {actual_score!r}"
                    )
                cases = result_data.get("test_results")
                expected_cases = public_result.get("num_cases")
                actual_cases = len(cases) if isinstance(cases, list) else None
                if isinstance(expected_cases, bool) or not isinstance(expected_cases, int) or actual_cases != expected_cases:
                    errors.append(
                        f"public_result.num_cases does not match {result_path}: "
                        f"expected {expected_cases!r}, got {actual_cases!r}"
                    )
                expected_feasible = public_result.get("num_feasible")
                actual_feasible = (
                    sum(case.get("is_feasible") is True for case in cases if isinstance(case, dict))
                    if isinstance(cases, list)
                    else None
                )
                if (
                    isinstance(expected_feasible, bool)
                    or not isinstance(expected_feasible, int)
                    or actual_feasible != expected_feasible
                ):
                    errors.append(
                        f"public_result.num_feasible does not match {result_path}: "
                        f"expected {expected_feasible!r}, got {actual_feasible!r}"
                    )

    try:
        package = _manifest_section(manifest, "submission_package")
        package_path = _resolve_repo_path(root, package.get("path"), "submission_package.path")
    except ValueError as exc:
        errors.append(str(exc))
    else:
        _check_file_hash(package_path, package.get("sha256"), "submission_package.sha256", errors)

    try:
        floorset = _manifest_section(manifest, "floorset")
    except ValueError as exc:
        errors.append(str(exc))
        floorset = {}
    floorset_commit = floorset.get("commit")
    if not isinstance(floorset_commit, str) or GIT_COMMIT_RE.fullmatch(floorset_commit) is None:
        errors.append("floorset.commit must be a lowercase 40-character Git commit")
    if not isinstance(floorset.get("repository"), str) or not floorset.get("repository"):
        errors.append("floorset.repository must be a non-empty string")

    _validate_decision_evidence(
        manifest,
        root,
        learned_order_mode,
        errors,
        verify_native_commit=verify_solver_commit,
    )

    return not errors, errors


def _iter_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if not path.exists():
            continue
        if path.is_file():
            if path.suffix.lower() in TEXT_SUFFIXES:
                yield path
            continue
        for child in path.rglob("*"):
            if child.is_file() and child.suffix.lower() in TEXT_SUFFIXES:
                yield child


def scan_public_safe(paths: Iterable[Path] | None = None) -> tuple[bool, list[str]]:
    findings: list[str] = []
    paths = DEFAULT_SCAN_PATHS if paths is None else paths
    sensitive = re.compile(r"\b(" + "|".join(re.escape(word) for word in SENSITIVE_WORDS) + r")\b", re.IGNORECASE)
    for path in _iter_files(paths):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        lowered = text.lower()
        for phrase in BLOCKED_PHRASES:
            if phrase.lower() in lowered:
                findings.append(f"{rel}: blocked public phrase {phrase!r}")
        for match in sensitive.finditer(text):
            findings.append(f"{rel}: sensitive word {match.group(0)!r}")
        for pattern in SECRET_LIKE_PATTERNS:
            if pattern.search(text):
                findings.append(f"{rel}: secret-like token pattern")
    return not findings, findings


def check_optimizer_sync(
    public_optimizer: Path, contest_optimizer: Path | None
) -> tuple[bool, list[str]]:
    if contest_optimizer is None:
        return True, []
    pairs = [(public_optimizer, contest_optimizer)]
    pairs.extend(
        (
            public_optimizer.with_name(component),
            contest_optimizer.with_name(component),
        )
        for component in LIVE_SOLVER_COMPONENTS
        if component != SOLVER_ENTRYPOINT
    )

    messages: list[str] = []
    for public_path, contest_path in pairs:
        if not public_path.exists():
            messages.append(f"public optimizer dependency is missing: {public_path}")
            continue
        if not contest_path.exists():
            messages.append(f"contest optimizer dependency is missing: {contest_path}")
            continue
        if not filecmp.cmp(public_path, contest_path, shallow=False):
            messages.append(f"optimizer copies differ: {public_path} vs {contest_path}")
    return not messages, messages


def run_checks(
    *,
    result_json: Path,
    expected_cases: int,
    max_score: float | None,
    require_positions: bool,
    public_optimizer: Path,
    contest_optimizer: Path | None,
    candidate_json: Path | None,
    release_manifest: dict[str, Any] | None = None,
    manifest_root: Path = ROOT,
    verify_manifest_commit: bool = True,
    official_sources_path: Path | None = None,
    floorset_path: Path | None = None,
    official_materials_dir: Path | None = None,
    require_floorset: bool = False,
    release_manifest_path: Path | None = None,
) -> tuple[bool, list[str]]:
    messages: list[str] = []
    ok = True

    if release_manifest is not None:
        manifest_ok, manifest_errors = validate_release_manifest(
            release_manifest,
            manifest_root,
            verify_solver_commit=verify_manifest_commit,
        )
        messages.append(f"release_manifest={'PASS' if manifest_ok else 'FAIL'}")
        messages.extend(f"  error: {error}" for error in manifest_errors)
        ok = ok and manifest_ok

    if official_sources_path is not None:
        official_ok, official_errors, official_notes = check_official_sources.verify_official_sources(
            official_sources_path,
            root=manifest_root,
            floorset_path=floorset_path,
            materials_dir=official_materials_dir,
            require_floorset=require_floorset,
            release_manifest_path=release_manifest_path,
        )
        messages.append(f"official_sources={'PASS' if official_ok else 'FAIL'}")
        messages.extend(f"  note: {note}" for note in official_notes)
        messages.extend(f"  error: {error}" for error in official_errors)
        ok = ok and official_ok

    data = audit_results.load_result(result_json)
    audit_ok, audit_errors, audit_warnings = audit_results.audit_result(
        data,
        expected_cases=expected_cases,
        require_full_feasible=True,
        max_score=max_score,
        require_positions=require_positions,
    )
    messages.append(f"result_audit={'PASS' if audit_ok else 'FAIL'}")
    messages.extend(f"  warning: {warning}" for warning in audit_warnings)
    messages.extend(f"  error: {error}" for error in audit_errors)
    ok = ok and audit_ok

    scan_ok, scan_findings = scan_public_safe()
    messages.append(f"public_safe_scan={'PASS' if scan_ok else 'FAIL'}")
    messages.extend(f"  finding: {finding}" for finding in scan_findings)
    ok = ok and scan_ok

    sync_ok, sync_messages = check_optimizer_sync(public_optimizer, contest_optimizer)
    if contest_optimizer is not None:
        messages.append(f"optimizer_sync={'PASS' if sync_ok else 'FAIL'}")
        messages.extend(f"  error: {message}" for message in sync_messages)
    ok = ok and sync_ok

    if candidate_json is not None:
        baseline = compare_results.load_result(result_json)
        candidate = compare_results.load_result(candidate_json)
        candidate_audit_ok, candidate_audit_errors, candidate_audit_warnings = audit_results.audit_result(
            candidate,
            expected_cases=expected_cases,
            require_full_feasible=True,
            max_score=max_score,
            require_positions=require_positions,
        )
        messages.append(f"candidate_result_audit={'PASS' if candidate_audit_ok else 'FAIL'}")
        messages.extend(f"  warning: {warning}" for warning in candidate_audit_warnings)
        messages.extend(f"  error: {error}" for error in candidate_audit_errors)
        ok = ok and candidate_audit_ok
        compare_ok, compare_messages = compare_results.compare(baseline, candidate)
        messages.append(f"candidate_compare={'PASS' if compare_ok else 'FAIL'}")
        messages.extend(f"  {message}" for message in compare_messages)
        ok = ok and compare_ok

    return ok, messages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Incumbent release manifest supplying validated defaults",
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=None,
        help="Published full result JSON (default: public_result.path in manifest)",
    )
    parser.add_argument(
        "--expected-cases",
        type=int,
        default=None,
        help="Expected number of evaluated cases (default: public_result.num_cases in manifest)",
    )
    parser.add_argument(
        "--max-score",
        type=float,
        default=None,
        help="Maximum allowed published score (default: public_result.total_score in manifest)",
    )
    parser.add_argument("--allow-missing-positions", action="store_true", help="Do not require saved rectangles")
    parser.add_argument(
        "--public-optimizer",
        type=Path,
        default=None,
        help="Public optimizer source (default: solver.entrypoint in manifest)",
    )
    parser.add_argument("--contest-optimizer", type=Path, default=None, help="Optional active contest optimizer to compare")
    parser.add_argument("--candidate", type=Path, default=None, help="Optional candidate full-result JSON to compare")
    parser.add_argument(
        "--official-sources",
        type=Path,
        default=check_official_sources.DEFAULT_MANIFEST,
        help="Offline official-source manifest to verify",
    )
    parser.add_argument("--floorset", type=Path, default=None, help="Optional pinned FloorSet checkout")
    parser.add_argument(
        "--official-materials-dir",
        type=Path,
        default=None,
        help="Optional directory containing the original downloaded PDFs and wrapper",
    )
    parser.add_argument(
        "--require-floorset",
        action="store_true",
        help="Fail if no local FloorSet checkout is available for byte verification",
    )
    args = parser.parse_args()

    try:
        manifest = load_release_manifest(args.manifest)
        manifest_result, manifest_cases, manifest_score, manifest_optimizer = release_manifest_defaults(manifest)
    except ValueError as exc:
        print("Public release check: FAIL")
        print(f"release_manifest=FAIL\n  error: {exc}")
        sys.exit(1)

    result_json = args.result if args.result is not None else manifest_result
    expected_cases = args.expected_cases if args.expected_cases is not None else manifest_cases
    max_score = args.max_score if args.max_score is not None else manifest_score
    public_optimizer = args.public_optimizer if args.public_optimizer is not None else manifest_optimizer

    ok, messages = run_checks(
        result_json=result_json,
        expected_cases=expected_cases,
        max_score=max_score,
        require_positions=not args.allow_missing_positions,
        public_optimizer=public_optimizer,
        contest_optimizer=args.contest_optimizer,
        candidate_json=args.candidate,
        release_manifest=manifest,
        official_sources_path=args.official_sources,
        floorset_path=args.floorset,
        official_materials_dir=args.official_materials_dir,
        require_floorset=args.require_floorset,
        release_manifest_path=args.manifest,
    )
    print("Public release check: " + ("PASS" if ok else "FAIL"))
    for message in messages:
        print(message)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
