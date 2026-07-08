#!/usr/bin/env python3
"""Run public-release checks for FloorSet result and documentation updates.

This combines the cheap guards that should pass before publishing repository
changes: result-artifact audit, public-safe wording scan, and optional optimizer
copy synchronization against an official contest checkout.

Examples:
    python scripts/check_public_release.py
    python scripts/check_public_release.py --contest-optimizer external/FloorSet/iccad2026contest/my_optimizer.py
    python scripts/check_public_release.py --candidate candidate_full.json
"""
from __future__ import annotations

import argparse
import filecmp
import re
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import audit_results, compare_results

DEFAULT_RESULT = ROOT / "results" / "integrated_v19.json"
DEFAULT_PUBLIC_OPTIMIZER = ROOT / "contest_solution" / "my_optimizer.py"
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


def check_optimizer_sync(public_optimizer: Path, contest_optimizer: Path | None) -> tuple[bool, list[str]]:
    if contest_optimizer is None:
        return True, []
    if not public_optimizer.exists():
        return False, [f"public optimizer is missing: {public_optimizer}"]
    if not contest_optimizer.exists():
        return False, [f"contest optimizer is missing: {contest_optimizer}"]
    if not filecmp.cmp(public_optimizer, contest_optimizer, shallow=False):
        return False, [f"optimizer copies differ: {public_optimizer} vs {contest_optimizer}"]
    return True, []


def run_checks(
    *,
    result_json: Path,
    expected_cases: int,
    max_score: float | None,
    require_positions: bool,
    public_optimizer: Path,
    contest_optimizer: Path | None,
    candidate_json: Path | None,
) -> tuple[bool, list[str]]:
    messages: list[str] = []
    ok = True

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
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT, help="Published full result JSON")
    parser.add_argument("--expected-cases", type=int, default=100, help="Expected number of evaluated cases")
    parser.add_argument("--max-score", type=float, default=1.67, help="Maximum allowed published score (current integrated = 1.6651)")
    parser.add_argument("--allow-missing-positions", action="store_true", help="Do not require saved rectangles")
    parser.add_argument("--public-optimizer", type=Path, default=DEFAULT_PUBLIC_OPTIMIZER)
    parser.add_argument("--contest-optimizer", type=Path, default=None, help="Optional active contest optimizer to compare")
    parser.add_argument("--candidate", type=Path, default=None, help="Optional candidate full-result JSON to compare")
    args = parser.parse_args()

    ok, messages = run_checks(
        result_json=args.result,
        expected_cases=args.expected_cases,
        max_score=args.max_score,
        require_positions=not args.allow_missing_positions,
        public_optimizer=args.public_optimizer,
        contest_optimizer=args.contest_optimizer,
        candidate_json=args.candidate,
    )
    print("Public release check: " + ("PASS" if ok else "FAIL"))
    for message in messages:
        print(message)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
