#!/usr/bin/env python3
"""Verify pinned official contest sources without accessing the network."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "docs" / "official_sources.json"
DEFAULT_RELEASE_MANIFEST = ROOT / "results" / "release_manifest.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _repo_path(root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty repository-relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"{field} must be repository-relative")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{field} escapes the repository") from exc
    return resolved


def _check_hash(path: Path, expected: Any, field: str, errors: list[str]) -> None:
    if not isinstance(expected, str) or SHA256_RE.fullmatch(expected) is None:
        errors.append(f"{field} is not a lowercase SHA-256")
    elif not path.is_file():
        errors.append(f"{field} file is missing: {path}")
    else:
        actual = _sha256(path)
        if actual != expected:
            errors.append(f"{field} hash mismatch for {path}: expected {expected}, got {actual}")


def _git(path: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(f"git {' '.join(args)} failed in {path}: {completed.stderr.strip()}")
    return completed.stdout.strip()


def verify_official_sources(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    root: Path = ROOT,
    floorset_path: Path | None = None,
    materials_dir: Path | None = None,
    require_floorset: bool = False,
    release_manifest_path: Path | None = DEFAULT_RELEASE_MANIFEST,
) -> tuple[bool, list[str], list[str]]:
    """Return ``(ok, errors, notes)`` for the offline source contract."""

    errors: list[str] = []
    notes: list[str] = []
    try:
        manifest = _load(manifest_path)
    except ValueError as exc:
        return False, [str(exc)], notes
    if manifest.get("schema_version") != 1:
        errors.append("official source schema_version must be 1")
    verified_at = manifest.get("verified_at_utc")
    try:
        if not isinstance(verified_at, str):
            raise ValueError
        dt.datetime.fromisoformat(verified_at.replace("Z", "+00:00"))
    except ValueError:
        errors.append("verified_at_utc must be an ISO-8601 timestamp")

    floorset = manifest.get("floorset")
    if not isinstance(floorset, dict):
        errors.append("floorset must be an object")
        floorset = {}
    expected_commit = floorset.get("commit")
    expected_tree = floorset.get("tree")
    if not isinstance(expected_commit, str) or GIT_RE.fullmatch(expected_commit) is None:
        errors.append("floorset.commit is not a 40-character Git object ID")
    if not isinstance(expected_tree, str) or GIT_RE.fullmatch(expected_tree) is None:
        errors.append("floorset.tree is not a 40-character Git object ID")
    files = floorset.get("files")
    if not isinstance(files, dict) or not files:
        errors.append("floorset.files must be a non-empty path/hash object")
        files = {}

    if floorset_path is None:
        candidate = root / "external" / "FloorSet"
        floorset_path = candidate if (candidate / ".git").exists() else None
    if floorset_path is None:
        message = "FloorSet checkout not present; commit/tree/file-byte checks skipped"
        if require_floorset:
            errors.append(message)
        else:
            notes.append(message)
    else:
        try:
            actual_commit = _git(floorset_path, "rev-parse", "HEAD")
            actual_tree = _git(floorset_path, "rev-parse", "HEAD^{tree}")
            tracked_status = _git(floorset_path, "status", "--porcelain", "--untracked-files=no")
        except ValueError as exc:
            errors.append(str(exc))
        else:
            if actual_commit != expected_commit:
                errors.append(f"FloorSet HEAD mismatch: expected {expected_commit}, got {actual_commit}")
            if actual_tree != expected_tree:
                errors.append(f"FloorSet tree mismatch: expected {expected_tree}, got {actual_tree}")
            if tracked_status:
                errors.append("FloorSet checkout has tracked modifications")
            for name, digest in files.items():
                try:
                    path = _repo_path(floorset_path, name, f"floorset.files[{name!r}]")
                except ValueError as exc:
                    errors.append(str(exc))
                    continue
                _check_hash(path, digest, f"floorset.files[{name!r}]", errors)

    documents = manifest.get("documents")
    if not isinstance(documents, list) or not documents:
        errors.append("documents must be a non-empty array")
        documents = []
    for index, document in enumerate(documents):
        field = f"documents[{index}]"
        if not isinstance(document, dict):
            errors.append(f"{field} must be an object")
            continue
        try:
            extract = _repo_path(root, document.get("tracked_extract"), f"{field}.tracked_extract")
        except ValueError as exc:
            errors.append(str(exc))
        else:
            _check_hash(extract, document.get("tracked_extract_sha256"), f"{field}.tracked_extract_sha256", errors)
        digest = document.get("sha256")
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            errors.append(f"{field}.sha256 is not a lowercase SHA-256")
        if materials_dir is not None:
            download_name = document.get("download_name")
            if not isinstance(download_name, str) or not download_name:
                errors.append(f"{field}.download_name is invalid")
            else:
                path = materials_dir / download_name
                _check_hash(path, digest, f"{field}.sha256", errors)
                if path.is_file() and path.stat().st_size != document.get("size_bytes"):
                    errors.append(f"{field}.size_bytes mismatch for {path}")

    wrapper = manifest.get("submission_wrapper")
    if not isinstance(wrapper, dict):
        errors.append("submission_wrapper must be an object")
        wrapper = {}
    try:
        wrapper_path = _repo_path(root, wrapper.get("tracked_path"), "submission_wrapper.tracked_path")
    except ValueError as exc:
        errors.append(str(exc))
    else:
        _check_hash(wrapper_path, wrapper.get("sha256"), "submission_wrapper.sha256", errors)
    if materials_dir is not None and isinstance(wrapper.get("download_name"), str):
        downloaded_wrapper = materials_dir / wrapper["download_name"]
        _check_hash(downloaded_wrapper, wrapper.get("sha256"), "downloaded submission wrapper", errors)
        if downloaded_wrapper.is_file() and downloaded_wrapper.stat().st_size != wrapper.get("size_bytes"):
            errors.append(f"submission_wrapper.size_bytes mismatch for {downloaded_wrapper}")

    if release_manifest_path is not None and release_manifest_path.is_file():
        try:
            release = _load(release_manifest_path)
            release_floorset = release["floorset"]
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f"cannot compare release manifest FloorSet pin: {exc}")
        else:
            if release_floorset.get("repository") != floorset.get("repository"):
                errors.append("release manifest and official sources disagree on FloorSet repository")
            if release_floorset.get("commit") != expected_commit:
                errors.append("release manifest and official sources disagree on FloorSet commit")

    if materials_dir is None:
        notes.append("original Drive downloads not supplied; recorded PDF hashes were not re-read")
    else:
        notes.append(f"verified original Drive downloads in {materials_dir}")
    return not errors, errors, notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--floorset", type=Path)
    parser.add_argument("--materials-dir", type=Path, default=os.environ.get("ICCAD_OFFICIAL_MATERIALS_DIR"))
    parser.add_argument("--require-floorset", action="store_true")
    parser.add_argument("--release-manifest", type=Path, default=DEFAULT_RELEASE_MANIFEST)
    args = parser.parse_args()

    ok, errors, notes = verify_official_sources(
        args.manifest,
        floorset_path=args.floorset,
        materials_dir=args.materials_dir,
        require_floorset=args.require_floorset,
        release_manifest_path=args.release_manifest,
    )
    print("Official source integrity: " + ("PASS" if ok else "FAIL"))
    for note in notes:
        print(f"  note: {note}")
    for error in errors:
        print(f"  error: {error}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
