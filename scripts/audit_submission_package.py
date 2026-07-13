#!/usr/bin/env python3
"""Fail-closed structural audit for an ICCAD submission archive.

The audit never trusts tar paths or metadata. It verifies the immutable wrapper,
the AMD64 executable, bundled source parity, glibc compatibility, and an
optional smoke run. It is intentionally independent of the contest evaluator
so it can run before a package is copied into FloorSet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import struct
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "submission" / "iccad2026_submission.tar.gz"
DEFAULT_OFFICIAL_SOURCES = ROOT / "docs" / "official_sources.json"
EXPECTED_ROOT = "iccad2026_submission"
EXECUTABLE = f"{EXPECTED_ROOT}/dist/my_optimizer/my_optimizer"
WRAPPER = f"{EXPECTED_ROOT}/op_wrapper.py"
MAX_COMPRESSED_BYTES = 512 * 1024 * 1024
MAX_EXPANDED_BYTES = 512 * 1024 * 1024
MAX_MEMBERS = 10_000
GLIBC_RE = re.compile(rb"GLIBC_(\d+)\.(\d+)")

SOURCE_BINDINGS = {
    "contest_solution/my_optimizer.py": "source_fallback/my_optimizer.py",
    "contest_solution/dissect.py": "source_fallback/dissect.py",
    "contest_solution/topology_polish.py": "source_fallback/topology_polish.py",
    "packaging/torch_stub.py": "source_fallback/torch.py",
    "packaging/eval_stub.py": "source_fallback/iccad2026_evaluate.py",
    "packaging/solver_main.py": "source_fallback/solver_main.py",
}


class PackageAuditError(ValueError):
    """Raised when an archive violates the submission contract."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackageAuditError(f"cannot load {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PackageAuditError(f"{description} {path} must contain a JSON object")
    return value


def official_wrapper_sha(path: Path) -> str:
    manifest = _load_json(path, "official-source manifest")
    wrapper = manifest.get("submission_wrapper")
    if not isinstance(wrapper, dict):
        raise PackageAuditError("official-source manifest lacks submission_wrapper")
    digest = wrapper.get("sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise PackageAuditError("submission_wrapper.sha256 is not a lowercase SHA-256")
    return digest


def release_package_sha(path: Path) -> str:
    manifest = _load_json(path, "release manifest")
    package = manifest.get("submission_package")
    if not isinstance(package, dict):
        raise PackageAuditError("release manifest lacks submission_package")
    digest = package.get("sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise PackageAuditError("submission_package.sha256 is not a lowercase SHA-256")
    return digest


def release_source_hashes(path: Path) -> dict[str, str]:
    manifest = _load_json(path, "release manifest")
    solver = manifest.get("solver")
    sources = solver.get("sources") if isinstance(solver, dict) else None
    if not isinstance(sources, dict) or not sources:
        raise PackageAuditError("release manifest lacks solver.sources")
    packaged: dict[str, str] = {}
    for source_name, digest in sources.items():
        if not isinstance(source_name, str) or not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise PackageAuditError("release manifest solver.sources is malformed")
        packaged[f"source_fallback/{Path(source_name).name}"] = digest
    return packaged


def _safe_member_name(name: str) -> PurePosixPath:
    if not name or name.startswith("/") or "\\" in name:
        raise PackageAuditError(f"unsafe archive member path: {name!r}")
    raw_parts = name.rstrip("/").split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise PackageAuditError(f"unsafe archive member path: {name!r}")
    path = PurePosixPath(*raw_parts)
    if not path.parts or path.parts[0] != EXPECTED_ROOT:
        raise PackageAuditError(f"archive member is outside {EXPECTED_ROOT}/: {name!r}")
    return path


def _member_bytes(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    handle = archive.extractfile(member)
    if handle is None:
        raise PackageAuditError(f"cannot read archive member {member.name!r}")
    return handle.read()


def _elf_machine(data: bytes, name: str) -> int:
    if len(data) < 20 or data[:4] != b"\x7fELF":
        raise PackageAuditError(f"{name} is not an ELF executable")
    if data[5] == 1:
        byte_order = "<"
    elif data[5] == 2:
        byte_order = ">"
    else:
        raise PackageAuditError(f"{name} has invalid ELF byte order {data[5]}")
    return struct.unpack(byte_order + "H", data[18:20])[0]


def _smoke(extracted_root: Path) -> None:
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        raise PackageAuditError("--smoke requires an AMD64 host or configured emulator")
    binary = extracted_root / EXPECTED_ROOT / "dist" / "my_optimizer" / "my_optimizer"
    request = {
        "block_count": 3,
        "area_targets": [100.0, 25.0, 4.0],
        "b2b_connectivity": [[0, 1, 1.0], [1, 2, 2.0]],
        "p2b_connectivity": [],
        "pins_pos": [],
        "constraints": [[0, 0, 0, 0, 0]] * 3,
        "target_positions": None,
    }
    try:
        completed = subprocess.run(
            [str(binary)],
            input=json.dumps(request) + "\n",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
            check=False,
            env={**os.environ, "PYTHONHASHSEED": "0"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PackageAuditError(f"binary smoke run failed to start or finish: {exc}") from exc
    if completed.returncode != 0:
        raise PackageAuditError(
            f"binary smoke run exited {completed.returncode}: {completed.stderr[-500:]}"
        )
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise PackageAuditError(f"binary smoke stdout is not one JSON value: {exc}") from exc
    if not isinstance(response, dict) or set(response) != {"positions"}:
        raise PackageAuditError("binary smoke output must be exactly an object containing positions")
    positions = response["positions"]
    if not isinstance(positions, list) or len(positions) != request["block_count"]:
        raise PackageAuditError("binary smoke output has the wrong block count")
    for index, row in enumerate(positions):
        if (
            not isinstance(row, list)
            or len(row) != 4
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in row)
            or any(not math.isfinite(float(value)) for value in row)
        ):
            raise PackageAuditError(f"binary smoke output row {index} is not four finite numbers")


def audit_archive(
    archive_path: Path,
    *,
    wrapper_sha256: str,
    expected_archive_sha256: str | None = None,
    source_root: Path | None = ROOT,
    expected_source_sha256: dict[str, str] | None = None,
    require_notices: bool = False,
    max_glibc: tuple[int, int] = (2, 41),
    smoke: bool = False,
) -> list[str]:
    """Audit one archive and return concise success details."""

    archive_path = archive_path.resolve()
    if not archive_path.is_file():
        raise PackageAuditError(f"archive does not exist: {archive_path}")
    if archive_path.stat().st_size > MAX_COMPRESSED_BYTES:
        raise PackageAuditError(f"archive exceeds {MAX_COMPRESSED_BYTES} compressed bytes")
    actual_archive_sha = sha256_file(archive_path)
    if expected_archive_sha256 is not None and actual_archive_sha != expected_archive_sha256:
        raise PackageAuditError(
            f"archive SHA-256 mismatch: expected {expected_archive_sha256}, got {actual_archive_sha}"
        )

    try:
        archive = tarfile.open(archive_path, mode="r:gz")
    except (OSError, tarfile.TarError) as exc:
        raise PackageAuditError(f"cannot open gzip tar archive {archive_path}: {exc}") from exc

    with archive:
        members = archive.getmembers()
        if not members or len(members) > MAX_MEMBERS:
            raise PackageAuditError(f"archive member count {len(members)} is outside 1..{MAX_MEMBERS}")
        names: dict[str, tarfile.TarInfo] = {}
        folded_names: set[str] = set()
        expanded_bytes = 0
        for member in members:
            normalized = str(_safe_member_name(member.name))
            if normalized in names or normalized.casefold() in folded_names:
                raise PackageAuditError(f"duplicate or case-colliding archive member: {member.name!r}")
            if not (member.isdir() or member.isfile()):
                raise PackageAuditError(f"links and special archive members are forbidden: {member.name!r}")
            names[normalized] = member
            folded_names.add(normalized.casefold())
            if member.isfile():
                expanded_bytes += member.size
        if expanded_bytes > MAX_EXPANDED_BYTES:
            raise PackageAuditError(f"archive expands to {expanded_bytes} bytes; limit is {MAX_EXPANDED_BYTES}")

        required = {
            WRAPPER,
            EXECUTABLE,
            f"{EXPECTED_ROOT}/README.md",
            f"{EXPECTED_ROOT}/requirements.txt",
        }
        missing = sorted(required - names.keys())
        if missing:
            raise PackageAuditError("archive is missing required members: " + ", ".join(missing))

        wrapper = _member_bytes(archive, names[WRAPPER])
        actual_wrapper_sha = sha256_bytes(wrapper)
        if actual_wrapper_sha != wrapper_sha256:
            raise PackageAuditError(
                f"organizer wrapper drift: expected {wrapper_sha256}, got {actual_wrapper_sha}"
            )

        executable_member = names[EXECUTABLE]
        if executable_member.mode & 0o111 == 0:
            raise PackageAuditError("optimizer executable has no execute bit")
        executable_bytes = _member_bytes(archive, executable_member)
        machine = _elf_machine(executable_bytes, EXECUTABLE)
        if machine != 62:
            raise PackageAuditError(f"optimizer ELF e_machine is {machine}; expected AMD64 (62)")

        observed_glibc: set[tuple[int, int]] = set()
        for name, member in names.items():
            lowered = name.lower()
            if "libtorch" in lowered or "/torch/_c" in lowered:
                raise PackageAuditError(f"real torch payload leaked into archive: {name}")
            if not member.isfile():
                continue
            data = _member_bytes(archive, member)
            if data[:4] == b"\x7fELF":
                observed_glibc.update((int(major), int(minor)) for major, minor in GLIBC_RE.findall(data))
        highest_glibc = max(observed_glibc, default=(0, 0))
        if highest_glibc > max_glibc:
            raise PackageAuditError(
                f"archive requires GLIBC_{highest_glibc[0]}.{highest_glibc[1]}, "
                f"newer than target GLIBC_{max_glibc[0]}.{max_glibc[1]}"
            )

        if expected_source_sha256 is not None:
            for packaged_name, expected_digest in expected_source_sha256.items():
                archive_name = f"{EXPECTED_ROOT}/{packaged_name}"
                member = names.get(archive_name)
                if member is None:
                    raise PackageAuditError(f"archive lacks manifest-bound source {archive_name}")
                actual_digest = sha256_bytes(_member_bytes(archive, member))
                if actual_digest != expected_digest:
                    raise PackageAuditError(
                        f"manifest-bound source hash mismatch for {packaged_name}: "
                        f"expected {expected_digest}, got {actual_digest}"
                    )
        elif source_root is not None:
            source_root = source_root.resolve()
            for local_name, packaged_name in SOURCE_BINDINGS.items():
                archive_name = f"{EXPECTED_ROOT}/{packaged_name}"
                member = names.get(archive_name)
                if member is None:
                    raise PackageAuditError(f"archive lacks source binding {archive_name}")
                local_path = source_root / local_name
                if not local_path.is_file():
                    raise PackageAuditError(f"local source binding is missing: {local_path}")
                if _member_bytes(archive, member) != local_path.read_bytes():
                    raise PackageAuditError(f"packaged source differs from {local_name}")

        notice_names = {
            f"{EXPECTED_ROOT}/THIRD_PARTY_NOTICES.md": ROOT / "THIRD_PARTY_NOTICES.md",
            f"{EXPECTED_ROOT}/LICENSES/Apache-2.0.txt": ROOT / "LICENSES" / "Apache-2.0.txt",
        }
        if require_notices:
            for archive_name, local_path in notice_names.items():
                member = names.get(archive_name)
                if member is None:
                    raise PackageAuditError(f"archive lacks required notice {archive_name}")
                if not local_path.is_file() or _member_bytes(archive, member) != local_path.read_bytes():
                    raise PackageAuditError(f"archive notice differs from {local_path.relative_to(ROOT)}")

        if smoke:
            with tempfile.TemporaryDirectory(prefix="opc-package-audit-") as temporary:
                # Every member was already proven relative, unique, regular or
                # a directory, and within the size/count limits above.
                archive.extractall(temporary, filter="fully_trusted")
                _smoke(Path(temporary))

    glibc_text = f"{highest_glibc[0]}.{highest_glibc[1]}" if observed_glibc else "none"
    return [
        f"archive_sha256={actual_archive_sha}",
        f"members={len(members)} expanded_bytes={expanded_bytes}",
        "elf_machine=AMD64",
        f"max_glibc={glibc_text}",
        f"smoke={'PASS' if smoke else 'SKIPPED'}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", nargs="?", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--official-sources", type=Path, default=DEFAULT_OFFICIAL_SOURCES)
    parser.add_argument("--release-manifest", type=Path)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--require-notices", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-glibc", default="2.41", metavar="MAJOR.MINOR")
    args = parser.parse_args()

    try:
        major_text, minor_text = args.max_glibc.split(".", 1)
        max_glibc = (int(major_text), int(minor_text))
        wrapper_sha = official_wrapper_sha(args.official_sources)
        expected_sha = release_package_sha(args.release_manifest) if args.release_manifest else None
        expected_sources = release_source_hashes(args.release_manifest) if args.release_manifest else None
        messages = audit_archive(
            args.archive,
            wrapper_sha256=wrapper_sha,
            expected_archive_sha256=expected_sha,
            source_root=None if args.release_manifest else args.source_root,
            expected_source_sha256=expected_sources,
            require_notices=args.require_notices,
            max_glibc=max_glibc,
            smoke=args.smoke,
        )
    except (PackageAuditError, ValueError) as exc:
        print(f"Submission package audit: FAIL\n  {exc}")
        sys.exit(1)

    print("Submission package audit: PASS")
    for message in messages:
        print(f"  {message}")


if __name__ == "__main__":
    main()
