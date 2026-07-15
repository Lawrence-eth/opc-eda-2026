#!/usr/bin/env python3
"""Audit the preserved native tournament ZIP against a release manifest.

The Actions artifact is intentionally kept outside Git.  This verifier checks
the separately hashed REST-downloaded ZIP, every file named by its checksum
ledger, the schema-2 build/schema-3 timing authority, the unchanged selected
source variant, and exact selected-package parity with the tracked source
result and release archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import statistics
import sys
from typing import Any
import zipfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import package_mode_tournament


MODES = ("off", "replacement", "additive", "additive_first_pass")
RELEASE_SCHEMA_VERSION = 2
BUILD_SCHEMA_VERSION = 2
TIMING_SCHEMA_VERSION = 3
BUILD_MODE = "four_mode_submission_package_build"
TIMING_MODE = "organizer_wrapper_williams_timing"
REPOSITORY = "Lawrence-eth/opc-eda-2026"
LEDGER_NAME = "native-evidence.sha256"
BUILD_MANIFEST_NAME = "opc-four-mode-build/build_manifest.json"
TIMING_MANIFEST_NAME = "opc-four-mode-timing/timing_manifest.json"
ENVIRONMENT_NAME = "native-environment.txt"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
RUN_URL_RE = re.compile(
    rf"^https://github\.com/{re.escape(REPOSITORY)}/actions/runs/([1-9][0-9]*)$"
)
LEDGER_RE = re.compile(r"^([0-9a-f]{64})  (.+)$")
MAX_FILES = 10_000
MAX_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
MAX_JSON_BYTES = 16 * 1024 * 1024


class NativeEvidenceError(ValueError):
    """A stable failure from the native release-evidence gate."""


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_stream(handle: Any) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NativeEvidenceError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise NativeEvidenceError(f"non-finite JSON constant is forbidden: {value}")


def _load_json_bytes(raw: bytes, description: str) -> dict[str, Any]:
    if len(raw) > MAX_JSON_BYTES:
        raise NativeEvidenceError(f"{description} exceeds the JSON size limit")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_no_duplicate_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NativeEvidenceError(f"invalid {description}: {error}") from error
    if not isinstance(value, dict):
        raise NativeEvidenceError(f"{description} must be a JSON object")
    return value


def _load_json_path(path: Path, description: str) -> dict[str, Any]:
    try:
        return _load_json_bytes(path.read_bytes(), description)
    except OSError as error:
        raise NativeEvidenceError(f"cannot read {description}: {error}") from error


def _safe_member_name(name: str) -> str:
    pure = PurePosixPath(name)
    if (
        not name
        or name.startswith("/")
        or "\\" in name
        or str(pure) != name
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise NativeEvidenceError(f"unsafe ZIP member name: {name!r}")
    return name


def _require_sha256(value: Any, description: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise NativeEvidenceError(f"{description} is not a lowercase SHA-256")
    return value


def _require_dict(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise NativeEvidenceError(f"{description} must be an object")
    return value


def _require_int(value: Any, description: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise NativeEvidenceError(
            f"{description} must be an integer greater than or equal to {minimum}"
        )
    return value


def _require_number(
    value: Any, description: str, *, minimum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NativeEvidenceError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise NativeEvidenceError(f"{description} is outside its allowed range")
    return result


def _require_commit(value: Any, description: str) -> str:
    if not isinstance(value, str) or COMMIT_RE.fullmatch(value) is None:
        raise NativeEvidenceError(f"{description} is not a full lowercase Git commit")
    return value


def _require_descriptor(value: Any, description: str) -> dict[str, Any]:
    descriptor = _require_dict(value, description)
    if set(descriptor) != {"path", "sha256", "size_bytes"}:
        raise NativeEvidenceError(f"{description} has an unsupported schema")
    if not isinstance(descriptor.get("path"), str):
        raise NativeEvidenceError(f"{description} path must be a string")
    _safe_member_name(descriptor["path"])
    _require_sha256(descriptor.get("sha256"), f"{description} digest")
    _require_int(descriptor.get("size_bytes"), f"{description} size", minimum=1)
    return descriptor


def _repo_path(root: Path, value: Any, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise NativeEvidenceError(f"{description} must be a repository-relative path")
    candidate = Path(value)
    if candidate.is_absolute():
        raise NativeEvidenceError(f"{description} must be repository-relative")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise NativeEvidenceError(f"{description} escapes the repository") from error
    return resolved


def _williams_sequences() -> tuple[tuple[str, ...], ...]:
    return (
        ("off", "replacement", "additive_first_pass", "additive"),
        ("replacement", "additive", "off", "additive_first_pass"),
        ("additive", "additive_first_pass", "replacement", "off"),
        ("additive_first_pass", "off", "additive", "replacement"),
    )


def _quality(value: Any, description: str) -> dict[str, Any]:
    quality = _require_dict(value, description)
    expected = {
        "test_id",
        "block_count",
        "is_feasible",
        "hpwl_gap",
        "area_gap",
        "violations_relative",
        "cost",
        "positions_sha256",
    }
    if set(quality) != expected:
        raise NativeEvidenceError(f"{description} has an unsupported schema")
    _require_int(quality.get("test_id"), f"{description} test ID")
    _require_int(quality.get("block_count"), f"{description} block count", minimum=1)
    if quality.get("is_feasible") is not True:
        raise NativeEvidenceError(f"{description} is infeasible")
    for field in ("hpwl_gap", "area_gap", "cost"):
        _require_number(quality.get(field), f"{description} {field}")
    _require_number(
        quality.get("violations_relative"),
        f"{description} violations_relative",
        minimum=0.0,
    )
    _require_sha256(quality.get("positions_sha256"), f"{description} positions")
    return quality


def _execution_descriptor(
    row: Any,
    expected: dict[str, Any],
    *,
    ledger: dict[str, str],
    member_sizes: dict[str, int],
    raw_results: dict[str, bytes],
    prefix: str,
) -> dict[str, Any]:
    row = _require_dict(row, "native execution")
    required = {
        "ordinal",
        "discarded_warmup",
        "case_id",
        "case_index",
        "cycle",
        "schedule_position",
        "sequence",
        "period",
        "mode",
        "runtime_seconds",
        "quality",
        "result",
        "log",
    }
    if set(row) != required:
        raise NativeEvidenceError("native execution has an unsupported schema")
    for field, value in expected.items():
        if row.get(field) != value:
            raise NativeEvidenceError(
                f"native execution {row.get('ordinal')!r} has invalid {field}"
            )
    runtime = _require_number(
        row.get("runtime_seconds"), "native execution runtime", minimum=0.0
    )
    if runtime <= 0.0:
        raise NativeEvidenceError("native execution runtime must be positive")
    quality = _quality(row.get("quality"), "native execution quality")
    if (
        quality.get("test_id") != expected["case_id"]
        or quality.get("block_count") != expected["case_id"] + 21
    ):
        raise NativeEvidenceError("native execution quality identifies the wrong case")
    kind = "warmup" if expected["discarded_warmup"] else "run"
    directory = "warmups" if expected["discarded_warmup"] else "runs"
    stem = (
        f"{kind}-{expected['ordinal']:05d}-case-{expected['case_id']:02d}-"
        f"cycle-{expected['cycle']:02d}-schedule-{expected['schedule_position']}-"
        f"sequence-{expected['sequence']}-period-{expected['period']}-"
        f"{expected['mode']}"
    )
    expected_paths = {
        "result": f"{directory}/{stem}.json",
        "log": f"logs/{stem}.log",
    }
    result_member = ""
    for field in ("result", "log"):
        descriptor = _require_descriptor(row.get(field), f"native {field}")
        if descriptor.get("path") != expected_paths[field]:
            raise NativeEvidenceError(f"native execution {field} path changed")
        member = f"{prefix}/{descriptor['path']}"
        if (
            ledger.get(member) != descriptor["sha256"]
            or member_sizes.get(member) != descriptor["size_bytes"]
        ):
            raise NativeEvidenceError(
                f"native execution {field} is not bound by the checksum ledger"
            )
        if field == "result":
            result_member = member
    _validate_raw_result(
        raw_results.get(result_member),
        row,
        description=f"raw result for execution {expected['ordinal']}",
    )
    return row


def _recomputed_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mode in MODES:
        selected = [row for row in runs if row["mode"] == mode]
        runtimes = [float(row["runtime_seconds"]) for row in selected]
        by_case: dict[str, Any] = {}
        for case_id in range(100):
            values = [
                float(row["runtime_seconds"])
                for row in selected
                if row["case_id"] == case_id
            ]
            by_case[str(case_id)] = {
                "runs": len(values),
                "mean_seconds": statistics.fmean(values),
                "median_seconds": statistics.median(values),
            }
        result[mode] = {
            "runs": len(runtimes),
            "mean_seconds": statistics.fmean(runtimes),
            "median_seconds": statistics.median(runtimes),
            "minimum_seconds": min(runtimes),
            "maximum_seconds": max(runtimes),
            "by_case": by_case,
        }
    return result


def _parse_ledger(raw: bytes) -> dict[str, str]:
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise NativeEvidenceError("checksum ledger is not UTF-8") from error
    if not lines:
        raise NativeEvidenceError("checksum ledger is empty")
    result: dict[str, str] = {}
    for index, line in enumerate(lines, start=1):
        match = LEDGER_RE.fullmatch(line)
        if match is None:
            raise NativeEvidenceError(f"malformed checksum ledger line {index}")
        digest, name = match.groups()
        name = _safe_member_name(name)
        if name == LEDGER_NAME or name in result:
            raise NativeEvidenceError(f"duplicate or recursive ledger path: {name}")
        result[name] = digest
    return result


def _canonical_positions_sha256(value: Any, expected_count: int) -> str:
    if not isinstance(value, list) or len(value) != expected_count:
        raise NativeEvidenceError("source result positions have the wrong length")
    rows: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            raise NativeEvidenceError("source result contains a malformed rectangle")
        numbers = []
        for item in row:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise NativeEvidenceError("source result position is not numeric")
            number = float(item)
            if not math.isfinite(number):
                raise NativeEvidenceError("source result position is non-finite")
            numbers.append(number)
        if numbers[2] <= 0.0 or numbers[3] <= 0.0:
            raise NativeEvidenceError("source result has non-positive dimensions")
        rows.append(numbers)
    encoded = json.dumps(
        rows, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_raw_result(
    raw: bytes | None,
    execution: dict[str, Any],
    *,
    description: str,
) -> None:
    if raw is None:
        raise NativeEvidenceError(f"{description} is missing")
    payload = _load_json_bytes(raw, description)
    rows = payload.get("test_results")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise NativeEvidenceError(f"{description} is not a single-case result")
    row = rows[0]
    case_id = execution["case_id"]
    block_count = case_id + 21
    if (
        row.get("test_id") != case_id
        or row.get("block_count") != block_count
        or row.get("is_feasible") is not True
        or row.get("error") is not None
    ):
        raise NativeEvidenceError(f"{description} identifies the wrong case")
    runtime = _require_number(
        row.get("runtime_seconds"), f"{description} runtime", minimum=0.0
    )
    if runtime <= 0.0 or runtime != execution["runtime_seconds"]:
        raise NativeEvidenceError(f"{description} runtime differs from timing manifest")
    quality = {
        "test_id": case_id,
        "block_count": block_count,
        "is_feasible": True,
        "hpwl_gap": _require_number(row.get("hpwl_gap"), f"{description} HPWL gap"),
        "area_gap": _require_number(row.get("area_gap"), f"{description} area gap"),
        "violations_relative": _require_number(
            row.get("violations_relative"),
            f"{description} violations",
            minimum=0.0,
        ),
        "cost": _require_number(row.get("cost"), f"{description} cost"),
        "positions_sha256": _canonical_positions_sha256(
            row.get("positions"), block_count
        ),
    }
    if quality != execution["quality"]:
        raise NativeEvidenceError(f"{description} quality differs from timing manifest")
    summary = _require_dict(payload.get("summary"), f"{description} summary")
    if (
        summary.get("num_tests") != 1
        or summary.get("num_feasible") != 1
        or summary.get("avg_cost") != quality["cost"]
        or summary.get("avg_runtime") != runtime
        or payload.get("total_score") != quality["cost"]
    ):
        raise NativeEvidenceError(f"{description} summary is inconsistent")


def _environment_bindings(raw: bytes) -> dict[str, str]:
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise NativeEvidenceError("native environment is not UTF-8") from error
    result: dict[str, str] = {}
    for line in lines[:8]:
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def audit_native_evidence(
    artifact_zip: Path,
    release_manifest: Path,
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Return concise verified bindings or raise :class:`NativeEvidenceError`."""

    artifact_zip = artifact_zip.resolve()
    root = root.resolve()
    manifest = _load_json_path(release_manifest, "release manifest")
    if manifest.get("schema_version") != RELEASE_SCHEMA_VERSION:
        raise NativeEvidenceError(
            f"release manifest schema must be {RELEASE_SCHEMA_VERSION}"
        )
    decision = _require_dict(manifest.get("decision_evidence"), "decision evidence")
    native = _require_dict(
        decision.get("native_tournament"), "native tournament evidence"
    )
    solver = _require_dict(manifest.get("solver"), "release solver")
    package = _require_dict(
        manifest.get("submission_package"), "release submission package"
    )
    public_result = _require_dict(
        manifest.get("public_result"), "release public result"
    )
    floorset = _require_dict(manifest.get("floorset"), "release FloorSet")

    run_id = _require_int(native.get("run_id"), "native run ID", minimum=1)
    run_attempt = _require_int(
        native.get("run_attempt"), "native run attempt", minimum=1
    )
    run_url = native.get("run_url")
    run_match = RUN_URL_RE.fullmatch(run_url) if isinstance(run_url, str) else None
    if run_match is None or int(run_match.group(1)) != run_id:
        raise NativeEvidenceError("native run URL does not bind the native run ID")
    head_sha = _require_commit(native.get("head_sha"), "native head")
    if native.get("conclusion") != "success":
        raise NativeEvidenceError("native workflow conclusion is not success")
    _require_int(native.get("artifact_id"), "native artifact ID", minimum=1)
    _require_int(native.get("artifact_size_bytes"), "native artifact size", minimum=1)
    _require_sha256(native.get("artifact_digest_sha256"), "native API artifact digest")
    expected_artifact_name = f"native-package-tournament-{head_sha}-{run_attempt}"
    if native.get("artifact_name") != expected_artifact_name:
        raise NativeEvidenceError("native artifact name does not bind head and attempt")
    if native.get("preserved_asset_name") != f"{expected_artifact_name}.zip":
        raise NativeEvidenceError("preserved evidence name does not bind native artifact")
    if native.get("build_manifest_schema_version") != BUILD_SCHEMA_VERSION:
        raise NativeEvidenceError(
            f"native build schema must be {BUILD_SCHEMA_VERSION}"
        )
    if native.get("timing_manifest_schema_version") != TIMING_SCHEMA_VERSION:
        raise NativeEvidenceError(
            f"native timing schema must be {TIMING_SCHEMA_VERSION}"
        )
    if head_sha != _require_commit(solver.get("commit"), "release solver commit"):
        raise NativeEvidenceError("native head differs from release solver commit")

    package_sha = _require_sha256(package.get("sha256"), "release package digest")
    package_path = _repo_path(root, package.get("path"), "release package path")
    if not package_path.is_file() or _sha256_path(package_path) != package_sha:
        raise NativeEvidenceError("release package bytes differ from release manifest")
    release_sources = _require_dict(solver.get("sources"), "release solver sources")
    if not release_sources:
        raise NativeEvidenceError("release solver source inventory is empty")
    for name, digest in release_sources.items():
        source_path = _repo_path(root, name, f"release solver source {name!r}")
        expected_digest = _require_sha256(
            digest, f"release solver source {name!r} digest"
        )
        if not source_path.is_file() or _sha256_path(source_path) != expected_digest:
            raise NativeEvidenceError(
                f"release solver source bytes differ from manifest: {name}"
            )

    preserved_sha = _require_sha256(
        native.get("preserved_asset_sha256"), "preserved evidence ZIP digest"
    )
    if not artifact_zip.is_file():
        raise NativeEvidenceError(f"preserved evidence ZIP is missing: {artifact_zip}")
    if artifact_zip.name != native.get("preserved_asset_name"):
        raise NativeEvidenceError("preserved evidence ZIP name differs from manifest")
    preserved_size = _require_int(
        native.get("preserved_asset_size_bytes"),
        "preserved evidence ZIP size",
        minimum=1,
    )
    if artifact_zip.stat().st_size != preserved_size:
        raise NativeEvidenceError("preserved evidence ZIP size differs from manifest")
    if _sha256_path(artifact_zip) != preserved_sha:
        raise NativeEvidenceError("preserved evidence ZIP digest differs from manifest")

    try:
        archive = zipfile.ZipFile(artifact_zip)
    except (OSError, zipfile.BadZipFile) as error:
        raise NativeEvidenceError(f"cannot open preserved evidence ZIP: {error}") from error
    with archive:
        infos = archive.infolist()
        if not infos or len(infos) > MAX_FILES:
            raise NativeEvidenceError("preserved evidence ZIP has an invalid file count")
        if sum(info.file_size for info in infos) > MAX_UNCOMPRESSED_BYTES:
            raise NativeEvidenceError("preserved evidence ZIP exceeds the size limit")
        members: dict[str, zipfile.ZipInfo] = {}
        for info in infos:
            name = _safe_member_name(info.filename)
            if info.is_dir() or name in members or info.flag_bits & 1:
                raise NativeEvidenceError(
                    f"directory, duplicate, or encrypted ZIP member: {name}"
                )
            mode = (info.external_attr >> 16) & 0xFFFF
            if mode and not stat.S_ISREG(mode):
                raise NativeEvidenceError(f"non-regular ZIP member: {name}")
            members[name] = info
        member_sizes = {name: info.file_size for name, info in members.items()}

        for required in (
            LEDGER_NAME,
            BUILD_MANIFEST_NAME,
            TIMING_MANIFEST_NAME,
            ENVIRONMENT_NAME,
        ):
            if required not in members:
                raise NativeEvidenceError(f"preserved evidence lacks {required}")
        for required in (
            LEDGER_NAME,
            BUILD_MANIFEST_NAME,
            TIMING_MANIFEST_NAME,
            ENVIRONMENT_NAME,
        ):
            if members[required].file_size > MAX_JSON_BYTES:
                raise NativeEvidenceError(f"preserved evidence member is too large: {required}")

        ledger_raw = archive.read(LEDGER_NAME)
        if hashlib.sha256(ledger_raw).hexdigest() != _require_sha256(
            native.get("evidence_bundle_sha256"), "checksum ledger digest"
        ):
            raise NativeEvidenceError("checksum ledger digest differs from manifest")
        ledger = _parse_ledger(ledger_raw)
        if set(ledger) != set(members) - {LEDGER_NAME}:
            raise NativeEvidenceError("checksum ledger and ZIP inventory differ")
        for name, expected in ledger.items():
            with archive.open(members[name]) as handle:
                if _sha256_stream(handle) != expected:
                    raise NativeEvidenceError(f"checksum mismatch for {name}")

        raw_result_names = [
            name
            for name in members
            if (
                name.startswith("opc-four-mode-timing/runs/")
                or name.startswith("opc-four-mode-timing/warmups/")
            )
            and name.endswith(".json")
        ]
        if len(raw_result_names) != 2000:
            raise NativeEvidenceError("preserved evidence lacks raw execution results")
        raw_results: dict[str, bytes] = {}
        for name in raw_result_names:
            if members[name].file_size > MAX_JSON_BYTES:
                raise NativeEvidenceError(f"raw execution result is too large: {name}")
            raw_results[name] = archive.read(name)

        build_raw = archive.read(BUILD_MANIFEST_NAME)
        timing_raw = archive.read(TIMING_MANIFEST_NAME)
        environment_raw = archive.read(ENVIRONMENT_NAME)

    if hashlib.sha256(build_raw).hexdigest() != _require_sha256(
        native.get("build_manifest_sha256"), "build manifest digest"
    ):
        raise NativeEvidenceError("build manifest digest differs from release manifest")
    if hashlib.sha256(timing_raw).hexdigest() != _require_sha256(
        native.get("timing_manifest_sha256"), "timing manifest digest"
    ):
        raise NativeEvidenceError("timing manifest digest differs from release manifest")

    build = _load_json_bytes(build_raw, "native build manifest")
    timing = _load_json_bytes(timing_raw, "native timing manifest")
    selected_mode = native.get("selected_mode")
    selected_package_sha = native.get("selected_package_sha256")
    if selected_mode not in MODES or selected_mode != solver.get("learned_order_mode"):
        raise NativeEvidenceError("selected native mode differs from release solver")
    if selected_package_sha != package_sha:
        raise NativeEvidenceError("selected native package differs from release package")
    if (
        build.get("schema_version") != BUILD_SCHEMA_VERSION
        or build.get("mode") != BUILD_MODE
    ):
        raise NativeEvidenceError("native build manifest has an unsupported authority")
    if (
        timing.get("schema_version") != TIMING_SCHEMA_VERSION
        or timing.get("mode") != TIMING_MODE
    ):
        raise NativeEvidenceError("native timing manifest has an unsupported authority")
    build_source = _require_dict(build.get("source"), "native build source")
    if build_source.get("commit") != head_sha or head_sha != solver.get("commit"):
        raise NativeEvidenceError("native build head differs from release solver commit")
    _require_commit(build_source.get("tree"), "native build source tree")
    for field in ("git_archive_sha256", "base_optimizer_sha256"):
        _require_sha256(build_source.get(field), f"native build source {field}")
    build_contract = _require_dict(build.get("contract"), "native build contract")
    if (
        build_contract.get("modes") != list(MODES)
        or build_contract.get("timing_design")
        != [list(row) for row in _williams_sequences()]
    ):
        raise NativeEvidenceError("native build contract changed")
    build_tooling = _require_dict(build.get("tooling"), "native build tooling")
    if not build_tooling:
        raise NativeEvidenceError("native build tooling inventory is empty")
    for name, digest in build_tooling.items():
        _require_sha256(digest, f"native build tooling {name}")

    variants = build.get("variants")
    if (
        not isinstance(variants, list)
        or any(not isinstance(row, dict) for row in variants)
        or [row.get("mode") for row in variants] != list(MODES)
    ):
        raise NativeEvidenceError("native build lacks the ordered four-mode panel")
    package_descriptors: dict[str, dict[str, Any]] = {}
    binary_descriptors: dict[str, dict[str, Any]] = {}
    wrapper_descriptors: dict[str, dict[str, Any]] = {}
    for row in variants:
        mode = row["mode"]
        descriptor = _require_descriptor(row.get("package"), f"{mode} build package")
        if descriptor.get("path") != f"packages/iccad2026_submission_{mode}.tar.gz":
            raise NativeEvidenceError(f"native {mode} package path changed")
        package_member = f"opc-four-mode-build/{descriptor['path']}"
        if (
            ledger.get(package_member) != descriptor["sha256"]
            or member_sizes.get(package_member) != descriptor["size_bytes"]
        ):
            raise NativeEvidenceError(f"native {mode} package is not ledger-bound")
        patch = _require_dict(row.get("source_patch"), f"{mode} source patch")
        expected_assignment = f'self._learned_order_mode = "{mode}"'
        if (
            patch.get("path") != "contest_solution/my_optimizer.py"
            or patch.get("assignment_before")
            != 'self._learned_order_mode = "replacement"'
            or patch.get("assignment_after") != expected_assignment
            or patch.get("changed") is not (mode != "replacement")
        ):
            raise NativeEvidenceError(f"native {mode} source-patch contract changed")
        if (patch.get("sha256_before") == patch.get("sha256_after")) is not (
            mode == "replacement"
        ):
            raise NativeEvidenceError(f"native {mode} source-patch hashes disagree")
        _require_sha256(patch.get("sha256_before"), f"{mode} patch-before digest")
        _require_sha256(patch.get("sha256_after"), f"{mode} patch-after digest")
        if patch.get("sha256_before") != build_source.get("base_optimizer_sha256"):
            raise NativeEvidenceError(f"native {mode} patch baseline changed")
        solver_components = _require_dict(
            row.get("solver_components"), f"{mode} solver components"
        )
        if (
            solver_components.get("contest_solution/my_optimizer.py")
            != patch.get("sha256_after")
            or any(
                _require_sha256(digest, f"native {mode} source {name}") != digest
                for name, digest in solver_components.items()
            )
        ):
            raise NativeEvidenceError(f"native {mode} solver inventory is inconsistent")
        support_sources = _require_dict(
            row.get("package_support_sources"), f"{mode} package support sources"
        )
        for name, digest in support_sources.items():
            _require_sha256(digest, f"native {mode} support source {name}")
        binary = _require_descriptor(row.get("binary"), f"{mode} build binary")
        wrapper = _require_descriptor(row.get("wrapper"), f"{mode} build wrapper")
        if (
            binary.get("path") != "iccad2026_submission/dist/my_optimizer/my_optimizer"
            or wrapper.get("path") != "iccad2026_submission/op_wrapper.py"
            or wrapper.get("sha256") != build_tooling.get("organizer_wrapper_sha256")
        ):
            raise NativeEvidenceError(f"native {mode} executable contract changed")
        mode_audit = _require_dict(row.get("audit"), f"{mode} package audit")
        details = _require_dict(mode_audit.get("details"), f"{mode} audit details")
        if (
            mode_audit.get("status") != "PASS"
            or details.get("archive_sha256") != descriptor.get("sha256")
            or details.get("elf_machine") != "AMD64"
            or details.get("smoke") != "PASS"
            or details.get("default_mode") != mode
        ):
            raise NativeEvidenceError(f"native {mode} package audit is inconsistent")
        package_descriptors[mode] = descriptor
        binary_descriptors[mode] = binary
        wrapper_descriptors[mode] = wrapper
    selected = next(row for row in variants if row.get("mode") == selected_mode)
    patch = _require_dict(selected.get("source_patch"), "selected source patch")
    if patch.get("changed") is not False or native.get(
        "selected_source_patch_changed"
    ) is not False:
        raise NativeEvidenceError("selected native variant patched committed source")
    if patch.get("sha256_before") != patch.get("sha256_after"):
        raise NativeEvidenceError("unchanged selected source has differing hashes")
    selected_package = _require_dict(selected.get("package"), "selected package")
    if selected_package.get("sha256") != package_sha:
        raise NativeEvidenceError("selected build package digest differs from release")
    selected_member = f"opc-four-mode-build/{selected_package.get('path')}"
    if ledger.get(selected_member) != package_sha:
        raise NativeEvidenceError("checksum ledger does not bind selected package")
    audit = _require_dict(selected.get("audit"), "selected package audit")
    audit_details = _require_dict(audit.get("details"), "selected audit details")
    if (
        audit.get("status") != "PASS"
        or audit_details.get("archive_sha256") != package_sha
        or audit_details.get("elf_machine") != "AMD64"
        or audit_details.get("smoke") != "PASS"
        or audit_details.get("default_mode") != selected_mode
    ):
        raise NativeEvidenceError("selected package build audit is inconsistent")

    selected_sources: dict[str, str] = {}
    for source_group in ("solver_components", "package_support_sources"):
        values = _require_dict(selected.get(source_group), source_group)
        if set(selected_sources).intersection(values):
            raise NativeEvidenceError("native selected source inventories overlap")
        selected_sources.update(values)
    if selected_sources != release_sources:
        raise NativeEvidenceError("native selected source inventory differs from release")

    source_verification = _require_dict(
        timing.get("build_source_verification"), "timing build-source verification"
    )
    if (
        source_verification.get("status") != "PASS"
        or source_verification.get("repository_head") != head_sha
        or source_verification.get("repository_tree") != build_source.get("tree")
        or source_verification.get("git_archive_sha256")
        != build_source.get("git_archive_sha256")
        or source_verification.get("tracked_worktree") != "clean"
        or source_verification.get("tooling_sha256") != build_tooling
        or source_verification.get("variant_source_contract") != "PASS"
    ):
        raise NativeEvidenceError("timing build-source replay authority is inconsistent")

    timing_build = _require_dict(timing.get("build_manifest"), "timing build binding")
    if (
        timing_build.get("sha256") != hashlib.sha256(build_raw).hexdigest()
        or timing_build.get("source_commit") != head_sha
    ):
        raise NativeEvidenceError("timing manifest does not bind the build manifest")
    bindings = _require_dict(timing.get("artifact_bindings"), "timing bindings")
    timing_packages = _require_dict(bindings.get("packages"), "timing packages")
    if timing_packages != package_descriptors:
        raise NativeEvidenceError("timing packages differ from build packages")
    binding_build = _require_descriptor(
        bindings.get("build_manifest"), "timing build-manifest artifact"
    )
    if (
        binding_build.get("path") != "build_manifest.json"
        or binding_build.get("sha256") != hashlib.sha256(build_raw).hexdigest()
        or binding_build.get("size_bytes") != len(build_raw)
    ):
        raise NativeEvidenceError("timing build artifact descriptor is inconsistent")
    official_sources_binding = _require_descriptor(
        bindings.get("official_sources"), "timing official-sources artifact"
    )
    evaluator_binding = _require_descriptor(
        bindings.get("official_evaluator"), "timing official evaluator"
    )
    checker_binding = _require_descriptor(
        bindings.get("official_source_checker"), "timing official-source checker"
    )
    if (
        official_sources_binding.get("sha256")
        != build_tooling.get("official_sources_sha256")
        or checker_binding.get("sha256")
        != build_tooling.get("official_source_checker_sha256")
        or bindings.get("extracted_wrappers") != wrapper_descriptors
        or bindings.get("extracted_binaries") != binary_descriptors
    ):
        raise NativeEvidenceError("timing executable/source bindings changed")
    timing_contract = _require_dict(timing.get("contract"), "timing contract")
    if (
        timing_contract.get("modes") != list(MODES)
        or timing_contract.get("williams_sequences")
        != [list(row) for row in _williams_sequences()]
        or timing_contract.get("complete_sequences_per_case_cycle") != 4
        or timing_contract.get("no_my_opt_bin_override") is not True
        or timing_contract.get("data_root_is_verified_official_root") is not True
    ):
        raise NativeEvidenceError("native timing contract changed")
    schedule = _require_dict(timing.get("schedule"), "timing schedule")
    if (
        schedule.get("case_ids") != list(range(100))
        or schedule.get("cycles") != 1
        or schedule.get("discarded_warmup_runs") != 400
        or schedule.get("total_runs") != 1600
        or schedule.get("total_process_executions") != 2000
    ):
        raise NativeEvidenceError("native timing schedule is incomplete")
    official = _require_dict(timing.get("official"), "timing official authority")
    if (
        official.get("floorset_commit") != floorset.get("commit")
        or official.get("evaluator_sha256") != evaluator_binding.get("sha256")
        or official.get("organizer_wrapper_sha256")
        != build_tooling.get("organizer_wrapper_sha256")
    ):
        raise NativeEvidenceError("timing FloorSet commit differs from release")
    case_artifacts = official.get("selected_public_case_artifacts")
    if (
        not isinstance(case_artifacts, list)
        or len(case_artifacts) != 100
        or any(not isinstance(row, dict) for row in case_artifacts)
        or [row.get("test_id") for row in case_artifacts] != list(range(100))
        or [row.get("block_count") for row in case_artifacts]
        != list(range(21, 121))
        or bindings.get("selected_public_case_artifacts") != case_artifacts
    ):
        raise NativeEvidenceError("native timing public-case authority is incomplete")
    environment = _require_dict(timing.get("environment"), "timing environment")
    if environment.get("native_amd64_attestation") != "PASS" or _require_dict(
        environment.get("drift"), "timing drift"
    ).get("status") != "PASS":
        raise NativeEvidenceError("native timing environment attestation failed")
    rehash = _require_dict(timing.get("post_timing_rehash"), "post-timing rehash")
    canonical_binding_sha = hashlib.sha256(
        json.dumps(
            bindings,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if (
        rehash.get("status") != "PASS"
        or rehash.get("pre_sha256") != canonical_binding_sha
        or rehash.get("post_sha256") != canonical_binding_sha
    ):
        raise NativeEvidenceError("native timing inputs drifted")

    warmups = timing.get("warmups")
    runs = timing.get("runs")
    if not isinstance(warmups, list) or len(warmups) != 400:
        raise NativeEvidenceError("native timing warmups are incomplete")
    if not isinstance(runs, list) or len(runs) != 1600:
        raise NativeEvidenceError("native timing runs are incomplete")

    result_path = _repo_path(root, public_result.get("path"), "public result path")
    result_sha = _require_sha256(public_result.get("sha256"), "public result digest")
    if not result_path.is_file() or _sha256_path(result_path) != result_sha:
        raise NativeEvidenceError("public result bytes differ from release manifest")
    source_result = _load_json_path(result_path, "release public result")
    source_score = _require_number(
        source_result.get("total_score"), "release public result score"
    )
    declared_score = _require_number(
        public_result.get("total_score"), "declared public result score"
    )
    if not math.isclose(source_score, declared_score, rel_tol=0.0, abs_tol=1e-12):
        raise NativeEvidenceError("public result score differs from release manifest")
    if (
        public_result.get("num_cases") != 100
        or public_result.get("num_feasible") != 100
    ):
        raise NativeEvidenceError("release public-result authority is incomplete")
    source_rows = source_result.get("test_results")
    if not isinstance(source_rows, list) or len(source_rows) != 100:
        raise NativeEvidenceError("release public result is not a complete panel")
    source_by_case: dict[int, dict[str, Any]] = {}
    for row in source_rows:
        if not isinstance(row, dict) or row.get("test_id") in source_by_case:
            raise NativeEvidenceError("release public result has duplicate cases")
        case_id = row.get("test_id")
        block_count = row.get("block_count")
        if (
            isinstance(case_id, bool)
            or not isinstance(case_id, int)
            or isinstance(block_count, bool)
            or not isinstance(block_count, int)
            or block_count != case_id + 21
            or row.get("is_feasible") is not True
        ):
            raise NativeEvidenceError("release public result has malformed case IDs")
        source_by_case[case_id] = {
            **row,
            "positions_sha256": _canonical_positions_sha256(
                row.get("positions"), block_count
            ),
        }
    if set(source_by_case) != set(range(100)):
        raise NativeEvidenceError("release public result case set is incomplete")

    quality_by_mode_case: dict[tuple[str, int], dict[str, Any]] = {}
    selected_execution_counts = {case_id: 0 for case_id in range(100)}

    def record_execution(row: dict[str, Any]) -> None:
        mode = row["mode"]
        case_id = row["case_id"]
        quality = row["quality"]
        key = (mode, case_id)
        previous = quality_by_mode_case.setdefault(key, quality)
        if quality != previous:
            raise NativeEvidenceError(
                f"native {mode} quality changed on case {case_id}"
            )
        if mode != selected_mode:
            return
        selected_execution_counts[case_id] += 1
        source = source_by_case[case_id]
        if quality["positions_sha256"] != source["positions_sha256"]:
            raise NativeEvidenceError(
                f"selected package/source position mismatch on case {case_id}"
            )
        for field in (
            "block_count",
            "cost",
            "hpwl_gap",
            "area_gap",
            "violations_relative",
        ):
            if quality.get(field) != source.get(field):
                raise NativeEvidenceError(
                    f"selected package/source {field} mismatch on case {case_id}"
                )

    sequences = _williams_sequences()
    warmup_index = 0
    run_index = 0
    ordinal = 0
    validated_runs: list[dict[str, Any]] = []
    for case_id in range(100):
        warmup_sequence = case_id % len(sequences)
        for period, mode in enumerate(sequences[warmup_sequence]):
            ordinal += 1
            row = _execution_descriptor(
                warmups[warmup_index],
                {
                    "ordinal": ordinal,
                    "discarded_warmup": True,
                    "case_id": case_id,
                    "case_index": case_id,
                    "cycle": -1,
                    "schedule_position": period,
                    "sequence": warmup_sequence,
                    "period": period,
                    "mode": mode,
                },
                ledger=ledger,
                member_sizes=member_sizes,
                raw_results=raw_results,
                prefix="opc-four-mode-timing",
            )
            warmup_index += 1
            record_execution(row)
        for schedule_position in range(4):
            sequence_index = (case_id + schedule_position) % 4
            for period, mode in enumerate(sequences[sequence_index]):
                ordinal += 1
                row = _execution_descriptor(
                    runs[run_index],
                    {
                        "ordinal": ordinal,
                        "discarded_warmup": False,
                        "case_id": case_id,
                        "case_index": case_id,
                        "cycle": 0,
                        "schedule_position": schedule_position,
                        "sequence": sequence_index,
                        "period": period,
                        "mode": mode,
                    },
                    ledger=ledger,
                    member_sizes=member_sizes,
                    raw_results=raw_results,
                    prefix="opc-four-mode-timing",
                )
                run_index += 1
                validated_runs.append(row)
                record_execution(row)

    if ordinal != 2000 or warmup_index != 400 or run_index != 1600:
        raise NativeEvidenceError("native timing execution inventory is incomplete")
    if set(quality_by_mode_case) != {
        (mode, case_id) for mode in MODES for case_id in range(100)
    }:
        raise NativeEvidenceError("native timing mode/case quality panel is incomplete")
    if set(selected_execution_counts.values()) != {5}:
        raise NativeEvidenceError("selected package lacks all timed and warmup executions")

    summary = _require_dict(timing.get("summary"), "timing summary")
    if summary != _recomputed_summary(validated_runs):
        raise NativeEvidenceError("native timing summary differs from raw executions")
    expected_frontier = package_mode_tournament._runtime_scenario_frontier(
        validated_runs, case_artifacts
    )
    if timing.get("runtime_scenario_frontier") != expected_frontier:
        raise NativeEvidenceError(
            "native runtime scenario frontier differs from raw executions"
        )

    environment_bindings = _environment_bindings(environment_raw)
    if (
        environment_bindings.get("github_repository")
        != "Lawrence-eth/opc-eda-2026"
        or environment_bindings.get("github_sha") != head_sha
        or environment_bindings.get("github_run_id") != str(native.get("run_id"))
        or environment_bindings.get("github_run_attempt")
        != str(native.get("run_attempt"))
        or environment_bindings.get("runner_arch") != "X64"
    ):
        raise NativeEvidenceError("native environment run identity differs from release")

    return {
        "status": "PASS",
        "run_id": native.get("run_id"),
        "run_attempt": native.get("run_attempt"),
        "head_sha": head_sha,
        "selected_mode": selected_mode,
        "selected_package_sha256": package_sha,
        "files_verified": len(ledger),
        "timed_runs": len(runs),
        "feasible_runs": len(runs),
        "stable_mode_cases": len(quality_by_mode_case),
        "source_position_cases": len(selected_execution_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_zip", type=Path)
    parser.add_argument(
        "--release-manifest",
        type=Path,
        default=ROOT / "results" / "release_manifest.json",
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        result = audit_native_evidence(
            args.artifact_zip,
            args.release_manifest,
            root=args.root,
        )
    except (NativeEvidenceError, OSError, zipfile.BadZipFile) as error:
        print(f"Native release evidence audit: FAIL\n  {error}")
        raise SystemExit(1) from error
    print("Native release evidence audit: PASS")
    for key, value in result.items():
        if key != "status":
            print(f"  {key}={value}")


if __name__ == "__main__":
    main()
