#!/usr/bin/env python3
"""Build and time four non-production learned-policy package variants.

``build`` snapshots one explicit Git commit with ``git archive`` and works only
inside disposable copies.  It changes exactly the copied optimizer's frozen
default-mode assignment, builds all four packages, runs the normal package
audit/live-module self-test, and records complete source/package hashes.

``time`` rehashes and re-audits those packages, extracts each separately, and
runs the pinned official evaluator through the organizer's exact ``op_wrapper``
in a balanced four-treatment Williams sequence.  Neither command writes into
the source repository or changes the production default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import stat
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODES = ("off", "replacement", "additive", "additive_first_pass")
OPTIMIZER_RELATIVE_PATH = Path("contest_solution/my_optimizer.py")
DEFAULT_ASSIGNMENT = '        self._learned_order_mode = "replacement"'
ASSIGNMENT_RE = re.compile(
    r"^[ \t]*self\._learned_order_mode[ \t]*=.*$", re.MULTILINE
)
BUILD_MODE = "four_mode_submission_package_build"
TIMING_MODE = "organizer_wrapper_williams_timing"
BUILD_SCHEMA_VERSION = 2
TIMING_SCHEMA_VERSION = 2
ARCHIVE_ROOT = "iccad2026_submission"
INTERNAL_BINARY_PATH = f"{ARCHIVE_ROOT}/dist/my_optimizer/my_optimizer"
INTERNAL_WRAPPER_PATH = f"{ARCHIVE_ROOT}/op_wrapper.py"
RUNTIME_SCENARIO_MEDIANS = (0.25, 0.5, 1.0, 2.0, 3.0)
QUALITY_ALPHA = 0.5
VIOLATION_BETA = 2.0
RUNTIME_GAMMA = 0.3
FEASIBLE_COST_CAP = 10.0 - 1e-6
NATIVE_AMD64_NAMES = {"x86_64", "amd64"}
EMULATION_ENV_NAMES = (
    "QEMU_CPU",
    "QEMU_LD_PREFIX",
    "QEMU_SET_ENV",
    "QEMU_UNSET_ENV",
    "BOX64_LD_LIBRARY_PATH",
    "BOX64_PATH",
    "FEX_ROOTFS",
)
MAX_SNAPSHOT_MEMBERS = 100_000
MAX_SNAPSHOT_BYTES = 2 * 1024 * 1024 * 1024


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _load_object(path: Path, context: str) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"cannot load {context} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{context} {path} must contain a JSON object")
    return raw, value


def _require_exact_keys(value: Any, expected: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        observed = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(
            f"{context} must have exactly {sorted(expected)}; observed {observed}"
        )
    return value


def _require_sha256(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _require_commit(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a full lowercase Git commit")
    return value


def _require_int(value: Any, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{context} must be an integer >= {minimum}")
    return value


def _require_number(value: Any, context: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        qualifier = "finite" if minimum is None else f"finite and >= {minimum}"
        raise ValueError(f"{context} must be {qualifier}")
    return result


def _run_checked(
    command: list[str],
    *,
    cwd: Path,
    timeout: int = 120,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"command failed to start or finish: {command}: {error}") from error
    if completed.returncode != 0:
        raise RuntimeError(
            f"command exited {completed.returncode}: {command}\n"
            f"stdout tail: {completed.stdout[-1000:]}\n"
            f"stderr tail: {completed.stderr[-1000:]}"
        )
    return completed


def _git_text(repository: Path, *arguments: str) -> str:
    return _run_checked(
        ["git", "-C", str(repository), *arguments], cwd=repository
    ).stdout.strip()


def _git_bytes(repository: Path, commit: str, relative_path: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), "show", f"{commit}:{relative_path}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError(f"cannot read {relative_path} from {commit}: {error}") from error
    if completed.returncode != 0:
        raise ValueError(
            f"cannot read {relative_path} from {commit}: "
            f"{completed.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return completed.stdout


def _resolve_commit(repository: Path, reference: str) -> tuple[str, str]:
    repository = repository.resolve()
    if not (repository / ".git").exists():
        raise ValueError(f"not a Git worktree: {repository}")
    commit = _git_text(repository, "rev-parse", "--verify", f"{reference}^{{commit}}")
    _require_commit(commit, "resolved source commit")
    tree = _git_text(repository, "show", "-s", "--format=%T", commit)
    _require_commit(tree, "resolved source tree")
    return commit, tree


def _ensure_outside_repository(path: Path, repository: Path, context: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(repository.resolve())
    except ValueError:
        return resolved
    raise ValueError(f"{context} must be outside the source repository: {resolved}")


def _verify_committed_orchestrator(repository: Path, commit: str) -> None:
    relative = "scripts/package_mode_tournament.py"
    committed = _git_bytes(repository, commit, relative)
    running = Path(__file__).resolve().read_bytes()
    if committed != running:
        raise ValueError(
            "the running package-mode orchestrator is not the version bound by "
            f"source commit {commit}"
        )


def _safe_extract_tar(archive_path: Path, destination: Path, *, expected_prefix: str | None = None) -> None:
    try:
        archive = tarfile.open(archive_path, mode="r:*")
    except (OSError, tarfile.TarError) as error:
        raise ValueError(f"cannot open archive {archive_path}: {error}") from error
    with archive:
        members = archive.getmembers()
        if not members or len(members) > MAX_SNAPSHOT_MEMBERS:
            raise ValueError(f"archive has invalid member count: {len(members)}")
        names = set()
        total = 0
        for member in members:
            path = PurePosixPath(member.name.rstrip("/"))
            if (
                not member.name
                or member.name.startswith("/")
                or "\\" in member.name
                or not path.parts
                or any(part in {"", ".", ".."} for part in path.parts)
            ):
                raise ValueError(f"unsafe archive member: {member.name!r}")
            normalized = str(path)
            if normalized in names:
                raise ValueError(f"duplicate archive member: {normalized}")
            names.add(normalized)
            if expected_prefix is not None and path.parts[0] != expected_prefix:
                raise ValueError(f"archive member is outside {expected_prefix}/: {normalized}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"links and special archive members are forbidden: {normalized}")
            if member.isfile():
                total += member.size
        if total > MAX_SNAPSHOT_BYTES:
            raise ValueError(f"archive expands to {total} bytes")
        destination.mkdir(parents=True, exist_ok=False)
        archive.extractall(destination, filter="fully_trusted")


def _extract_git_snapshot(repository: Path, commit: str, destination: Path) -> str:
    tar_path = destination.parent / f".{destination.name}.git-archive.tar"
    if tar_path.exists():
        raise FileExistsError(tar_path)
    try:
        with tar_path.open("xb") as stream:
            completed = subprocess.run(
                ["git", "-C", str(repository), "archive", "--format=tar", commit],
                stdout=stream,
                stderr=subprocess.PIPE,
                timeout=300,
                check=False,
            )
        if completed.returncode != 0:
            raise ValueError(
                "git archive failed: "
                + completed.stderr.decode("utf-8", errors="replace").strip()
            )
        archive_sha256 = _sha256_file(tar_path)
        _safe_extract_tar(tar_path, destination)
        return archive_sha256
    finally:
        tar_path.unlink(missing_ok=True)


def _tree_hashes(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"snapshot contains a symbolic link: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = _sha256_file(path)
        elif not path.is_dir():
            raise ValueError(f"snapshot contains a special file: {path}")
    if not result:
        raise ValueError(f"snapshot contains no files: {root}")
    return result


def _verify_mode_independent_self_test(path: Path) -> None:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read package self-test {path}: {error}") from error
    required_once = (
        '    optimizer._learned_order_enabled = True',
        '    optimizer._learned_order_mode = "replacement"',
        '    abstention_optimizer._learned_order_enabled = True',
        '    abstention_optimizer._learned_order_mode = "replacement"',
    )
    missing = [line for line in required_once if source.count(line) != 1]
    if missing:
        raise ValueError(
            "package self-test is not explicitly replacement-capable: "
            + ", ".join(repr(line) for line in missing)
        )


def _default_assignment(source: str) -> str:
    assignments = ASSIGNMENT_RE.findall(source)
    if len(assignments) != 1 or assignments[0] != DEFAULT_ASSIGNMENT:
        raise ValueError(
            "optimizer must contain exactly one byte-exact frozen default assignment; "
            f"observed {assignments}"
        )
    return assignments[0]


def patch_copied_optimizer(path: Path, mode: str) -> dict[str, Any]:
    """Patch exactly one copied default assignment and return its hashes."""

    if mode not in MODES:
        raise ValueError(f"unsupported learned-policy mode: {mode!r}")
    try:
        metadata = path.stat()
        raw = path.read_bytes()
        source = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read copied optimizer {path}: {error}") from error
    _default_assignment(source)
    replacement = f'        self._learned_order_mode = "{mode}"'
    patched = source.replace(DEFAULT_ASSIGNMENT, replacement, 1)
    if ASSIGNMENT_RE.findall(patched) != [replacement]:
        raise AssertionError("mode patch did not produce exactly one assignment")
    encoded = patched.encode("utf-8")
    if mode == "replacement" and encoded != raw:
        raise AssertionError("replacement control unexpectedly changed source bytes")
    if mode != "replacement" and encoded == raw:
        raise AssertionError("non-control mode did not change source bytes")
    if encoded != raw:
        path.write_bytes(encoded)
        os.chmod(path, stat.S_IMODE(metadata.st_mode))
        os.utime(path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    return {
        "path": OPTIMIZER_RELATIVE_PATH.as_posix(),
        "assignment_before": DEFAULT_ASSIGNMENT.strip(),
        "assignment_after": replacement.strip(),
        "sha256_before": _sha256_bytes(raw),
        "sha256_after": _sha256_bytes(encoded),
        "changed": encoded != raw,
    }


def _verify_only_mode_change(
    baseline: dict[str, str], variant_root: Path, mode: str
) -> dict[str, str]:
    observed = _tree_hashes(variant_root)
    all_names = set(baseline) | set(observed)
    changed = sorted(name for name in all_names if baseline.get(name) != observed.get(name))
    expected = [] if mode == "replacement" else [OPTIMIZER_RELATIVE_PATH.as_posix()]
    if changed != expected:
        raise ValueError(
            f"{mode} copied snapshot changed {changed}; expected exactly {expected}"
        )
    return observed


def williams_sequences(treatments: tuple[str, ...] = MODES) -> tuple[tuple[str, ...], ...]:
    """Return the standard even-treatment balanced Williams Latin square."""

    count = len(treatments)
    if count < 2 or count % 2:
        raise ValueError("Williams construction requires an even treatment count >= 2")
    if len(set(treatments)) != count:
        raise ValueError("Williams treatments must be unique")
    first_indices = [0]
    for position in range(1, count):
        if position % 2:
            first_indices.append((position + 1) // 2)
        else:
            first_indices.append(count - position // 2)
    rows = []
    for offset in range(count):
        rows.append(tuple(treatments[(index + offset) % count] for index in first_indices))
    return tuple(rows)


def _component_names(snapshot: Path) -> tuple[str, ...]:
    completed = _run_checked(
        [sys.executable, str(snapshot / "scripts/solver_components.py")],
        cwd=snapshot,
    )
    names = tuple(line for line in completed.stdout.splitlines() if line)
    if not names or len(names) != len(set(names)):
        raise ValueError("isolated solver component registry is empty or duplicated")
    return names


def _audit_details(
    stdout: str,
    expected_package_sha256: str,
    expected_mode: str | None = None,
) -> dict[str, str]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines or lines[0] != "Submission package audit: PASS":
        raise ValueError("package audit did not report PASS")
    details = {}
    for line in lines[1:]:
        if "=" in line:
            key, value = line.split("=", 1)
            details[key] = value
    if details.get("archive_sha256") != expected_package_sha256:
        raise ValueError("package audit archive hash does not match built package")
    if details.get("smoke") != "PASS":
        raise ValueError("package audit did not run the live-module smoke test")
    if details.get("elf_machine") != "AMD64":
        raise ValueError("package audit did not verify an AMD64 executable")
    if expected_mode is not None and details.get("default_mode") != expected_mode:
        raise ValueError(
            "package audit binary default-mode attestation differs from variant mode"
        )
    return details


def _source_hash_contract(
    snapshot: Path, component_names: tuple[str, ...]
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Hash every registry and package-support source and its archive binding."""

    solver_sources = {
        f"contest_solution/{name}": _sha256_file(snapshot / "contest_solution" / name)
        for name in component_names
    }
    support_bindings = {
        "packaging/torch_stub.py": "source_fallback/torch.py",
        "packaging/eval_stub.py": "source_fallback/iccad2026_evaluate.py",
        "packaging/solver_main.py": "source_fallback/solver_main.py",
    }
    support_sources = {
        name: _sha256_file(snapshot / name) for name in support_bindings
    }
    archived = {
        **{
            f"source_fallback/{Path(name).name}": digest
            for name, digest in solver_sources.items()
        },
        **{
            archive_name: support_sources[source_name]
            for source_name, archive_name in support_bindings.items()
        },
    }
    return solver_sources, support_sources, archived


def _file_descriptor(path: str, actual: Path) -> dict[str, Any]:
    return {
        "path": path,
        "sha256": _sha256_file(actual),
        "size_bytes": actual.stat().st_size,
    }


def _write_log(path: Path, completed: subprocess.CompletedProcess[str]) -> str:
    text = (
        "$ " + " ".join(completed.args) + "\n"
        + "--- stdout ---\n" + completed.stdout
        + "\n--- stderr ---\n" + completed.stderr
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return _sha256_file(path)


def _clean_build_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in (
        "ICCAD_ALLOW_NATIVE_BUILD",
        "ICCAD_AMD64_BUILD_CONTAINER",
        "MY_OPT_BIN",
    ):
        environment.pop(name, None)
    environment["PYTHONHASHSEED"] = "0"
    return environment


def _build_one_variant(
    base_snapshot: Path,
    baseline_hashes: dict[str, str],
    mode: str,
    workspace: Path,
    output_staging: Path,
    timeout: int,
) -> dict[str, Any]:
    variant_root = workspace / mode
    shutil.copytree(base_snapshot, variant_root)
    patch = patch_copied_optimizer(variant_root / OPTIMIZER_RELATIVE_PATH, mode)
    _verify_only_mode_change(baseline_hashes, variant_root, mode)
    _verify_mode_independent_self_test(variant_root / "packaging/solver_main.py")
    component_names = _component_names(variant_root)
    solver_sources, support_sources, expected_archived_sources = (
        _source_hash_contract(variant_root, component_names)
    )

    build = _run_checked(
        ["bash", "packaging/build_submission.sh"],
        cwd=variant_root,
        timeout=timeout,
        env=_clean_build_environment(),
    )
    build_log = output_staging / "logs" / f"{mode}.build.log"
    build_log_sha256 = _write_log(build_log, build)

    archive = variant_root / "submission/iccad2026_submission.tar.gz"
    binary = variant_root / "submission/dist/my_optimizer/my_optimizer"
    fallback = variant_root / f"submission/{ARCHIVE_ROOT}/source_fallback"
    if not archive.is_file() or not binary.is_file() or not fallback.is_dir():
        raise ValueError(f"{mode} build did not produce the complete package layout")
    archive_sha256 = _sha256_file(archive)

    audit = _run_checked(
        [
            sys.executable,
            str(variant_root / "scripts/audit_submission_package.py"),
            str(archive),
            "--official-sources",
            str(variant_root / "docs/official_sources.json"),
            "--source-root",
            str(variant_root),
            "--require-notices",
            "--smoke",
        ],
        cwd=variant_root,
        timeout=180,
        env=_clean_build_environment(),
    )
    details = _audit_details(audit.stdout, archive_sha256, mode)
    audit_log = output_staging / "logs" / f"{mode}.audit.log"
    audit_log_sha256 = _write_log(audit_log, audit)

    archived_sources = {
        f"source_fallback/{path.name}": _sha256_file(path)
        for path in sorted(fallback.glob("*.py"))
        if path.is_file()
    }
    if archived_sources != expected_archived_sources:
        raise ValueError(f"{mode} package archived-source inventory differs from contract")
    if solver_sources[OPTIMIZER_RELATIVE_PATH.as_posix()] != patch["sha256_after"]:
        raise ValueError(f"{mode} generated optimizer differs from its source contract")

    package_relative = Path("packages") / f"iccad2026_submission_{mode}.tar.gz"
    package_output = output_staging / package_relative
    package_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(archive, package_output)
    if _sha256_file(package_output) != archive_sha256:
        raise IOError(f"{mode} package changed while publishing")

    return {
        "mode": mode,
        "source_patch": patch,
        "solver_components": solver_sources,
        "package_support_sources": support_sources,
        "archived_source_sha256": archived_sources,
        "binary": _file_descriptor(INTERNAL_BINARY_PATH, binary),
        "wrapper": _file_descriptor(
            INTERNAL_WRAPPER_PATH,
            variant_root / f"submission/{ARCHIVE_ROOT}/op_wrapper.py",
        ),
        "package": {
            "path": package_relative.as_posix(),
            "sha256": archive_sha256,
            "size_bytes": package_output.stat().st_size,
        },
        "audit": {
            "status": "PASS",
            "details": details,
            "log": {
                "path": f"logs/{mode}.audit.log",
                "sha256": audit_log_sha256,
                "size_bytes": audit_log.stat().st_size,
            },
        },
        "build_log": {
            "path": f"logs/{mode}.build.log",
            "sha256": build_log_sha256,
            "size_bytes": build_log.stat().st_size,
        },
    }


def _publish_directory(staging: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    os.replace(staging, output)


def build_mode_packages(
    *,
    repository: Path,
    reference: str,
    output_dir: Path,
    workspace_root: Path | None = None,
    timeout: int = 3600,
) -> dict[str, Any]:
    """Build all four isolated variants and atomically publish their manifest."""

    repository = repository.resolve()
    output_dir = _ensure_outside_repository(output_dir, repository, "output directory")
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if workspace_root is not None:
        workspace_root = _ensure_outside_repository(
            workspace_root, repository, "workspace root"
        )
        workspace_root.mkdir(parents=True, exist_ok=True)
    commit, tree = _resolve_commit(repository, reference)
    _verify_committed_orchestrator(repository, commit)

    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        with tempfile.TemporaryDirectory(
            prefix="opc-four-mode-build-",
            dir=workspace_root,
        ) as temporary:
            workspace = Path(temporary)
            base_snapshot = workspace / "base"
            git_archive_sha256 = _extract_git_snapshot(
                repository, commit, base_snapshot
            )
            _verify_mode_independent_self_test(
                base_snapshot / "packaging/solver_main.py"
            )
            base_optimizer = base_snapshot / OPTIMIZER_RELATIVE_PATH
            _default_assignment(base_optimizer.read_text(encoding="utf-8"))
            baseline_hashes = _tree_hashes(base_snapshot)
            variants = []
            for mode in MODES:
                print(f"building isolated {mode} package", flush=True)
                variants.append(
                    _build_one_variant(
                        base_snapshot,
                        baseline_hashes,
                        mode,
                        workspace,
                        staging,
                        timeout,
                    )
                )
            if _tree_hashes(base_snapshot) != baseline_hashes:
                raise AssertionError("immutable base snapshot changed during variant builds")

            manifest = {
                "schema_version": BUILD_SCHEMA_VERSION,
                "mode": BUILD_MODE,
                "source": {
                    "commit": commit,
                    "tree": tree,
                    "git_archive_sha256": git_archive_sha256,
                    "base_optimizer_sha256": _sha256_file(base_optimizer),
                },
                "contract": {
                    "modes": list(MODES),
                    "source_mutation": (
                        "exactly one frozen default-mode assignment in a disposable "
                        "git-archive copy; source worktree is read-only"
                    ),
                    "build_environment": (
                        "packaging/build_submission.sh pinned AMD64 Debian 13 container"
                    ),
                    "audit": "existing package audit with notices and live-module smoke",
                    "timing_design": [list(row) for row in williams_sequences()],
                },
                "tooling": {
                    "orchestrator_sha256": _sha256_file(Path(__file__).resolve()),
                    "build_submission_sha256": _sha256_file(
                        base_snapshot / "packaging/build_submission.sh"
                    ),
                    "package_audit_sha256": _sha256_file(
                        base_snapshot / "scripts/audit_submission_package.py"
                    ),
                    "package_self_test_sha256": _sha256_file(
                        base_snapshot / "packaging/solver_main.py"
                    ),
                    "official_sources_sha256": _sha256_file(
                        base_snapshot / "docs/official_sources.json"
                    ),
                    "organizer_wrapper_sha256": _sha256_file(
                        base_snapshot / "packaging/op_wrapper.py"
                    ),
                    "solver_registry_sha256": _sha256_file(
                        base_snapshot / "scripts/solver_components.py"
                    ),
                    "official_source_checker_sha256": _sha256_file(
                        base_snapshot / "scripts/check_official_sources.py"
                    ),
                },
                "build_host": {
                    "platform": platform.platform(),
                    "machine": platform.machine(),
                    "python": platform.python_version(),
                },
                "variants": variants,
            }
        manifest_path = staging / "build_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _publish_directory(staging, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _safe_manifest_path(base: Path, relative: Any, context: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{context} path must be a nonempty string")
    pure = PurePosixPath(relative)
    if (
        relative.startswith("/")
        or "\\" in relative
        or str(pure) != relative
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"unsafe {context} path: {relative!r}")
    resolved = (base / Path(*pure.parts)).resolve()
    try:
        resolved.relative_to(base.resolve())
    except ValueError as error:
        raise ValueError(f"{context} escapes manifest directory") from error
    return resolved


def _require_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a nonempty string")
    return value


def _validate_file_descriptor(
    value: Any,
    context: str,
    *,
    expected_path: str | None = None,
) -> dict[str, Any]:
    descriptor = _require_exact_keys(
        value, {"path", "sha256", "size_bytes"}, context
    )
    path = _require_string(descriptor["path"], f"{context} path")
    _safe_manifest_path(Path("/manifest"), path, context)
    if expected_path is not None and path != expected_path:
        raise ValueError(f"{context} path must be {expected_path!r}")
    _require_sha256(descriptor["sha256"], f"{context} hash")
    _require_int(descriptor["size_bytes"], f"{context} size", minimum=1)
    return descriptor


def _validate_build_manifest(
    path: Path,
) -> tuple[bytes, dict[str, Any], dict[str, dict[str, Any]]]:
    raw, manifest = _load_object(path, "four-mode build manifest")
    _require_exact_keys(
        manifest,
        {
            "schema_version",
            "mode",
            "source",
            "contract",
            "tooling",
            "build_host",
            "variants",
        },
        "four-mode build manifest",
    )
    if manifest["schema_version"] != BUILD_SCHEMA_VERSION or manifest["mode"] != BUILD_MODE:
        raise ValueError(
            f"not a schema-{BUILD_SCHEMA_VERSION} four-mode package build manifest; "
            "schema-1 manifests are intentionally rejected"
        )
    source = _require_exact_keys(
        manifest["source"],
        {"commit", "tree", "git_archive_sha256", "base_optimizer_sha256"},
        "build source",
    )
    _require_commit(source["commit"], "build source commit")
    _require_commit(source["tree"], "build source tree")
    for field in ("git_archive_sha256", "base_optimizer_sha256"):
        _require_sha256(source.get(field), f"build source {field}")
    contract = _require_exact_keys(
        manifest["contract"],
        {
            "modes",
            "source_mutation",
            "build_environment",
            "audit",
            "timing_design",
        },
        "build contract",
    )
    if contract["modes"] != list(MODES):
        raise ValueError("build manifest does not contain the complete four-mode contract")
    for field in ("source_mutation", "build_environment", "audit"):
        _require_string(contract[field], f"build contract {field}")
    if contract["timing_design"] != [list(row) for row in williams_sequences()]:
        raise ValueError("build manifest Williams design changed")
    tooling = manifest["tooling"]
    required_tooling = {
        "orchestrator_sha256",
        "build_submission_sha256",
        "package_audit_sha256",
        "package_self_test_sha256",
        "official_sources_sha256",
        "organizer_wrapper_sha256",
        "solver_registry_sha256",
        "official_source_checker_sha256",
    }
    _require_exact_keys(tooling, required_tooling, "build tooling")
    for field in required_tooling:
        _require_sha256(tooling[field], f"build tooling {field}")
    if tooling["orchestrator_sha256"] != _sha256_file(Path(__file__).resolve()):
        raise ValueError("build manifest was produced by different orchestrator bytes")
    build_host = _require_exact_keys(
        manifest["build_host"], {"platform", "machine", "python"}, "build host"
    )
    for field in build_host:
        _require_string(build_host[field], f"build host {field}")

    component_names = _component_names(ROOT)
    expected_solver_sources = {
        f"contest_solution/{name}" for name in component_names
    }
    expected_support_sources = {
        "packaging/torch_stub.py",
        "packaging/eval_stub.py",
        "packaging/solver_main.py",
    }
    expected_archived_sources = {
        *(f"source_fallback/{name}" for name in component_names),
        "source_fallback/torch.py",
        "source_fallback/iccad2026_evaluate.py",
        "source_fallback/solver_main.py",
    }
    variants = manifest["variants"]
    if not isinstance(variants, list) or len(variants) != len(MODES):
        raise ValueError("build manifest must contain exactly four variants")
    by_mode = {}
    for index, variant in enumerate(variants):
        variant = _require_exact_keys(
            variant,
            {
                "mode",
                "source_patch",
                "solver_components",
                "package_support_sources",
                "archived_source_sha256",
                "binary",
                "wrapper",
                "package",
                "audit",
                "build_log",
            },
            f"variant {index}",
        )
        mode = variant["mode"]
        if mode not in MODES or mode in by_mode:
            raise ValueError(f"variant {index} has unknown or duplicate mode")
        patch = _require_exact_keys(
            variant["source_patch"],
            {
                "path",
                "assignment_before",
                "assignment_after",
                "sha256_before",
                "sha256_after",
                "changed",
            },
            f"variant {mode} source patch",
        )
        expected_assignment = f'self._learned_order_mode = "{mode}"'
        if (
            patch["path"] != OPTIMIZER_RELATIVE_PATH.as_posix()
            or patch["assignment_before"] != DEFAULT_ASSIGNMENT.strip()
            or patch["assignment_after"] != expected_assignment
            or patch["changed"] is not (mode != "replacement")
        ):
            raise ValueError(f"variant {mode} source patch contract changed")
        _require_sha256(patch["sha256_before"], f"variant {mode} patch before")
        _require_sha256(patch["sha256_after"], f"variant {mode} patch after")
        if patch["sha256_before"] != source["base_optimizer_sha256"]:
            raise ValueError(f"variant {mode} patch is not based on frozen optimizer")
        if (patch["sha256_after"] == patch["sha256_before"]) is not (
            mode == "replacement"
        ):
            raise ValueError(f"variant {mode} patch changed-state hashes are inconsistent")

        solver_sources = variant["solver_components"]
        if not isinstance(solver_sources, dict) or set(solver_sources) != expected_solver_sources:
            raise ValueError(f"variant {mode} solver registry source inventory changed")
        support_sources = variant["package_support_sources"]
        if not isinstance(support_sources, dict) or set(support_sources) != expected_support_sources:
            raise ValueError(f"variant {mode} package-support source inventory changed")
        for group_name, group in (
            ("solver", solver_sources),
            ("support", support_sources),
        ):
            for name, digest in group.items():
                _safe_manifest_path(Path("/manifest"), name, f"variant {mode} {group_name} source")
                _require_sha256(digest, f"variant {mode} {group_name} source {name}")
        if solver_sources[OPTIMIZER_RELATIVE_PATH.as_posix()] != patch["sha256_after"]:
            raise ValueError(f"variant {mode} patch hash differs from solver inventory")

        sources = variant["archived_source_sha256"]
        if not isinstance(sources, dict) or set(sources) != expected_archived_sources:
            raise ValueError(f"variant {mode} archived source inventory changed")
        expected_archive_hashes = {
            **{
                f"source_fallback/{Path(name).name}": digest
                for name, digest in solver_sources.items()
            },
            "source_fallback/torch.py": support_sources["packaging/torch_stub.py"],
            "source_fallback/iccad2026_evaluate.py": support_sources["packaging/eval_stub.py"],
            "source_fallback/solver_main.py": support_sources["packaging/solver_main.py"],
        }
        if sources != expected_archive_hashes:
            raise ValueError(f"variant {mode} archived source hashes do not bind source inventory")

        binary = _validate_file_descriptor(
            variant["binary"], f"variant {mode} binary", expected_path=INTERNAL_BINARY_PATH
        )
        wrapper = _validate_file_descriptor(
            variant["wrapper"], f"variant {mode} wrapper", expected_path=INTERNAL_WRAPPER_PATH
        )
        if wrapper["sha256"] != tooling["organizer_wrapper_sha256"]:
            raise ValueError(f"variant {mode} wrapper differs from organizer binding")
        package = _validate_file_descriptor(
            variant["package"], f"variant {mode} package"
        )
        expected_package_path = f"packages/iccad2026_submission_{mode}.tar.gz"
        if package["path"] != expected_package_path:
            raise ValueError(f"variant {mode} package path changed")

        audit = _require_exact_keys(
            variant["audit"], {"status", "details", "log"}, f"variant {mode} audit"
        )
        if audit["status"] != "PASS":
            raise ValueError(f"variant {mode} did not pass its build-time audit")
        details = _require_exact_keys(
            audit["details"],
            {"archive_sha256", "members", "elf_machine", "max_glibc", "smoke", "default_mode"},
            f"variant {mode} audit details",
        )
        if (
            details["archive_sha256"] != package["sha256"]
            or details["elf_machine"] != "AMD64"
            or details["smoke"] != "PASS"
            or details["default_mode"] != mode
        ):
            raise ValueError(f"variant {mode} audit attestation is inconsistent")
        _require_string(details["members"], f"variant {mode} audit member summary")
        _require_string(details["max_glibc"], f"variant {mode} audit glibc")
        _validate_file_descriptor(
            audit["log"], f"variant {mode} audit log",
            expected_path=f"logs/{mode}.audit.log",
        )
        _validate_file_descriptor(
            variant["build_log"], f"variant {mode} build log",
            expected_path=f"logs/{mode}.build.log",
        )
        by_mode[mode] = variant
    if tuple(variant["mode"] for variant in variants) != MODES:
        raise ValueError("build manifest variant order changed")
    return raw, manifest, by_mode


def _canonical_positions_sha256(value: Any, expected_count: int) -> str:
    if not isinstance(value, list) or len(value) != expected_count:
        raise ValueError("official result positions have the wrong block count")
    rows = []
    for index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError(f"official result position {index} is malformed")
        rows.append(
            [
                _require_number(number, f"official position {index}")
                for number in row
            ]
        )
    encoded = json.dumps(
        rows, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _validate_official_result(
    path: Path,
    case_id: int,
    expected_block_count: int | None = None,
) -> dict[str, Any]:
    _raw, payload = _load_object(path, "official wrapper result")
    rows = payload.get("test_results")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError("official wrapper result must contain exactly one case")
    row = rows[0]
    if row.get("test_id") != case_id:
        raise ValueError("official wrapper result has the wrong test ID")
    block_count = _require_int(row.get("block_count"), "official block count", minimum=1)
    if block_count > 120:
        raise ValueError("official result block count exceeds contest maximum")
    if expected_block_count is not None and block_count != expected_block_count:
        raise ValueError(
            "official result block count differs from the selected dataset case"
        )
    if row.get("is_feasible") is not True or row.get("error") is not None:
        raise ValueError("mode package produced an infeasible or errored result")
    runtime = _require_number(row.get("runtime_seconds"), "official runtime", minimum=0.0)
    if runtime <= 0.0:
        raise ValueError("official wrapper runtime must be positive")
    quality = {
        "test_id": case_id,
        "block_count": block_count,
        "is_feasible": True,
        "hpwl_gap": _require_number(row.get("hpwl_gap"), "official hpwl gap"),
        "area_gap": _require_number(row.get("area_gap"), "official area gap"),
        "violations_relative": _require_number(
            row.get("violations_relative"), "official violations", minimum=0.0
        ),
        "cost": _require_number(row.get("cost"), "official cost", minimum=0.0),
        "positions_sha256": _canonical_positions_sha256(
            row.get("positions"), block_count
        ),
    }
    summary = payload.get("summary")
    if (
        not isinstance(summary, dict)
        or summary.get("num_tests") != 1
        or summary.get("num_feasible") != 1
    ):
        raise ValueError("official wrapper summary is inconsistent")
    return {"runtime_seconds": runtime, "quality": quality}


def _timing_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for mode in MODES:
        selected = [row for row in runs if row["mode"] == mode]
        runtimes = [row["runtime_seconds"] for row in selected]
        if not runtimes:
            raise ValueError(f"timing schedule has no {mode} runs")
        by_case = {}
        for case_id in sorted({row["case_id"] for row in selected}):
            case_times = [
                row["runtime_seconds"] for row in selected if row["case_id"] == case_id
            ]
            by_case[str(case_id)] = {
                "runs": len(case_times),
                "mean_seconds": statistics.fmean(case_times),
                "median_seconds": statistics.median(case_times),
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


def _official_scenario_cost(
    quality: dict[str, Any], runtime_seconds: float, median_runtime_seconds: float
) -> float:
    """Recompute the organizer's feasible contest cost from raw metrics."""

    if quality.get("is_feasible") is not True:
        return 10.0
    hpwl_gap = _require_number(quality.get("hpwl_gap"), "scenario HPWL gap")
    area_gap = _require_number(quality.get("area_gap"), "scenario area gap")
    violations = _require_number(
        quality.get("violations_relative"), "scenario violation ratio", minimum=0.0
    )
    runtime = _require_number(runtime_seconds, "scenario runtime", minimum=0.0)
    median = _require_number(
        median_runtime_seconds, "scenario median runtime", minimum=0.0
    )
    if runtime <= 0.0 or median <= 0.0:
        raise ValueError("scenario runtime and median must be positive")
    quality_factor = 1.0 + QUALITY_ALPHA * (
        max(0.0, hpwl_gap) + max(0.0, area_gap)
    )
    violation_factor = math.exp(VIOLATION_BETA * violations)
    runtime_factor = max(0.7, max(0.01, runtime / max(median, 0.01)) ** RUNTIME_GAMMA)
    return min(
        quality_factor * violation_factor * runtime_factor,
        FEASIBLE_COST_CAP,
    )


def _weighted_panel_cost(costs: list[float], block_counts: list[int]) -> float:
    if not costs or len(costs) != len(block_counts):
        raise ValueError("weighted panel requires matching nonempty cost/block arrays")
    maximum = max(block_counts)
    weights = [math.exp((count - maximum) / 12.0) for count in block_counts]
    return sum(cost * weight for cost, weight in zip(costs, weights)) / sum(weights)


def _scenario_key(value: float) -> str:
    return f"{value:g}_seconds"


def _runtime_scenario_frontier(
    runs: list[dict[str, Any]],
    case_artifacts: list[dict[str, Any]],
    scenario_medians: tuple[float, ...] = RUNTIME_SCENARIO_MEDIANS,
) -> dict[str, Any]:
    """Build per-case median-runtime cost frontiers for unknown field medians."""

    if not case_artifacts:
        raise ValueError("runtime scenario frontier requires selected cases")
    case_ids = [row["test_id"] for row in case_artifacts]
    if case_ids != sorted(set(case_ids)):
        raise ValueError("scenario case artifacts must be sorted and unique")
    block_by_case = {row["test_id"]: row["block_count"] for row in case_artifacts}
    full_panel = case_ids == list(range(100)) and [
        block_by_case[index] for index in case_ids
    ] == list(range(21, 121))
    score_name = (
        "official_100_case_total_score"
        if full_panel
        else "partial_panel_exp_n_over_12_weighted_cost_not_official_total_score"
    )

    per_mode_case: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        mode_cases: dict[str, Any] = {}
        for case_id in case_ids:
            selected = [
                row for row in runs
                if row["mode"] == mode and row["case_id"] == case_id
            ]
            if not selected:
                raise ValueError(f"scenario frontier lacks {mode} case {case_id}")
            qualities = {json.dumps(row["quality"], sort_keys=True) for row in selected}
            if len(qualities) != 1:
                raise ValueError(f"scenario frontier found quality drift for {mode} case {case_id}")
            quality = selected[0]["quality"]
            if quality["block_count"] != block_by_case[case_id]:
                raise ValueError(f"scenario block count mismatch for case {case_id}")
            median_runtime = statistics.median(
                row["runtime_seconds"] for row in selected
            )
            quality_factor = 1.0 + QUALITY_ALPHA * (
                max(0.0, quality["hpwl_gap"]) + max(0.0, quality["area_gap"])
            )
            violation_factor = math.exp(
                VIOLATION_BETA * quality["violations_relative"]
            )
            mode_cases[str(case_id)] = {
                "block_count": quality["block_count"],
                "median_runtime_seconds": median_runtime,
                "quality_factor": quality_factor,
                "violation_factor": violation_factor,
                "cost_by_field_median": {
                    _scenario_key(median): _official_scenario_cost(
                        quality, median_runtime, median
                    )
                    for median in scenario_medians
                },
            }
        per_mode_case[mode] = mode_cases

    scenario_rows = []
    score_vectors = {mode: [] for mode in MODES}
    winner_sets = []
    block_counts = [block_by_case[case_id] for case_id in case_ids]
    for median in scenario_medians:
        key = _scenario_key(median)
        scores = {
            mode: _weighted_panel_cost(
                [per_mode_case[mode][str(case_id)]["cost_by_field_median"][key]
                 for case_id in case_ids],
                block_counts,
            )
            for mode in MODES
        }
        for mode, score in scores.items():
            score_vectors[mode].append(score)
        best = min(scores.values())
        winners = [mode for mode in MODES if math.isclose(scores[mode], best, rel_tol=0.0, abs_tol=1e-15)]
        winner_sets.append(set(winners))
        scenario_rows.append({
            "field_median_runtime_seconds": median,
            score_name: scores,
            "delta_vs_off": {mode: scores[mode] - scores["off"] for mode in MODES},
            "delta_vs_replacement": {
                mode: scores[mode] - scores["replacement"] for mode in MODES
            },
            "winners": winners,
        })

    nondominated = []
    for candidate in MODES:
        dominated = any(
            other != candidate
            and all(
                left <= right
                for left, right in zip(score_vectors[other], score_vectors[candidate])
            )
            and any(
                left < right
                for left, right in zip(score_vectors[other], score_vectors[candidate])
            )
            for other in MODES
        )
        if not dominated:
            nondominated.append(candidate)
    intersection = set(MODES)
    for winners in winner_sets:
        intersection &= winners
    return {
        "panel": {
            "kind": "official_complete_100_case_panel" if full_panel else "partial_report_only_panel",
            "case_count": len(case_ids),
            "case_ids": case_ids,
            "aggregate_name": score_name,
        },
        "formula": {
            "quality_alpha": QUALITY_ALPHA,
            "violation_beta": VIOLATION_BETA,
            "runtime_gamma": RUNTIME_GAMMA,
            "runtime_floor": 0.7,
            "feasible_cost_cap": FEASIBLE_COST_CAP,
            "case_weight": "exp(n/12), stabilized by subtracting max(n)",
            "runtime_source": "per-mode/per-case median of measured organizer-wrapper runs",
        },
        "per_mode_case": per_mode_case,
        "scenarios": scenario_rows,
        "nondominated_modes": nondominated,
        "all_scenario_winner_intersection": [
            mode for mode in MODES if mode in intersection
        ],
    }


def _verify_official_checkout(
    official_root: Path, official_sources: Path, timeout: int
) -> str:
    completed = _run_checked(
        [
            sys.executable,
            str(ROOT / "scripts/check_official_sources.py"),
            "--manifest",
            str(official_sources),
            "--floorset",
            str(official_root),
            "--require-floorset",
        ],
        cwd=ROOT,
        timeout=timeout,
    )
    if not completed.stdout.startswith("Official source integrity: PASS"):
        raise ValueError("official source checker did not report PASS")
    return _sha256_bytes(completed.stdout.encode("utf-8"))


def _selected_public_case_artifacts(
    data_root: Path, case_ids: list[int]
) -> list[dict[str, Any]]:
    """Bind the exact files addressed by the official 21..120 case ordering."""

    dataset = data_root / "LiteTensorDataTest"
    inventory = []
    for block_count in range(21, 121):
        config = dataset / f"config_{block_count}"
        inputs = sorted(config.glob("litedata_*.pth"))
        labels = sorted(config.glob("litelabel_*.pth"))
        if len(inputs) != 1 or len(labels) != 1:
            raise ValueError(
                "public dataset must contain exactly one input/label file for "
                f"config_{block_count}; observed {len(inputs)}/{len(labels)}"
            )
        input_id = inputs[0].stem.removeprefix("litedata_")
        label_id = labels[0].stem.removeprefix("litelabel_")
        if input_id != label_id:
            raise ValueError(f"public config_{block_count} input/label identifiers differ")
        inventory.append((block_count, inputs[0], labels[0]))
    artifacts = []
    for case_id in case_ids:
        block_count, input_path, label_path = inventory[case_id]
        artifacts.append({
            "test_id": case_id,
            "block_count": block_count,
            "input": {
                "path": input_path.relative_to(data_root).as_posix(),
                "sha256": _sha256_file(input_path),
                "size_bytes": input_path.stat().st_size,
            },
            "label": {
                "path": label_path.relative_to(data_root).as_posix(),
                "sha256": _sha256_file(label_path),
                "size_bytes": label_path.stat().st_size,
            },
        })
    return artifacts


def _read_optional_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _native_amd64_attestation(
    probe: dict[str, str] | None = None,
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Fail closed on architecture/emulation contradictions, not virtualization."""

    if probe is None:
        uname = os.uname()
        probe = {
            "platform_machine": platform.machine(),
            "platform_platform": platform.platform(),
            "uname_machine": uname.machine,
            "uname_sysname": uname.sysname,
            "uname_release": uname.release,
            "cpuinfo": _read_optional_text(Path("/proc/cpuinfo")),
        }
    environment = os.environ if environ is None else environ
    required_probe = {
        "platform_machine",
        "platform_platform",
        "uname_machine",
        "uname_sysname",
        "uname_release",
        "cpuinfo",
    }
    if set(probe) != required_probe or any(
        not isinstance(probe[name], str) for name in required_probe
    ):
        raise ValueError("native-host probe has an invalid schema")
    platform_machine = probe["platform_machine"].strip().lower()
    uname_machine = probe["uname_machine"].strip().lower()
    if platform_machine not in NATIVE_AMD64_NAMES or uname_machine not in NATIVE_AMD64_NAMES:
        raise ValueError(
            "timing requires agreeing AMD64 platform.machine and uname -m attestations"
        )
    contradiction_text = " ".join(
        (probe["platform_platform"], probe["uname_machine"], probe["cpuinfo"])
    ).lower()
    if re.search(r"(?:aarch64|arm64|armv[5-9]|riscv|ppc64|s390x)", contradiction_text):
        raise ValueError("native AMD64 timing probe contains a non-x86 architecture contradiction")
    if re.search(r"(?:qemu|tcg acceleration|bochs)", contradiction_text):
        raise ValueError("native AMD64 timing rejects QEMU/TCG/Bochs emulation evidence")
    emulation_environment = sorted(
        name for name in EMULATION_ENV_NAMES if environment.get(name)
    )
    if emulation_environment:
        raise ValueError(
            "native AMD64 timing rejects emulation environment markers: "
            + ", ".join(emulation_environment)
        )

    cpuinfo = probe["cpuinfo"]
    vendors = sorted(set(re.findall(r"^vendor_id\s*:\s*(\S+)", cpuinfo, re.MULTILINE)))
    recognized_vendors = {
        "GenuineIntel",
        "AuthenticAMD",
        "HygonGenuine",
        "CentaurHauls",
        "Shanghai",
    }
    if not vendors or not set(vendors).issubset(recognized_vendors):
        raise ValueError("native AMD64 timing lacks a recognized x86 CPU vendor attestation")
    flag_lines = re.findall(r"^(?:flags|Features)\s*:\s*(.*)$", cpuinfo, re.MULTILINE)
    flags = sorted({flag for line in flag_lines for flag in line.split()})
    if not {"lm", "sse2"}.issubset(flags):
        raise ValueError("native AMD64 timing CPU flags do not attest 64-bit x86 execution")
    models = sorted(set(re.findall(r"^model name\s*:\s*(.+)$", cpuinfo, re.MULTILINE)))
    if not models:
        raise ValueError("native AMD64 timing lacks CPU model evidence")
    # The hypervisor CPU flag is intentionally allowed: native GitHub-hosted
    # AMD64 runners are virtualized, but are not cross-architecture emulation.
    return {
        "status": "PASS",
        "platform_machine": probe["platform_machine"],
        "platform_platform": probe["platform_platform"],
        "uname": {
            "machine": probe["uname_machine"],
            "sysname": probe["uname_sysname"],
            "release": probe["uname_release"],
        },
        "cpu_vendors": vendors,
        "cpu_models": models,
        "cpu_flags": flags,
        "hypervisor_flag": "hypervisor" in flags,
        "cpuinfo_sha256": _sha256_bytes(cpuinfo.encode("utf-8")),
        "emulation_environment_markers": [],
    }


def _cpu_governors() -> dict[str, str]:
    result = {}
    for path in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor")):
        value = _read_optional_text(path).strip()
        if value:
            result[path.as_posix()] = value
    return result


def _host_measurement_snapshot() -> dict[str, Any]:
    affinity = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else list(range(os.cpu_count() or 0))
    )
    return {
        "native_amd64": _native_amd64_attestation(),
        "logical_cpu_count": os.cpu_count(),
        "process_affinity": affinity,
        "cpu_governors": _cpu_governors(),
        "load_average_1_5_15": list(os.getloadavg()),
        "monotonic_seconds": time.monotonic(),
    }


def _host_drift(pre: dict[str, Any], post: dict[str, Any]) -> dict[str, Any]:
    immutable_paths = (
        ("native_amd64.platform_machine", pre["native_amd64"]["platform_machine"], post["native_amd64"]["platform_machine"]),
        ("native_amd64.uname.machine", pre["native_amd64"]["uname"]["machine"], post["native_amd64"]["uname"]["machine"]),
        ("native_amd64.cpu_vendors", pre["native_amd64"]["cpu_vendors"], post["native_amd64"]["cpu_vendors"]),
        ("native_amd64.cpu_models", pre["native_amd64"]["cpu_models"], post["native_amd64"]["cpu_models"]),
        ("native_amd64.cpu_flags", pre["native_amd64"]["cpu_flags"], post["native_amd64"]["cpu_flags"]),
        ("logical_cpu_count", pre["logical_cpu_count"], post["logical_cpu_count"]),
        ("process_affinity", pre["process_affinity"], post["process_affinity"]),
        ("cpu_governors", pre["cpu_governors"], post["cpu_governors"]),
    )
    changed = [name for name, before, after in immutable_paths if before != after]
    if changed:
        raise ValueError("timing host identity/configuration drifted: " + ", ".join(changed))
    return {
        "status": "PASS",
        "immutable_fields_changed": [],
        "elapsed_monotonic_seconds": post["monotonic_seconds"] - pre["monotonic_seconds"],
        "load_average_delta": [
            after - before
            for before, after in zip(
                pre["load_average_1_5_15"], post["load_average_1_5_15"]
            )
        ],
    }


def _sanitized_timing_environment(
    source: dict[str, str] | None = None,
) -> dict[str, str]:
    original = os.environ if source is None else source
    allowed = (
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "TZ",
        "TMPDIR",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
    )
    environment = {
        name: original[name]
        for name in allowed
        if isinstance(original.get(name), str) and original[name]
    }
    environment.update({
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    })
    return environment


def _rotated_sequence_indices(case_index: int, cycle: int) -> tuple[int, ...]:
    count = len(williams_sequences())
    start = (case_index + cycle) % count
    return tuple((start + offset) % count for offset in range(count))


def _discarded_warmup_schedule(case_index: int) -> tuple[tuple[int, int, str], ...]:
    sequences = williams_sequences()
    sequence_index = case_index % len(sequences)
    return tuple(
        (sequence_index, period, mode)
        for period, mode in enumerate(sequences[sequence_index])
    )


def _official_evaluator_command(
    evaluator: Path,
    wrapper: Path,
    data_root: Path,
    case_id: int,
    result_path: Path,
) -> list[str]:
    return [
        sys.executable,
        str(evaluator),
        "--evaluate",
        str(wrapper),
        "--data-path",
        str(data_root),
        "--test-id",
        str(case_id),
        "--output",
        str(result_path),
    ]


def _artifact_bindings(
    *,
    build_manifest: Path,
    official_sources: Path,
    evaluator: Path,
    data_root: Path,
    case_ids: list[int],
    variants: dict[str, dict[str, Any]],
    package_roots: dict[str, Path],
) -> dict[str, Any]:
    manifest_dir = build_manifest.resolve().parent
    return {
        "build_manifest": _file_descriptor(build_manifest.name, build_manifest),
        "official_sources": _file_descriptor(official_sources.name, official_sources),
        "official_evaluator": _file_descriptor(evaluator.name, evaluator),
        "official_source_checker": _file_descriptor(
            "scripts/check_official_sources.py",
            ROOT / "scripts/check_official_sources.py",
        ),
        "selected_public_case_artifacts": _selected_public_case_artifacts(
            data_root, case_ids
        ),
        "packages": {
            mode: _file_descriptor(
                variants[mode]["package"]["path"],
                _safe_manifest_path(
                    manifest_dir,
                    variants[mode]["package"]["path"],
                    f"{mode} package",
                ),
            )
            for mode in MODES
        },
        "extracted_wrappers": {
            mode: _file_descriptor(
                INTERNAL_WRAPPER_PATH,
                package_roots[mode] / "op_wrapper.py",
            )
            for mode in MODES
        },
        "extracted_binaries": {
            mode: _file_descriptor(
                INTERNAL_BINARY_PATH,
                package_roots[mode] / "dist/my_optimizer/my_optimizer",
            )
            for mode in MODES
        },
    }


def _require_unchanged_bindings(
    before: dict[str, Any], after: dict[str, Any]
) -> None:
    if before != after:
        raise ValueError(
            "timing input/provenance artifacts changed between pre- and post-run rehash"
        )


def _audit_and_extract_variants(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    variants: dict[str, dict[str, Any]],
    official_sources: Path,
    destination: Path,
) -> dict[str, Path]:
    scripts = str(ROOT)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from scripts.audit_submission_package import (  # pylint: disable=import-outside-toplevel
        audit_archive,
        official_wrapper_sha,
    )

    tooling = manifest["tooling"]
    if tooling["package_audit_sha256"] != _sha256_file(
        ROOT / "scripts/audit_submission_package.py"
    ):
        raise ValueError("current package audit differs from the build-time audit")
    if tooling["solver_registry_sha256"] != _sha256_file(
        ROOT / "scripts/solver_components.py"
    ):
        raise ValueError("current solver registry differs from the build-time registry")
    if tooling["official_source_checker_sha256"] != _sha256_file(
        ROOT / "scripts/check_official_sources.py"
    ):
        raise ValueError("current official-source checker differs from build time")
    wrapper_sha256 = official_wrapper_sha(official_sources)
    if wrapper_sha256 != tooling["organizer_wrapper_sha256"]:
        raise ValueError("organizer wrapper binding differs from package build")

    extracted = {}
    manifest_dir = manifest_path.resolve().parent
    for mode in MODES:
        variant = variants[mode]
        for field, context in (
            (variant["build_log"], "build log"),
            (variant["audit"]["log"], "audit log"),
        ):
            log_path = _safe_manifest_path(
                manifest_dir, field["path"], f"{mode} {context}"
            )
            if (
                not log_path.is_file()
                or log_path.stat().st_size != field["size_bytes"]
                or _sha256_file(log_path) != field["sha256"]
            ):
                raise ValueError(f"{mode} {context} changed after package build")
        descriptor = variant["package"]
        package = _safe_manifest_path(
            manifest_dir, descriptor["path"], f"{mode} package"
        )
        if not package.is_file():
            raise ValueError(f"{mode} package is missing: {package}")
        if package.stat().st_size != descriptor["size_bytes"]:
            raise ValueError(f"{mode} package size changed")
        if _sha256_file(package) != descriptor["sha256"]:
            raise ValueError(f"{mode} package hash changed")
        messages = audit_archive(
            package,
            wrapper_sha256=wrapper_sha256,
            expected_archive_sha256=descriptor["sha256"],
            source_root=None,
            expected_source_sha256=variant["archived_source_sha256"],
            require_notices=True,
            smoke=True,
        )
        if "smoke=PASS" not in messages:
            raise ValueError(f"{mode} package re-audit did not run its self-test")
        if f"default_mode={mode}" not in messages:
            raise ValueError(f"{mode} package binary reports the wrong default mode")
        mode_destination = destination / mode
        _safe_extract_tar(package, mode_destination, expected_prefix=ARCHIVE_ROOT)
        package_root = mode_destination / ARCHIVE_ROOT
        wrapper = package_root / "op_wrapper.py"
        binary = package_root / "dist/my_optimizer/my_optimizer"
        optimizer = package_root / "source_fallback/my_optimizer.py"
        if (
            wrapper.stat().st_size != variant["wrapper"]["size_bytes"]
            or _sha256_file(wrapper) != variant["wrapper"]["sha256"]
            or _sha256_file(wrapper) != wrapper_sha256
        ):
            raise ValueError(f"{mode} extracted wrapper hash changed")
        if (
            binary.stat().st_size != variant["binary"]["size_bytes"]
            or _sha256_file(binary) != variant["binary"]["sha256"]
        ):
            raise ValueError(f"{mode} extracted binary hash changed")
        assignment = ASSIGNMENT_RE.findall(optimizer.read_text(encoding="utf-8"))
        expected = [f'        self._learned_order_mode = "{mode}"']
        if assignment != expected:
            raise ValueError(f"{mode} package contains the wrong production default")
        extracted[mode] = package_root
    return extracted


def time_mode_packages(
    *,
    build_manifest: Path,
    official_root: Path,
    data_root: Path,
    official_sources: Path,
    case_ids: list[int],
    cycles: int,
    output_dir: Path,
    timeout: int = 180,
) -> dict[str, Any]:
    """Run organizer-wrapper package timing in complete Williams blocks."""

    if not case_ids:
        raise ValueError("at least one explicit --case-id is required")
    if any(isinstance(case, bool) or not isinstance(case, int) or not 0 <= case < 100 for case in case_ids):
        raise ValueError("case IDs must be integers in 0..99")
    if case_ids != sorted(set(case_ids)):
        raise ValueError("case IDs must be sorted and unique")
    _require_int(cycles, "timing cycles", minimum=1)
    _native_amd64_attestation()
    output_dir = _ensure_outside_repository(output_dir, ROOT, "timing output directory")
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    official_root = official_root.resolve()
    data_root = data_root.resolve()
    official_sources = official_sources.resolve()
    manifest_raw, manifest, variants = _validate_build_manifest(build_manifest)
    if _sha256_file(official_sources) != manifest["tooling"]["official_sources_sha256"]:
        raise ValueError("official-source manifest differs from package build")
    official_check_sha256 = _verify_official_checkout(
        official_root, official_sources, timeout
    )
    official_payload = json.loads(official_sources.read_text(encoding="utf-8"))
    evaluator = official_root / "iccad2026contest/iccad2026_evaluate.py"
    expected_evaluator_sha = official_payload["floorset"]["files"][
        "iccad2026contest/iccad2026_evaluate.py"
    ]
    if not evaluator.is_file() or _sha256_file(evaluator) != expected_evaluator_sha:
        raise ValueError("official evaluator bytes do not match the pinned manifest")
    selected_case_artifacts = _selected_public_case_artifacts(data_root, case_ids)
    block_by_case = {
        row["test_id"]: row["block_count"] for row in selected_case_artifacts
    }

    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        packages_dir = staging / ".packages"
        packages_dir.mkdir()
        package_roots = _audit_and_extract_variants(
            manifest_path=build_manifest,
            manifest=manifest,
            variants=variants,
            official_sources=official_sources,
            destination=packages_dir,
        )
        pre_bindings = _artifact_bindings(
            build_manifest=build_manifest,
            official_sources=official_sources,
            evaluator=evaluator,
            data_root=data_root,
            case_ids=case_ids,
            variants=variants,
            package_roots=package_roots,
        )
        binding_sha256 = _sha256_bytes(json.dumps(
            pre_bindings, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8"))
        host_pre = _host_measurement_snapshot()
        sequences = williams_sequences()
        runs: list[dict[str, Any]] = []
        warmups: list[dict[str, Any]] = []
        quality_by_mode_case: dict[tuple[str, int], dict[str, Any]] = {}
        environment = _sanitized_timing_environment()
        ordinal = 0

        def execute(
            *,
            case_id: int,
            case_index: int,
            cycle: int,
            sequence_index: int,
            schedule_position: int,
            period: int,
            mode: str,
            warmup: bool,
        ) -> dict[str, Any]:
            nonlocal ordinal
            ordinal += 1
            prefix = "warmup" if warmup else "run"
            stem = (
                f"{prefix}-{ordinal:05d}-case-{case_id:02d}-cycle-{cycle:02d}-"
                f"schedule-{schedule_position}-sequence-{sequence_index}-"
                f"period-{period}-{mode}"
            )
            directory = "warmups" if warmup else "runs"
            result_path = staging / directory / f"{stem}.json"
            log_path = staging / "logs" / f"{stem}.log"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            wrapper = package_roots[mode] / "op_wrapper.py"
            completed = _run_checked(
                _official_evaluator_command(
                    evaluator, wrapper, data_root, case_id, result_path
                ),
                cwd=evaluator.parent,
                timeout=timeout,
                env=environment,
            )
            _write_log(log_path, completed)
            validated = _validate_official_result(
                result_path, case_id, block_by_case[case_id]
            )
            key = (mode, case_id)
            previous_quality = quality_by_mode_case.setdefault(
                key, validated["quality"]
            )
            if validated["quality"] != previous_quality:
                raise ValueError(
                    f"{mode} case {case_id} output changed across warmup/timing repeats"
                )
            return {
                "ordinal": ordinal,
                "discarded_warmup": warmup,
                "case_id": case_id,
                "case_index": case_index,
                "cycle": cycle,
                "schedule_position": schedule_position,
                "sequence": sequence_index,
                "period": period,
                "mode": mode,
                "runtime_seconds": validated["runtime_seconds"],
                "quality": validated["quality"],
                "result": _file_descriptor(
                    f"{directory}/{result_path.name}", result_path
                ),
                "log": _file_descriptor(f"logs/{log_path.name}", log_path),
            }

        for case_index, case_id in enumerate(case_ids):
            # One complete, balanced treatment pass warms the evaluator,
            # wrapper and binary paths. These four runs are retained but never
            # enter medians or score scenarios.
            for warmup_sequence_index, period, mode in _discarded_warmup_schedule(
                case_index
            ):
                warmups.append(execute(
                    case_id=case_id,
                    case_index=case_index,
                    cycle=-1,
                    sequence_index=warmup_sequence_index,
                    schedule_position=period,
                    period=period,
                    mode=mode,
                    warmup=True,
                ))
            for cycle in range(cycles):
                for schedule_position, sequence_index in enumerate(
                    _rotated_sequence_indices(case_index, cycle)
                ):
                    sequence = sequences[sequence_index]
                    for period, mode in enumerate(sequence):
                        runs.append(execute(
                            case_id=case_id,
                            case_index=case_index,
                            cycle=cycle,
                            sequence_index=sequence_index,
                            schedule_position=schedule_position,
                            period=period,
                            mode=mode,
                            warmup=False,
                        ))
        post_bindings = _artifact_bindings(
            build_manifest=build_manifest,
            official_sources=official_sources,
            evaluator=evaluator,
            data_root=data_root,
            case_ids=case_ids,
            variants=variants,
            package_roots=package_roots,
        )
        _require_unchanged_bindings(pre_bindings, post_bindings)
        post_binding_sha256 = _sha256_bytes(json.dumps(
            post_bindings, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8"))
        post_official_check_sha256 = _verify_official_checkout(
            official_root, official_sources, timeout
        )
        if post_official_check_sha256 != official_check_sha256:
            raise ValueError("official-source checker output changed during timing")
        host_post = _host_measurement_snapshot()
        drift = _host_drift(host_pre, host_post)
        shutil.rmtree(packages_dir)
        result = {
            "schema_version": TIMING_SCHEMA_VERSION,
            "mode": TIMING_MODE,
            "build_manifest": {
                "path": build_manifest.resolve().name,
                "sha256": _sha256_bytes(manifest_raw),
                "source_commit": manifest["source"]["commit"],
            },
            "contract": {
                "interface": "organizer op_wrapper.py spawning packaged executable once per solve",
                "modes": list(MODES),
                "williams_sequences": [list(row) for row in sequences],
                "complete_sequences_per_case_cycle": len(sequences),
                "no_my_opt_bin_override": True,
                "discarded_balanced_warmup_per_case": list(MODES),
                "sequence_row_start": "rotated by (case_index + cycle) modulo 4",
                "environment": "allowlisted and thread-count pinned",
            },
            "official": {
                "floorset_commit": official_payload["floorset"]["commit"],
                "evaluator_sha256": expected_evaluator_sha,
                "organizer_wrapper_sha256": manifest["tooling"][
                    "organizer_wrapper_sha256"
                ],
                "official_source_check_stdout_sha256": official_check_sha256,
                "selected_public_case_artifacts": selected_case_artifacts,
            },
            "environment": {
                "native_amd64_attestation": "PASS",
                "python": platform.python_version(),
                "python_executable": str(Path(sys.executable).resolve()),
                "python_executable_sha256": _sha256_file(Path(sys.executable).resolve()),
                "sanitized_environment_keys": sorted(environment),
                "fixed_environment": {
                    name: environment[name]
                    for name in (
                        "PYTHONHASHSEED", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                    )
                },
                "pre": host_pre,
                "post": host_post,
                "drift": drift,
            },
            "schedule": {
                "case_ids": case_ids,
                "cycles": cycles,
                "discarded_warmup_runs": len(warmups),
                "total_runs": len(runs),
                "total_process_executions": len(warmups) + len(runs),
            },
            "artifact_bindings": pre_bindings,
            "post_timing_rehash": {
                "status": "PASS",
                "pre_sha256": binding_sha256,
                "post_sha256": post_binding_sha256,
            },
            "warmups": warmups,
            "runs": runs,
            "summary": _timing_summary(runs),
            "runtime_scenario_frontier": _runtime_scenario_frontier(
                runs, selected_case_artifacts
            ),
        }
        output_path = staging / "timing_manifest.json"
        output_path.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _publish_directory(staging, output_dir)
        return result
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build all four isolated packages")
    build.add_argument("--repository", type=Path, default=ROOT)
    build.add_argument("--commit", required=True, help="one explicit commit or tag")
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--workspace-root", type=Path)
    build.add_argument("--timeout", type=int, default=3600, help="seconds per build")

    timing = subparsers.add_parser("time", help="run balanced organizer-wrapper timing")
    timing.add_argument("--build-manifest", type=Path, required=True)
    timing.add_argument("--official-root", type=Path, default=ROOT / "external/FloorSet")
    timing.add_argument("--data-root", type=Path, default=ROOT / "external/FloorSet")
    timing.add_argument("--official-sources", type=Path, default=ROOT / "docs/official_sources.json")
    timing.add_argument("--case-id", type=int, action="append", required=True)
    timing.add_argument("--cycles", type=int, default=1)
    timing.add_argument("--output-dir", type=Path, required=True)
    timing.add_argument("--timeout", type=int, default=180, help="seconds per evaluator run")
    args = parser.parse_args()

    try:
        if args.command == "build":
            result = build_mode_packages(
                repository=args.repository,
                reference=args.commit,
                output_dir=args.output_dir,
                workspace_root=args.workspace_root,
                timeout=args.timeout,
            )
            summary = {
                "source_commit": result["source"]["commit"],
                "packages": {
                    row["mode"]: row["package"]["sha256"]
                    for row in result["variants"]
                },
            }
        else:
            result = time_mode_packages(
                build_manifest=args.build_manifest,
                official_root=args.official_root,
                data_root=args.data_root,
                official_sources=args.official_sources,
                case_ids=args.case_id,
                cycles=args.cycles,
                output_dir=args.output_dir,
                timeout=args.timeout,
            )
            summary = {
                "source_commit": result["build_manifest"]["source_commit"],
                "total_runs": result["schedule"]["total_runs"],
                "mean_seconds": {
                    mode: result["summary"][mode]["mean_seconds"] for mode in MODES
                },
            }
    except Exception as error:
        print(f"Package-mode tournament: FAIL\n  {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
