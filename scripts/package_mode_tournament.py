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
ARCHIVE_ROOT = "iccad2026_submission"
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


def _audit_details(stdout: str, expected_package_sha256: str) -> dict[str, str]:
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
    return details


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
    components = {
        f"contest_solution/{name}": _sha256_file(
            variant_root / "contest_solution" / name
        )
        for name in component_names
    }

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
    details = _audit_details(audit.stdout, archive_sha256)
    audit_log = output_staging / "logs" / f"{mode}.audit.log"
    audit_log_sha256 = _write_log(audit_log, audit)

    archived_sources = {
        f"source_fallback/{path.name}": _sha256_file(path)
        for path in sorted(fallback.glob("*.py"))
        if path.is_file()
    }
    if not archived_sources or "source_fallback/my_optimizer.py" not in archived_sources:
        raise ValueError(f"{mode} package has no bound source fallback")
    if archived_sources["source_fallback/my_optimizer.py"] != patch["sha256_after"]:
        raise ValueError(f"{mode} archived optimizer differs from the generated source")

    package_relative = Path("packages") / f"iccad2026_submission_{mode}.tar.gz"
    package_output = output_staging / package_relative
    package_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(archive, package_output)
    if _sha256_file(package_output) != archive_sha256:
        raise IOError(f"{mode} package changed while publishing")

    return {
        "mode": mode,
        "source_patch": patch,
        "solver_components": components,
        "archived_source_sha256": archived_sources,
        "binary": {
            "sha256": _sha256_file(binary),
            "size_bytes": binary.stat().st_size,
        },
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
            },
        },
        "build_log": {
            "path": f"logs/{mode}.build.log",
            "sha256": build_log_sha256,
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
                "schema_version": 1,
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
    if manifest["schema_version"] != 1 or manifest["mode"] != BUILD_MODE:
        raise ValueError("not a schema-1 four-mode package build manifest")
    source = manifest["source"]
    if not isinstance(source, dict):
        raise ValueError("build manifest source must be an object")
    _require_commit(source.get("commit"), "build source commit")
    _require_commit(source.get("tree"), "build source tree")
    for field in ("git_archive_sha256", "base_optimizer_sha256"):
        _require_sha256(source.get(field), f"build source {field}")
    contract = manifest["contract"]
    if not isinstance(contract, dict) or contract.get("modes") != list(MODES):
        raise ValueError("build manifest does not contain the complete four-mode contract")
    if contract.get("timing_design") != [list(row) for row in williams_sequences()]:
        raise ValueError("build manifest Williams design changed")
    tooling = manifest["tooling"]
    required_tooling = {
        "orchestrator_sha256",
        "build_submission_sha256",
        "package_audit_sha256",
        "package_self_test_sha256",
        "official_sources_sha256",
        "organizer_wrapper_sha256",
    }
    _require_exact_keys(tooling, required_tooling, "build tooling")
    for field in required_tooling:
        _require_sha256(tooling[field], f"build tooling {field}")
    if tooling["orchestrator_sha256"] != _sha256_file(Path(__file__).resolve()):
        raise ValueError("build manifest was produced by different orchestrator bytes")
    variants = manifest["variants"]
    if not isinstance(variants, list) or len(variants) != len(MODES):
        raise ValueError("build manifest must contain exactly four variants")
    by_mode = {}
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise ValueError(f"variant {index} is not an object")
        mode = variant.get("mode")
        if mode not in MODES or mode in by_mode:
            raise ValueError(f"variant {index} has unknown or duplicate mode")
        package = variant.get("package")
        if not isinstance(package, dict):
            raise ValueError(f"variant {mode} has no package descriptor")
        _require_sha256(package.get("sha256"), f"variant {mode} package")
        _require_int(package.get("size_bytes"), f"variant {mode} package size", minimum=1)
        sources = variant.get("archived_source_sha256")
        if not isinstance(sources, dict) or "source_fallback/my_optimizer.py" not in sources:
            raise ValueError(f"variant {mode} has incomplete archived source hashes")
        for name, digest in sources.items():
            _safe_manifest_path(Path("/manifest"), name, f"variant {mode} source")
            _require_sha256(digest, f"variant {mode} source {name}")
        audit = variant.get("audit")
        if not isinstance(audit, dict) or audit.get("status") != "PASS":
            raise ValueError(f"variant {mode} did not pass its build-time audit")
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


def _validate_official_result(path: Path, case_id: int) -> dict[str, Any]:
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
    wrapper_sha256 = official_wrapper_sha(official_sources)
    if wrapper_sha256 != tooling["organizer_wrapper_sha256"]:
        raise ValueError("organizer wrapper binding differs from package build")

    extracted = {}
    manifest_dir = manifest_path.resolve().parent
    for mode in MODES:
        variant = variants[mode]
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
        mode_destination = destination / mode
        _safe_extract_tar(package, mode_destination, expected_prefix=ARCHIVE_ROOT)
        package_root = mode_destination / ARCHIVE_ROOT
        wrapper = package_root / "op_wrapper.py"
        optimizer = package_root / "source_fallback/my_optimizer.py"
        if _sha256_file(wrapper) != wrapper_sha256:
            raise ValueError(f"{mode} extracted wrapper hash changed")
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
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        raise ValueError("organizer-wrapper timing requires a native AMD64 host")
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
        sequences = williams_sequences()
        runs = []
        quality_by_mode_case: dict[tuple[str, int], dict[str, Any]] = {}
        environment = dict(os.environ)
        environment.pop("MY_OPT_BIN", None)
        environment["PYTHONHASHSEED"] = "0"
        ordinal = 0
        for case_id in case_ids:
            for cycle in range(cycles):
                for sequence_index, sequence in enumerate(sequences):
                    for period, mode in enumerate(sequence):
                        ordinal += 1
                        stem = (
                            f"run-{ordinal:05d}-case-{case_id:02d}-cycle-{cycle:02d}-"
                            f"sequence-{sequence_index}-period-{period}-{mode}"
                        )
                        result_path = staging / "runs" / f"{stem}.json"
                        log_path = staging / "logs" / f"{stem}.log"
                        result_path.parent.mkdir(parents=True, exist_ok=True)
                        wrapper = package_roots[mode] / "op_wrapper.py"
                        completed = _run_checked(
                            [
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
                            ],
                            cwd=evaluator.parent,
                            timeout=timeout,
                            env=environment,
                        )
                        log_sha256 = _write_log(log_path, completed)
                        validated = _validate_official_result(result_path, case_id)
                        key = (mode, case_id)
                        previous_quality = quality_by_mode_case.setdefault(
                            key, validated["quality"]
                        )
                        if validated["quality"] != previous_quality:
                            raise ValueError(
                                f"{mode} case {case_id} output changed across timing repeats"
                            )
                        runs.append({
                            "ordinal": ordinal,
                            "case_id": case_id,
                            "cycle": cycle,
                            "sequence": sequence_index,
                            "period": period,
                            "mode": mode,
                            "runtime_seconds": validated["runtime_seconds"],
                            "quality": validated["quality"],
                            "result": {
                                "path": f"runs/{result_path.name}",
                                "sha256": _sha256_file(result_path),
                            },
                            "log": {
                                "path": f"logs/{log_path.name}",
                                "sha256": log_sha256,
                            },
                        })
        shutil.rmtree(packages_dir)
        result = {
            "schema_version": 1,
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
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "python_executable": str(Path(sys.executable).resolve()),
                "python_executable_sha256": _sha256_file(Path(sys.executable).resolve()),
                "pythonhashseed": "0",
            },
            "schedule": {
                "case_ids": case_ids,
                "cycles": cycles,
                "total_runs": len(runs),
            },
            "runs": runs,
            "summary": _timing_summary(runs),
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
