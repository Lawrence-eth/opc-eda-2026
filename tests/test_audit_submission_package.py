import hashlib
import io
import json
import struct
import subprocess
import tarfile
from pathlib import Path

import pytest

from scripts import audit_submission_package as package_audit


WRAPPER = b"official wrapper\n"


def _elf(machine=62, glibc=b"GLIBC_2.38") -> bytes:
    header = bytearray(64)
    header[:6] = b"\x7fELF\x02\x01"
    header[18:20] = struct.pack("<H", machine)
    return bytes(header) + glibc


def _add_file(archive, name, data, mode=0o644):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = mode
    archive.addfile(info, io.BytesIO(data))


def _archive(
    tmp_path: Path,
    *,
    machine=62,
    glibc=b"GLIBC_2.38",
    extra=None,
    files=None,
) -> Path:
    path = tmp_path / "submission.tar.gz"
    root = package_audit.EXPECTED_ROOT
    with tarfile.open(path, "w:gz") as archive:
        _add_file(archive, f"{root}/op_wrapper.py", WRAPPER)
        _add_file(archive, f"{root}/README.md", b"readme\n")
        _add_file(archive, f"{root}/requirements.txt", b"# none\n")
        _add_file(
            archive,
            f"{root}/dist/my_optimizer/my_optimizer",
            _elf(machine, glibc),
            mode=0o755,
        )
        if extra is not None:
            name, data = extra
            _add_file(archive, name, data)
        for name, data in (files or {}).items():
            _add_file(archive, name, data)
    return path


def _audit(path: Path, **kwargs):
    kwargs.setdefault("source_root", None)
    return package_audit.audit_archive(
        path,
        wrapper_sha256=hashlib.sha256(WRAPPER).hexdigest(),
        **kwargs,
    )


def _source_bound_fixture(tmp_path: Path):
    source_root = tmp_path / "source-root"
    archive_files = {}
    for local_name, packaged_name in package_audit.SOURCE_BINDINGS.items():
        data = f"fixture for {local_name}\n".encode()
        local_path = source_root / local_name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(data)
        archive_files[
            f"{package_audit.EXPECTED_ROOT}/{packaged_name}"
        ] = data
    return source_root, archive_files


def _release_manifest(tmp_path: Path, *, extra_sources=None) -> Path:
    sources = {
        local_name: hashlib.sha256(local_name.encode()).hexdigest()
        for local_name in package_audit.SOURCE_BINDINGS
    }
    sources.update(extra_sources or {})
    path = tmp_path / "release_manifest.json"
    path.write_text(json.dumps({"solver": {"sources": sources}}), encoding="utf-8")
    return path


def _live_self_test_payload():
    digest = "a" * 64
    mib_positions = [[float(index), 0.0, 1.0, 1.0] for index in range(4)]
    return {
        "schema_version": 1,
        "learned": {
            "model_payload_sha256": package_audit.EXPECTED_LEARNED_MODEL_PAYLOAD_SHA256,
            "compiled_model_sha256": digest,
            "prior_sha256": digest,
            "raw_candidate_sha256": digest,
            "candidate_attempted": True,
            "candidate_selected": False,
            "production_eligible_block_count": 100,
            "abstention_block_count": 101,
            "abstention_verified": True,
            "final_positions": [
                [float(index), 0.0, 1.0, 1.0] for index in range(100)
            ],
        },
        "safe_mib": {
            "repaired": True,
            "positions": mib_positions,
            "positions_sha256": hashlib.sha256(
                json.dumps(
                    mib_positions, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
        },
    }


def test_minimal_package_passes_structural_audit(tmp_path):
    messages = _audit(_archive(tmp_path))

    assert "elf_machine=AMD64" in messages
    assert "max_glibc=2.38" in messages


def test_path_traversal_is_rejected(tmp_path):
    path = _archive(tmp_path, extra=("../escape", b"bad"))

    with pytest.raises(package_audit.PackageAuditError, match="unsafe archive member|outside"):
        _audit(path)


def test_non_amd64_executable_is_rejected(tmp_path):
    path = _archive(tmp_path, machine=183)

    with pytest.raises(package_audit.PackageAuditError, match="expected AMD64"):
        _audit(path)


def test_too_new_glibc_is_rejected(tmp_path):
    path = _archive(tmp_path, glibc=b"GLIBC_2.42")

    with pytest.raises(package_audit.PackageAuditError, match="newer than target"):
        _audit(path)


def test_required_notices_are_fail_closed(tmp_path):
    path = _archive(tmp_path)

    with pytest.raises(package_audit.PackageAuditError, match="lacks required notice"):
        _audit(path, require_notices=True)


def test_source_binding_requires_every_registry_component(tmp_path):
    source_root, files = _source_bound_fixture(tmp_path)
    missing = (
        f"{package_audit.EXPECTED_ROOT}/"
        f"{package_audit.LIVE_SOURCE_BINDINGS['contest_solution/golden_plus_repair.py']}"
    )
    del files[missing]

    with pytest.raises(package_audit.PackageAuditError, match="lacks source binding"):
        _audit(_archive(tmp_path, files=files), source_root=source_root)


def test_source_binding_rejects_drifted_registry_component(tmp_path):
    source_root, files = _source_bound_fixture(tmp_path)
    packaged = package_audit.LIVE_SOURCE_BINDINGS[
        "contest_solution/golden_plus_repair.py"
    ]
    files[f"{package_audit.EXPECTED_ROOT}/{packaged}"] = b"drifted\n"

    with pytest.raises(
        package_audit.PackageAuditError,
        match="packaged source differs from contest_solution/golden_plus_repair.py",
    ):
        _audit(_archive(tmp_path, files=files), source_root=source_root)


def test_release_source_hashes_require_complete_live_registry(tmp_path):
    manifest = _release_manifest(tmp_path)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    del data["solver"]["sources"]["contest_solution/golden_plus_repair.py"]
    manifest.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(
        package_audit.PackageAuditError,
        match="lacks archived sources.*golden_plus_repair.py",
    ):
        package_audit.release_source_hashes(manifest)


def test_release_source_hashes_require_package_support_sources(tmp_path):
    manifest = _release_manifest(tmp_path)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    del data["solver"]["sources"]["packaging/solver_main.py"]
    manifest.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(
        package_audit.PackageAuditError,
        match="lacks archived sources.*packaging/solver_main.py",
    ):
        package_audit.release_source_hashes(manifest)


def test_release_source_hashes_reject_basename_collisions(tmp_path):
    manifest = _release_manifest(
        tmp_path,
        extra_sources={
            "other/my_optimizer.py": hashlib.sha256(b"other").hexdigest()
        },
    )

    with pytest.raises(package_audit.PackageAuditError, match="basenames collide"):
        package_audit.release_source_hashes(manifest)


def test_release_source_hashes_retain_package_support_sources(tmp_path):
    hashes = package_audit.release_source_hashes(_release_manifest(tmp_path))

    assert set(package_audit.SUPPORT_SOURCE_BINDINGS.values()).issubset(hashes)


def test_binary_smoke_requires_positions_object(tmp_path, monkeypatch):
    monkeypatch.setattr(package_audit.platform, "machine", lambda: "x86_64")

    def completed(stdout):
        return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")

    responses = iter(
        (
            '{"positions": [[0, 0, 10, 10], [10, 0, 5, 5], [15, 0, 2, 2]]}',
            json.dumps(_live_self_test_payload()),
            json.dumps(_live_self_test_payload()),
        )
    )
    monkeypatch.setattr(
        package_audit.subprocess,
        "run",
        lambda *args, **kwargs: completed(next(responses)),
    )
    package_audit._smoke(tmp_path)

    monkeypatch.setattr(
        package_audit.subprocess,
        "run",
        lambda *args, **kwargs: completed("[[0, 0, 10, 10], [10, 0, 5, 5], [15, 0, 2, 2]]"),
    )
    with pytest.raises(package_audit.PackageAuditError, match="object containing positions"):
        package_audit._smoke(tmp_path)


def test_binary_smoke_rejects_live_module_source_binary_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(package_audit.platform, "machine", lambda: "x86_64")

    source_payload = _live_self_test_payload()
    binary_payload = _live_self_test_payload()
    binary_payload["learned"]["prior_sha256"] = "b" * 64
    responses = iter(
        (
            '{"positions": [[0, 0, 10, 10], [10, 0, 5, 5], [15, 0, 2, 2]]}',
            json.dumps(source_payload),
            json.dumps(binary_payload),
        )
    )
    monkeypatch.setattr(
        package_audit.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            [], 0, stdout=next(responses), stderr=""
        ),
    )

    with pytest.raises(package_audit.PackageAuditError, match="payloads differ"):
        package_audit._smoke(tmp_path)


def test_binary_smoke_compares_live_payload_bytes_exactly(tmp_path, monkeypatch):
    monkeypatch.setattr(package_audit.platform, "machine", lambda: "x86_64")

    source_payload = _live_self_test_payload()
    binary_payload = _live_self_test_payload()
    binary_payload["learned"]["final_positions"][0][0] = -0.0
    responses = iter(
        (
            '{"positions": [[0, 0, 10, 10], [10, 0, 5, 5], [15, 0, 2, 2]]}',
            json.dumps(source_payload),
            json.dumps(binary_payload),
        )
    )
    monkeypatch.setattr(
        package_audit.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            [], 0, stdout=next(responses), stderr=""
        ),
    )

    with pytest.raises(package_audit.PackageAuditError, match="payloads differ"):
        package_audit._smoke(tmp_path)
