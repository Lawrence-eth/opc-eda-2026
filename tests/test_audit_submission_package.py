import hashlib
import io
import struct
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


def _archive(tmp_path: Path, *, machine=62, glibc=b"GLIBC_2.38", extra=None) -> Path:
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
    return path


def _audit(path: Path, **kwargs):
    return package_audit.audit_archive(
        path,
        wrapper_sha256=hashlib.sha256(WRAPPER).hexdigest(),
        source_root=None,
        **kwargs,
    )


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
