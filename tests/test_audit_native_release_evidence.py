import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import stat
from typing import Callable
import zipfile

import pytest

from scripts import audit_native_release_evidence as AUDIT
from scripts import package_mode_tournament as TOURNAMENT
from scripts.solver_components import (
    LIVE_SOLVER_COMPONENTS,
    PACKAGE_SUPPORT_SOURCE_BINDINGS,
)


HEAD_SHA = "1" * 40
FLOORSET_SHA = "2" * 40
RUN_ID = 29394600065
RUN_ATTEMPT = 1
ARTIFACT_NAME = f"native-package-tournament-{HEAD_SHA}-{RUN_ATTEMPT}"
ARTIFACT_ZIP_NAME = f"{ARTIFACT_NAME}.zip"
SELECTED_MODE = "replacement"


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=True) + "\n").encode("utf-8")


def _descriptor(path: str, raw: bytes) -> dict:
    return {"path": path, "sha256": _sha256(raw), "size_bytes": len(raw)}


def _positions(case_id: int) -> list[list[float]]:
    return [
        [float(index), float(case_id), 0.5, 0.5]
        for index in range(case_id + 21)
    ]


def _source_row(case_id: int) -> dict:
    return {
        "test_id": case_id,
        "block_count": case_id + 21,
        "is_feasible": True,
        "error": None,
        "runtime_seconds": 0.1,
        "hpwl_gap": 0.0,
        "area_gap": 0.0,
        "violations_relative": 0.0,
        "cost": 1.0,
        "positions": _positions(case_id),
    }


def _quality(case_id: int) -> dict:
    source = _source_row(case_id)
    return {
        "test_id": case_id,
        "block_count": source["block_count"],
        "is_feasible": True,
        "hpwl_gap": source["hpwl_gap"],
        "area_gap": source["area_gap"],
        "violations_relative": source["violations_relative"],
        "cost": source["cost"],
        "positions_sha256": AUDIT._canonical_positions_sha256(
            source["positions"], source["block_count"]
        ),
    }


def _write_sources(root: Path) -> tuple[dict[str, str], str]:
    source_hashes: dict[str, str] = {}
    optimizer_text = (
        "class MyOptimizer:\n"
        "    def __init__(self):\n"
        f"{TOURNAMENT.DEFAULT_ASSIGNMENT}\n"
    )
    for component in LIVE_SOLVER_COMPONENTS:
        relative = f"contest_solution/{component}"
        raw = (
            optimizer_text.encode("utf-8")
            if component == "my_optimizer.py"
            else f"# fixture {component}\n".encode("utf-8")
        )
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        source_hashes[relative] = _sha256(raw)
    for relative in PACKAGE_SUPPORT_SOURCE_BINDINGS:
        raw = f"# fixture {relative}\n".encode("utf-8")
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        source_hashes[relative] = _sha256(raw)
    return source_hashes, optimizer_text


def _build_manifest(
    source_hashes: dict[str, str], optimizer_text: str, members: dict[str, bytes]
) -> tuple[dict, dict[str, dict]]:
    base_optimizer_sha = source_hashes["contest_solution/my_optimizer.py"]
    package_descriptors: dict[str, dict] = {}
    variants = []
    for mode in TOURNAMENT.MODES:
        package_path = f"packages/iccad2026_submission_{mode}.tar.gz"
        package_raw = f"package-{mode}\n".encode("utf-8")
        members[f"opc-four-mode-build/{package_path}"] = package_raw
        package = _descriptor(package_path, package_raw)
        package_descriptors[mode] = package

        assignment = f'        self._learned_order_mode = "{mode}"'
        patched = optimizer_text.replace(TOURNAMENT.DEFAULT_ASSIGNMENT, assignment, 1)
        patched_sha = _sha256(patched.encode("utf-8"))
        solver_sources = {
            name: digest
            for name, digest in source_hashes.items()
            if name.startswith("contest_solution/")
        }
        solver_sources["contest_solution/my_optimizer.py"] = patched_sha
        support_sources = {
            name: source_hashes[name] for name in PACKAGE_SUPPORT_SOURCE_BINDINGS
        }
        archived_sources = {
            **{
                f"source_fallback/{Path(name).name}": digest
                for name, digest in solver_sources.items()
            },
            **{
                packaged_name: support_sources[source_name]
                for source_name, packaged_name in PACKAGE_SUPPORT_SOURCE_BINDINGS.items()
            },
        }
        build_log_path = f"logs/{mode}.build.log"
        audit_log_path = f"logs/{mode}.audit.log"
        build_log = f"built {mode}\n".encode("utf-8")
        audit_log = f"audited {mode}\n".encode("utf-8")
        members[f"opc-four-mode-build/{build_log_path}"] = build_log
        members[f"opc-four-mode-build/{audit_log_path}"] = audit_log
        variants.append(
            {
                "mode": mode,
                "source_patch": {
                    "path": "contest_solution/my_optimizer.py",
                    "assignment_before": 'self._learned_order_mode = "replacement"',
                    "assignment_after": f'self._learned_order_mode = "{mode}"',
                    "sha256_before": base_optimizer_sha,
                    "sha256_after": patched_sha,
                    "changed": mode != SELECTED_MODE,
                },
                "solver_components": solver_sources,
                "package_support_sources": support_sources,
                "archived_source_sha256": archived_sources,
                "binary": {
                    "path": TOURNAMENT.INTERNAL_BINARY_PATH,
                    "sha256": "a" * 64,
                    "size_bytes": 100,
                },
                "wrapper": {
                    "path": TOURNAMENT.INTERNAL_WRAPPER_PATH,
                    "sha256": "b" * 64,
                    "size_bytes": 100,
                },
                "package": package,
                "audit": {
                    "status": "PASS",
                    "details": {
                        "archive_sha256": package["sha256"],
                        "members": "fixture",
                        "elf_machine": "AMD64",
                        "max_glibc": "2.38",
                        "smoke": "PASS",
                        "default_mode": mode,
                    },
                    "log": _descriptor(audit_log_path, audit_log),
                },
                "build_log": _descriptor(build_log_path, build_log),
            }
        )
    build = {
        "schema_version": 2,
        "mode": TOURNAMENT.BUILD_MODE,
        "source": {
            "commit": HEAD_SHA,
            "tree": "3" * 40,
            "git_archive_sha256": "4" * 64,
            "base_optimizer_sha256": base_optimizer_sha,
        },
        "contract": {
            "modes": list(TOURNAMENT.MODES),
            "source_mutation": "isolated fixture patch",
            "build_environment": "pinned AMD64 fixture",
            "audit": "fixture package audit",
            "timing_design": [
                list(sequence) for sequence in TOURNAMENT.williams_sequences()
            ],
        },
        "tooling": {
            "orchestrator_sha256": _sha256(
                (Path(TOURNAMENT.__file__).resolve()).read_bytes()
            ),
            "build_submission_sha256": "5" * 64,
            "package_audit_sha256": "6" * 64,
            "package_self_test_sha256": source_hashes["packaging/solver_main.py"],
            "official_sources_sha256": "7" * 64,
            "organizer_wrapper_sha256": "b" * 64,
            "solver_registry_sha256": "8" * 64,
            "official_source_checker_sha256": "9" * 64,
        },
        "build_host": {"platform": "Linux", "machine": "x86_64", "python": "3.12"},
        "variants": variants,
    }
    return build, package_descriptors


def _execution(
    *,
    ordinal: int,
    case_id: int,
    cycle: int,
    schedule_position: int,
    sequence: int,
    period: int,
    mode: str,
    warmup: bool,
    members: dict[str, bytes],
) -> dict:
    prefix = "warmup" if warmup else "run"
    stem = (
        f"{prefix}-{ordinal:05d}-case-{case_id:02d}-cycle-{cycle:02d}-"
        f"schedule-{schedule_position}-sequence-{sequence}-period-{period}-{mode}"
    )
    directory = "warmups" if warmup else "runs"
    result_path = f"{directory}/{stem}.json"
    log_path = f"logs/{stem}.log"
    source = _source_row(case_id)
    result_raw = _json_bytes(
        {
            "total_score": 1.0,
            "summary": {
                "num_tests": 1,
                "num_feasible": 1,
                "avg_cost": 1.0,
                "avg_runtime": 0.1,
            },
            "test_results": [source],
        }
    )
    log_raw = b"official fixture execution\n"
    members[f"opc-four-mode-timing/{result_path}"] = result_raw
    members[f"opc-four-mode-timing/{log_path}"] = log_raw
    return {
        "ordinal": ordinal,
        "discarded_warmup": warmup,
        "case_id": case_id,
        "case_index": case_id,
        "cycle": cycle,
        "schedule_position": schedule_position,
        "sequence": sequence,
        "period": period,
        "mode": mode,
        "runtime_seconds": 0.1,
        "quality": _quality(case_id),
        "result": _descriptor(result_path, result_raw),
        "log": _descriptor(log_path, log_raw),
    }


def _timing_manifest(
    build_raw: bytes,
    package_descriptors: dict[str, dict],
    members: dict[str, bytes],
) -> dict:
    artifacts = [
        {
            "test_id": case_id,
            "block_count": case_id + 21,
            "input": {
                "path": f"LiteTensorDataTest/config_{case_id + 21}/litedata_1.pth",
                "sha256": "a" * 64,
                "size_bytes": 1,
            },
            "label": {
                "path": f"LiteTensorDataTest/config_{case_id + 21}/litelabel_1.pth",
                "sha256": "b" * 64,
                "size_bytes": 1,
            },
        }
        for case_id in range(100)
    ]
    warmups = []
    runs = []
    ordinal = 0
    sequences = TOURNAMENT.williams_sequences()
    for case_id in range(100):
        warmup_sequence = case_id % len(sequences)
        for period, mode in enumerate(sequences[warmup_sequence]):
            ordinal += 1
            warmups.append(
                _execution(
                    ordinal=ordinal,
                    case_id=case_id,
                    cycle=-1,
                    schedule_position=period,
                    sequence=warmup_sequence,
                    period=period,
                    mode=mode,
                    warmup=True,
                    members=members,
                )
            )
        for schedule_position, sequence in enumerate(
            TOURNAMENT._rotated_sequence_indices(case_id, 0)
        ):
            for period, mode in enumerate(sequences[sequence]):
                ordinal += 1
                runs.append(
                    _execution(
                        ordinal=ordinal,
                        case_id=case_id,
                        cycle=0,
                        schedule_position=schedule_position,
                        sequence=sequence,
                        period=period,
                        mode=mode,
                        warmup=False,
                        members=members,
                    )
                )
    timing = {
        "schema_version": 3,
        "mode": TOURNAMENT.TIMING_MODE,
        "build_manifest": {
            "path": "build_manifest.json",
            "sha256": _sha256(build_raw),
            "source_commit": HEAD_SHA,
        },
        "contract": {
            "interface": "organizer op_wrapper.py spawning packaged executable once per solve",
            "modes": list(TOURNAMENT.MODES),
            "williams_sequences": [list(sequence) for sequence in sequences],
            "complete_sequences_per_case_cycle": 4,
            "no_my_opt_bin_override": True,
            "discarded_balanced_warmup_per_case": list(TOURNAMENT.MODES),
            "sequence_row_start": "rotated by (case_index + cycle) modulo 4",
            "environment": "allowlisted and thread-count pinned",
            "data_root_is_verified_official_root": True,
        },
        "build_source_verification": {
            "status": "PASS",
            "repository_head": HEAD_SHA,
            "repository_tree": "3" * 40,
            "git_archive_sha256": "4" * 64,
            "tracked_worktree": "clean",
            "tooling_sha256": json.loads(build_raw)["tooling"],
            "variant_source_contract": "PASS",
        },
        "official": {
            "floorset_commit": FLOORSET_SHA,
            "evaluator_sha256": "d" * 64,
            "organizer_wrapper_sha256": "b" * 64,
            "official_source_check_stdout_sha256": "e" * 64,
            "selected_public_case_artifacts": artifacts,
        },
        "environment": {
            "native_amd64_attestation": "PASS",
            "python": "3.12.11",
            "python_executable": "/opt/hostedtoolcache/Python/3.12/bin/python",
            "python_executable_sha256": "f" * 64,
            "sanitized_environment_keys": [],
            "fixed_environment": {},
            "pre": {},
            "post": {},
            "drift": {"status": "PASS", "immutable_fields_changed": []},
        },
        "schedule": {
            "case_ids": list(range(100)),
            "cycles": 1,
            "discarded_warmup_runs": 400,
            "total_runs": 1600,
            "total_process_executions": 2000,
        },
        "artifact_bindings": {
            "build_manifest": _descriptor("build_manifest.json", build_raw),
            "official_sources": {
                "path": "official_sources.json",
                "sha256": "7" * 64,
                "size_bytes": 1,
            },
            "official_evaluator": {
                "path": "iccad2026_evaluate.py",
                "sha256": "d" * 64,
                "size_bytes": 1,
            },
            "official_source_checker": {
                "path": "scripts/check_official_sources.py",
                "sha256": "9" * 64,
                "size_bytes": 1,
            },
            "selected_public_case_artifacts": artifacts,
            "packages": package_descriptors,
            "extracted_wrappers": {
                mode: {
                    "path": TOURNAMENT.INTERNAL_WRAPPER_PATH,
                    "sha256": "b" * 64,
                    "size_bytes": 100,
                }
                for mode in TOURNAMENT.MODES
            },
            "extracted_binaries": {
                mode: {
                    "path": TOURNAMENT.INTERNAL_BINARY_PATH,
                    "sha256": "a" * 64,
                    "size_bytes": 100,
                }
                for mode in TOURNAMENT.MODES
            },
            "extracted_package_files": {},
        },
        "post_timing_rehash": {
            "status": "PASS",
            "pre_sha256": "c" * 64,
            "post_sha256": "c" * 64,
        },
        "warmups": warmups,
        "runs": runs,
        "summary": TOURNAMENT._timing_summary(runs),
        "runtime_scenario_frontier": TOURNAMENT._runtime_scenario_frontier(
            runs, artifacts
        ),
    }
    binding_hash = _sha256(
        json.dumps(
            timing["artifact_bindings"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    timing["post_timing_rehash"]["pre_sha256"] = binding_hash
    timing["post_timing_rehash"]["post_sha256"] = binding_hash
    return timing


@dataclass(frozen=True)
class EvidenceFixture:
    root: Path
    artifact_zip: Path
    release_manifest: Path


PayloadMutator = Callable[[dict, dict, dict], None]
ReleaseMutator = Callable[[dict], None]
MemberMutator = Callable[[dict[str, bytes]], None]


def _evidence_fixture(
    tmp_path: Path,
    *,
    mutate: PayloadMutator | None = None,
    mutate_final: PayloadMutator | None = None,
    mutate_release: ReleaseMutator | None = None,
    mutate_members: MemberMutator | None = None,
) -> EvidenceFixture:
    root = tmp_path / "repository"
    root.mkdir()
    source_hashes, optimizer_text = _write_sources(root)
    members: dict[str, bytes] = {}
    build, package_descriptors = _build_manifest(
        source_hashes, optimizer_text, members
    )
    build_raw = _json_bytes(build)
    timing = _timing_manifest(build_raw, package_descriptors, members)
    result = {
        "total_score": 1.0,
        "summary": {
            "num_tests": 100,
            "num_feasible": 100,
            "avg_cost": 1.0,
            "avg_runtime": 0.1,
        },
        "test_results": [_source_row(case_id) for case_id in range(100)],
    }
    if mutate is not None:
        mutate(build, timing, result)

    build_raw = _json_bytes(build)
    timing["build_manifest"]["sha256"] = _sha256(build_raw)
    timing["artifact_bindings"]["build_manifest"] = _descriptor(
        "build_manifest.json", build_raw
    )
    binding_hash = _sha256(
        json.dumps(
            timing["artifact_bindings"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    timing["post_timing_rehash"]["pre_sha256"] = binding_hash
    timing["post_timing_rehash"]["post_sha256"] = binding_hash
    if mutate_final is not None:
        mutate_final(build, timing, result)
    timing_raw = _json_bytes(timing)
    result_path = root / "results" / "integrated_v33_beta.json"
    result_path.parent.mkdir(parents=True)
    result_raw = _json_bytes(result)
    result_path.write_bytes(result_raw)

    selected_package_raw = members[
        "opc-four-mode-build/"
        + build["variants"][1]["package"]["path"]
    ]
    release_package_path = root / "submission" / "iccad2026_submission.tar.gz"
    release_package_path.parent.mkdir()
    release_package_path.write_bytes(selected_package_raw)

    members[AUDIT.BUILD_MANIFEST_NAME] = build_raw
    members[AUDIT.TIMING_MANIFEST_NAME] = timing_raw
    members[AUDIT.ENVIRONMENT_NAME] = (
        "github_repository=Lawrence-eth/opc-eda-2026\n"
        f"github_sha={HEAD_SHA}\n"
        f"github_run_id={RUN_ID}\n"
        f"github_run_attempt={RUN_ATTEMPT}\n"
        "runner_arch=X64\n"
        "image_os=ubuntu24\n"
        "image_version=20260701.1\n"
        "Linux native-fixture x86_64\n"
    ).encode("utf-8")
    if mutate_members is not None:
        mutate_members(members)
    ledger_raw = "".join(
        f"{_sha256(raw)}  {name}\n" for name, raw in sorted(members.items())
    ).encode("utf-8")
    members[AUDIT.LEDGER_NAME] = ledger_raw

    artifact_zip = tmp_path / ARTIFACT_ZIP_NAME
    with zipfile.ZipFile(artifact_zip, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, raw in sorted(members.items()):
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, raw)

    selector_path = root / "results/research/policy_tournament_v1/sealed_selector.json"
    selector_path.parent.mkdir(parents=True)
    selector_raw = _json_bytes(
        {
            "schema_version": 2,
            "status": "sealed_confirmation_failed_fallback_off",
            "final_mode": "off",
        }
    )
    selector_path.write_bytes(selector_raw)
    package_sha = _sha256(selected_package_raw)
    release = {
        "schema_version": 2,
        "release": "v33-beta-test",
        "verified_on": "2026-07-15",
        "solver": {
            "version": "v33-beta-test",
            "commit": HEAD_SHA,
            "entrypoint": "contest_solution/my_optimizer.py",
            "learned_order_mode": SELECTED_MODE,
            "sources": source_hashes,
        },
        "public_result": {
            "path": "results/integrated_v33_beta.json",
            "sha256": _sha256(result_raw),
            "total_score": 1.0,
            "num_cases": 100,
            "num_feasible": 100,
        },
        "submission_package": {
            "path": "submission/iccad2026_submission.tar.gz",
            "sha256": package_sha,
        },
        "floorset": {
            "repository": "https://github.com/IntelLabs/FloorSet.git",
            "commit": FLOORSET_SHA,
        },
        "decision_evidence": {
            "native_tournament": {
                "run_id": RUN_ID,
                "run_attempt": RUN_ATTEMPT,
                "run_url": (
                    "https://github.com/Lawrence-eth/opc-eda-2026/actions/runs/"
                    f"{RUN_ID}"
                ),
                "head_sha": HEAD_SHA,
                "conclusion": "success",
                "artifact_id": 123456789,
                "artifact_name": ARTIFACT_NAME,
                "artifact_size_bytes": artifact_zip.stat().st_size,
                "artifact_digest_sha256": "0" * 64,
                "preserved_asset_name": ARTIFACT_ZIP_NAME,
                "preserved_asset_size_bytes": artifact_zip.stat().st_size,
                "preserved_asset_sha256": AUDIT._sha256_path(artifact_zip),
                "build_manifest_schema_version": build["schema_version"],
                "timing_manifest_schema_version": timing["schema_version"],
                "build_manifest_sha256": _sha256(build_raw),
                "timing_manifest_sha256": _sha256(timing_raw),
                "evidence_bundle_sha256": _sha256(ledger_raw),
                "selected_mode": SELECTED_MODE,
                "selected_package_sha256": package_sha,
                "selected_source_patch_changed": False,
            },
            "sealed_selector": {
                "path": "results/research/policy_tournament_v1/sealed_selector.json",
                "sha256": _sha256(selector_raw),
                "status": "sealed_confirmation_failed_fallback_off",
            },
            "sealed_policy_overridden": True,
            "rationale": "Fixture expected-value override backed by native evidence.",
        },
    }
    if mutate_release is not None:
        mutate_release(release)
    release_manifest = root / "results/release_manifest.json"
    release_manifest.write_bytes(_json_bytes(release))
    return EvidenceFixture(root, artifact_zip, release_manifest)


def _audit(fixture: EvidenceFixture) -> dict:
    return AUDIT.audit_native_evidence(
        fixture.artifact_zip,
        fixture.release_manifest,
        root=fixture.root,
    )


def test_complete_schema2_build_schema3_timing_bundle_passes(tmp_path):
    fixture = _evidence_fixture(tmp_path)

    result = _audit(fixture)

    assert result == {
        "status": "PASS",
        "run_id": RUN_ID,
        "run_attempt": RUN_ATTEMPT,
        "head_sha": HEAD_SHA,
        "selected_mode": SELECTED_MODE,
        "selected_package_sha256": hashlib.sha256(
            b"package-replacement\n"
        ).hexdigest(),
        "files_verified": 4015,
        "timed_runs": 1600,
        "feasible_runs": 1600,
        "stable_mode_cases": 400,
        "source_position_cases": 100,
    }


@pytest.mark.parametrize(
    ("mutate_release", "message"),
    [
        (
            lambda release: release["decision_evidence"]["native_tournament"].__setitem__(
                "preserved_asset_sha256", "f" * 64
            ),
            "digest differs",
        ),
        (
            lambda release: release["decision_evidence"]["native_tournament"].__setitem__(
                "preserved_asset_size_bytes", 1
            ),
            "size differs",
        ),
        (
            lambda release: release["decision_evidence"]["native_tournament"].__setitem__(
                "preserved_asset_name", "unbound.zip"
            ),
            "name does not bind|name differs",
        ),
    ],
)
def test_preserved_zip_identity_is_fail_closed(tmp_path, mutate_release, message):
    fixture = _evidence_fixture(tmp_path, mutate_release=mutate_release)

    with pytest.raises(AUDIT.NativeEvidenceError, match=message):
        _audit(fixture)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a\\b", "a//b"])
def test_zip_and_ledger_paths_reject_noncanonical_names(name):
    with pytest.raises(AUDIT.NativeEvidenceError, match="unsafe"):
        AUDIT._safe_member_name(name)


def test_ledger_rejects_duplicate_recursive_and_malformed_entries():
    digest = "a" * 64
    with pytest.raises(AUDIT.NativeEvidenceError, match="duplicate"):
        AUDIT._parse_ledger(
            f"{digest}  file\n{digest}  file\n".encode("utf-8")
        )
    with pytest.raises(AUDIT.NativeEvidenceError, match="recursive"):
        AUDIT._parse_ledger(
            f"{digest}  {AUDIT.LEDGER_NAME}\n".encode("utf-8")
        )
    with pytest.raises(AUDIT.NativeEvidenceError, match="malformed"):
        AUDIT._parse_ledger(f"{digest} *file\n".encode("utf-8"))


def test_ledger_inventory_must_cover_every_zip_member(tmp_path):
    fixture = _evidence_fixture(tmp_path)
    tampered = tmp_path / "tampered" / ARTIFACT_ZIP_NAME
    tampered.parent.mkdir()
    with zipfile.ZipFile(fixture.artifact_zip) as source, zipfile.ZipFile(
        tampered, "w", compression=zipfile.ZIP_STORED
    ) as target:
        for info in source.infolist():
            target.writestr(info, source.read(info.filename))
        extra = zipfile.ZipInfo("unledgered.txt")
        extra.create_system = 3
        extra.external_attr = (stat.S_IFREG | 0o644) << 16
        target.writestr(extra, b"not in ledger")
    release = json.loads(fixture.release_manifest.read_text(encoding="utf-8"))
    native = release["decision_evidence"]["native_tournament"]
    native["preserved_asset_size_bytes"] = tampered.stat().st_size
    native["preserved_asset_sha256"] = AUDIT._sha256_path(tampered)
    fixture.release_manifest.write_bytes(_json_bytes(release))
    tampered_fixture = EvidenceFixture(fixture.root, tampered, fixture.release_manifest)

    with pytest.raises(AUDIT.NativeEvidenceError, match="inventory differ"):
        _audit(tampered_fixture)


def test_release_and_embedded_json_reject_duplicate_keys(tmp_path):
    fixture = _evidence_fixture(tmp_path)
    fixture.release_manifest.write_text(
        '{"schema_version":2,"schema_version":2}', encoding="utf-8"
    )

    with pytest.raises(AUDIT.NativeEvidenceError, match="duplicate JSON key"):
        _audit(fixture)


def test_exact_release_build_and_timing_schema_versions_are_required(tmp_path):
    def mutate(build, timing, _result):
        build["schema_version"] = 91
        timing["schema_version"] = 92

    fixture = _evidence_fixture(tmp_path, mutate=mutate)

    with pytest.raises(AUDIT.NativeEvidenceError, match="schema"):
        _audit(fixture)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda build, _timing, _result: build.__setitem__("mode", "forged"),
        lambda build, _timing, _result: build["variants"][1][
            "source_patch"
        ].__setitem__("assignment_after", 'self._learned_order_mode = "off"'),
        lambda build, _timing, _result: build["variants"][1][
            "solver_components"
        ].pop("contest_solution/dissect.py"),
    ],
)
def test_embedded_build_manifest_requires_full_schema_contract(tmp_path, mutate):
    fixture = _evidence_fixture(tmp_path, mutate=mutate)

    with pytest.raises(AUDIT.NativeEvidenceError):
        _audit(fixture)


def test_public_result_declared_digest_is_verified(tmp_path):
    fixture = _evidence_fixture(
        tmp_path,
        mutate_release=lambda release: release["public_result"].__setitem__(
            "sha256", "f" * 64
        ),
    )

    with pytest.raises(AUDIT.NativeEvidenceError, match="public result"):
        _audit(fixture)


def test_native_run_authority_fields_are_verified(tmp_path):
    def mutate_release(release):
        native = release["decision_evidence"]["native_tournament"]
        native["conclusion"] = "cancelled"
        native["run_url"] = "https://example.invalid/forged"
        native["artifact_name"] = "forged-artifact"

    fixture = _evidence_fixture(tmp_path, mutate_release=mutate_release)

    with pytest.raises(AUDIT.NativeEvidenceError):
        _audit(fixture)


def test_timing_run_multiplicity_cannot_be_forged_by_summary(tmp_path):
    def mutate(_build, timing, _result):
        kept_off_cases = set()
        for row in timing["runs"]:
            if row["mode"] != "off":
                continue
            if row["case_id"] not in kept_off_cases:
                kept_off_cases.add(row["case_id"])
            else:
                row["mode"] = "additive"
        assert sum(row["mode"] == "off" for row in timing["runs"]) == 100
        assert timing["summary"]["off"]["runs"] == 400

    fixture = _evidence_fixture(tmp_path, mutate=mutate)

    with pytest.raises(
        AUDIT.NativeEvidenceError, match="timing|schedule|runs|execution|invalid"
    ):
        _audit(fixture)


def test_williams_run_schedule_metadata_is_verified(tmp_path):
    def mutate(_build, timing, _result):
        timing["runs"][0]["sequence"] = 3
        timing["runs"][0]["period"] = 3
        timing["runs"][0]["schedule_position"] = 3

    fixture = _evidence_fixture(tmp_path, mutate=mutate)

    with pytest.raises(AUDIT.NativeEvidenceError, match="schedule|Williams"):
        _audit(fixture)


def test_warmup_schedule_and_quality_are_verified(tmp_path):
    def mutate(_build, timing, _result):
        timing["warmups"][0].update(
            {
                "discarded_warmup": False,
                "case_id": 99,
                "mode": "replacement",
                "quality": {"is_feasible": False},
            }
        )

    fixture = _evidence_fixture(tmp_path, mutate=mutate)

    with pytest.raises(AUDIT.NativeEvidenceError, match="warmup"):
        _audit(fixture)


def test_nonfinite_runtime_is_rejected_at_json_boundary(tmp_path):
    def mutate(_build, timing, _result):
        timing["runs"][0]["runtime_seconds"] = float("nan")

    fixture = _evidence_fixture(tmp_path, mutate=mutate)

    with pytest.raises(AUDIT.NativeEvidenceError, match="non-finite|invalid"):
        _audit(fixture)


def test_timing_summary_is_recomputed_from_raw_runs(tmp_path):
    def mutate(_build, timing, _result):
        timing["summary"]["replacement"]["median_seconds"] = 999.0

    fixture = _evidence_fixture(tmp_path, mutate=mutate)

    with pytest.raises(AUDIT.NativeEvidenceError, match="summary"):
        _audit(fixture)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda build, _timing, _result: build["variants"][1]["package"].__setitem__(
            "size_bytes", 1
        ),
        lambda _build, timing, _result: timing["runs"][0]["result"].__setitem__(
            "size_bytes", 1
        ),
        lambda _build, timing, _result: timing["runs"][0].__setitem__(
            "log", copy.deepcopy(timing["runs"][1]["log"])
        ),
    ],
)
def test_ledger_bound_descriptors_require_exact_path_hash_and_size(tmp_path, mutate):
    fixture = _evidence_fixture(tmp_path, mutate=mutate)

    with pytest.raises(
        AUDIT.NativeEvidenceError,
        match="descriptor|size|path|package|checksum ledger|not bound",
    ):
        _audit(fixture)


def test_rehash_digest_is_recomputed_from_artifact_bindings(tmp_path):
    def mutate_final(_build, timing, _result):
        timing["post_timing_rehash"]["pre_sha256"] = "d" * 64
        timing["post_timing_rehash"]["post_sha256"] = "d" * 64

    fixture = _evidence_fixture(tmp_path, mutate_final=mutate_final)

    with pytest.raises(AUDIT.NativeEvidenceError, match="rehash|bindings|drifted"):
        _audit(fixture)


def test_all_repeated_quality_fields_must_be_stable(tmp_path):
    def mutate(_build, timing, _result):
        first_off = next(row for row in timing["runs"] if row["mode"] == "off")
        first_off["quality"]["cost"] = 2.0

    fixture = _evidence_fixture(tmp_path, mutate=mutate)

    with pytest.raises(AUDIT.NativeEvidenceError, match="quality|stable|drift"):
        _audit(fixture)


def test_timing_quality_is_replayed_from_ledger_bound_raw_results(tmp_path):
    def mutate(_build, timing, result):
        result["test_results"][0]["positions"][0][0] = 0.25
        digest = AUDIT._canonical_positions_sha256(
            result["test_results"][0]["positions"], 21
        )
        for row in [*timing["warmups"], *timing["runs"]]:
            if row["case_id"] == 0 and row["mode"] == SELECTED_MODE:
                row["quality"]["positions_sha256"] = digest

    fixture = _evidence_fixture(tmp_path, mutate=mutate)

    with pytest.raises(AUDIT.NativeEvidenceError, match="raw|result|quality"):
        _audit(fixture)


def test_runtime_scenario_frontier_is_recomputed_from_raw_runs(tmp_path):
    def mutate(_build, timing, _result):
        timing["runtime_scenario_frontier"]["nondominated_modes"] = ["off"]

    fixture = _evidence_fixture(tmp_path, mutate=mutate)

    with pytest.raises(AUDIT.NativeEvidenceError, match="scenario|frontier"):
        _audit(fixture)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda _build, timing, _result: timing["environment"].__setitem__(
                "native_amd64_attestation", "FAIL"
            ),
            "attestation",
        ),
        (
            lambda _build, timing, _result: timing["post_timing_rehash"].__setitem__(
                "status", "FAIL"
            ),
            "drifted",
        ),
        (
            lambda _build, timing, _result: timing["runs"][0]["quality"].__setitem__(
                "is_feasible", False
            ),
            "infeasible",
        ),
        (
            lambda _build, timing, _result: timing["runs"][1]["quality"].__setitem__(
                "positions_sha256", "f" * 64
            ),
            "unstable|mismatch|quality differs",
        ),
    ],
)
def test_existing_timing_integrity_gates_fail_closed(tmp_path, mutate, message):
    fixture = _evidence_fixture(tmp_path, mutate=mutate)

    with pytest.raises(AUDIT.NativeEvidenceError, match=message):
        _audit(fixture)


def test_public_result_path_must_remain_inside_repository(tmp_path):
    fixture = _evidence_fixture(
        tmp_path,
        mutate_release=lambda release: release["public_result"].__setitem__(
            "path", "../../outside.json"
        ),
    )

    with pytest.raises(AUDIT.NativeEvidenceError, match="escapes"):
        _audit(fixture)


def test_native_environment_text_binds_run_identity(tmp_path):
    def mutate_members(members):
        members[AUDIT.ENVIRONMENT_NAME] = members[AUDIT.ENVIRONMENT_NAME].replace(
            f"github_run_id={RUN_ID}".encode(), b"github_run_id=1"
        )

    fixture = _evidence_fixture(tmp_path, mutate_members=mutate_members)

    with pytest.raises(AUDIT.NativeEvidenceError, match="run identity"):
        _audit(fixture)
