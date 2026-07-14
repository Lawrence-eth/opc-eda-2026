import copy
import hashlib
import importlib.util
import json
import shutil
import tarfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "package_mode_tournament.py"
    spec = importlib.util.spec_from_file_location("package_mode_tournament", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOURNAMENT = _load_module()


def _optimizer_source():
    return (
        "class MyOptimizer:\n"
        "    def __init__(self):\n"
        f"{TOURNAMENT.DEFAULT_ASSIGNMENT}\n"
        "        self.other = True\n"
    )


@pytest.mark.parametrize("mode", TOURNAMENT.MODES)
def test_patch_changes_only_exact_copied_default(tmp_path, mode):
    source = tmp_path / "source.py"
    copied = tmp_path / "copied.py"
    source.write_text(_optimizer_source())
    shutil.copy2(source, copied)
    original = source.read_bytes()

    result = TOURNAMENT.patch_copied_optimizer(copied, mode)

    assert source.read_bytes() == original
    assert result["sha256_before"] == hashlib.sha256(original).hexdigest()
    assert result["sha256_after"] == hashlib.sha256(copied.read_bytes()).hexdigest()
    expected = f'        self._learned_order_mode = "{mode}"'
    assert TOURNAMENT.ASSIGNMENT_RE.findall(copied.read_text()) == [expected]
    assert result["changed"] is (mode != "replacement")


@pytest.mark.parametrize(
    "source",
    [
        "",
        '        self._learned_order_mode="replacement"\n',
        _optimizer_source() + TOURNAMENT.DEFAULT_ASSIGNMENT + "\n",
        _optimizer_source().replace("replacement", "off"),
    ],
)
def test_patch_fails_closed_on_nonexact_or_duplicate_assignment(tmp_path, source):
    path = tmp_path / "my_optimizer.py"
    path.write_text(source)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="exactly one byte-exact"):
        TOURNAMENT.patch_copied_optimizer(path, "additive")
    assert path.read_bytes() == before


def test_whole_tree_gate_allows_only_copied_optimizer(tmp_path):
    base = tmp_path / "base"
    variant = tmp_path / "variant"
    optimizer = base / TOURNAMENT.OPTIMIZER_RELATIVE_PATH
    optimizer.parent.mkdir(parents=True)
    optimizer.write_text(_optimizer_source())
    (base / "unchanged.txt").write_text("bound\n")
    baseline = TOURNAMENT._tree_hashes(base)

    shutil.copytree(base, variant)
    TOURNAMENT.patch_copied_optimizer(
        variant / TOURNAMENT.OPTIMIZER_RELATIVE_PATH, "off"
    )
    TOURNAMENT._verify_only_mode_change(baseline, variant, "off")

    (variant / "unchanged.txt").write_text("drift\n")
    with pytest.raises(ValueError, match="expected exactly"):
        TOURNAMENT._verify_only_mode_change(baseline, variant, "off")


def test_replacement_control_requires_byte_identical_snapshot(tmp_path):
    root = tmp_path / "snapshot"
    optimizer = root / TOURNAMENT.OPTIMIZER_RELATIVE_PATH
    optimizer.parent.mkdir(parents=True)
    optimizer.write_text(_optimizer_source())
    baseline = TOURNAMENT._tree_hashes(root)
    TOURNAMENT.patch_copied_optimizer(optimizer, "replacement")
    assert TOURNAMENT._verify_only_mode_change(baseline, root, "replacement") == baseline


def test_package_selftest_must_force_replacement_for_both_probes(tmp_path):
    path = tmp_path / "solver_main.py"
    required = (
        '    optimizer._learned_order_enabled = True\n'
        '    optimizer._learned_order_mode = "replacement"\n'
        '    abstention_optimizer._learned_order_enabled = True\n'
        '    abstention_optimizer._learned_order_mode = "replacement"\n'
    )
    path.write_text(required)
    TOURNAMENT._verify_mode_independent_self_test(path)

    path.write_text(required.replace(
        'abstention_optimizer._learned_order_mode = "replacement"',
        'abstention_optimizer._learned_order_mode = "off"',
    ))
    with pytest.raises(ValueError, match="not explicitly replacement-capable"):
        TOURNAMENT._verify_mode_independent_self_test(path)


def test_williams_sequences_are_exact_and_first_order_balanced():
    sequences = TOURNAMENT.williams_sequences()
    assert sequences == (
        ("off", "replacement", "additive_first_pass", "additive"),
        ("replacement", "additive", "off", "additive_first_pass"),
        ("additive", "additive_first_pass", "replacement", "off"),
        ("additive_first_pass", "off", "additive", "replacement"),
    )
    for period in range(4):
        assert {row[period] for row in sequences} == set(TOURNAMENT.MODES)
    transitions = [
        (left, right)
        for row in sequences
        for left, right in zip(row, row[1:])
    ]
    assert len(transitions) == 12
    assert set(transitions) == {
        (left, right)
        for left in TOURNAMENT.MODES
        for right in TOURNAMENT.MODES
        if left != right
    }


@pytest.mark.parametrize("treatments", [("a", "b", "c"), ("a", "a")])
def test_williams_sequences_reject_invalid_treatments(treatments):
    with pytest.raises(ValueError):
        TOURNAMENT.williams_sequences(treatments)


@pytest.mark.parametrize(
    "path",
    ["../escape.tar.gz", "/absolute.tar.gz", "a/../../b", "a\\b"],
)
def test_manifest_paths_cannot_escape(path, tmp_path):
    with pytest.raises(ValueError, match="unsafe"):
        TOURNAMENT._safe_manifest_path(tmp_path, path, "package")


def test_manifest_path_resolves_within_manifest_directory(tmp_path):
    assert TOURNAMENT._safe_manifest_path(
        tmp_path, "packages/off.tar.gz", "package"
    ) == (tmp_path / "packages/off.tar.gz").resolve()


def test_audit_parser_requires_hash_smoke_and_amd64():
    digest = "a" * 64
    stdout = (
        "Submission package audit: PASS\n"
        f"  archive_sha256={digest}\n"
        "  members=10 expanded_bytes=100\n"
        "  elf_machine=AMD64\n"
        "  max_glibc=2.41\n"
        "  smoke=PASS\n"
    )
    assert TOURNAMENT._audit_details(stdout, digest)["smoke"] == "PASS"
    with pytest.raises(ValueError, match="hash"):
        TOURNAMENT._audit_details(stdout, "b" * 64)
    with pytest.raises(ValueError, match="smoke test"):
        TOURNAMENT._audit_details(stdout.replace("smoke=PASS", "smoke=SKIPPED"), digest)


def _official_result(case_id=79, runtime=0.25):
    return {
        "submission_name": "op_wrapper",
        "timestamp": "2026-07-14T00:00:00",
        "total_score": 1.0,
        "test_results": [
            {
                "test_id": case_id,
                "block_count": 2,
                "is_feasible": True,
                "hpwl_gap": 0.1,
                "area_gap": 0.2,
                "violations_relative": 0.0,
                "runtime_seconds": runtime,
                "cost": 1.1,
                "positions": [[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0]],
                "error": None,
            }
        ],
        "summary": {
            "num_tests": 1,
            "num_feasible": 1,
            "avg_cost": 1.1,
            "avg_runtime": runtime,
        },
    }


def test_official_result_validator_binds_case_quality_and_positions(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps(_official_result()))
    result = TOURNAMENT._validate_official_result(path, 79)
    assert result["runtime_seconds"] == 0.25
    assert result["quality"]["test_id"] == 79
    assert len(result["quality"]["positions_sha256"]) == 64


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda result: result["test_results"][0].__setitem__("test_id", 80), "test ID"),
        (lambda result: result["test_results"][0].__setitem__("is_feasible", False), "infeasible"),
        (lambda result: result["test_results"][0].__setitem__("runtime_seconds", 0), "positive"),
        (lambda result: result["test_results"][0].__setitem__("positions", []), "block count"),
        (lambda result: result["summary"].__setitem__("num_feasible", 0), "summary"),
    ],
)
def test_official_result_validator_fails_closed(tmp_path, mutate, message):
    payload = _official_result()
    mutate(payload)
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=message):
        TOURNAMENT._validate_official_result(path, 79)


def test_timing_summary_preserves_per_case_replicates():
    runs = []
    for mode_index, mode in enumerate(TOURNAMENT.MODES):
        for case_id in (79, 80):
            for repeat in range(4):
                runs.append({
                    "mode": mode,
                    "case_id": case_id,
                    "runtime_seconds": 0.1 * (mode_index + 1) + 0.01 * repeat,
                })
    summary = TOURNAMENT._timing_summary(runs)
    assert set(summary) == set(TOURNAMENT.MODES)
    assert all(summary[mode]["runs"] == 8 for mode in TOURNAMENT.MODES)
    assert all(
        summary[mode]["by_case"]["79"]["runs"] == 4
        for mode in TOURNAMENT.MODES
    )


def test_selected_public_cases_bind_official_order_and_exact_bytes(tmp_path):
    for block_count in range(21, 121):
        config = tmp_path / "LiteTensorDataTest" / f"config_{block_count}"
        config.mkdir(parents=True)
        (config / "litedata_1.pth").write_bytes(f"input-{block_count}".encode())
        (config / "litelabel_1.pth").write_bytes(f"label-{block_count}".encode())
    artifacts = TOURNAMENT._selected_public_case_artifacts(tmp_path, [0, 79, 99])
    assert [(row["test_id"], row["block_count"]) for row in artifacts] == [
        (0, 21),
        (79, 100),
        (99, 120),
    ]
    assert artifacts[1]["input"]["sha256"] == hashlib.sha256(b"input-100").hexdigest()

    extra = tmp_path / "LiteTensorDataTest/config_100/litedata_2.pth"
    extra.write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="exactly one input/label"):
        TOURNAMENT._selected_public_case_artifacts(tmp_path, [79])


def test_output_and_workspace_are_required_outside_source_repository(tmp_path):
    inside = ROOT / "results/work/forbidden"
    with pytest.raises(ValueError, match="outside"):
        TOURNAMENT._ensure_outside_repository(inside, ROOT, "output")
    outside = tmp_path / "allowed"
    assert TOURNAMENT._ensure_outside_repository(outside, ROOT, "output") == outside.resolve()


def test_safe_tar_extraction_rejects_links(tmp_path):
    archive = tmp_path / "bad.tar"
    target = tmp_path / "target.txt"
    target.write_text("target")
    with tarfile.open(archive, "w") as stream:
        stream.add(target, arcname="root/file.txt")
        info = tarfile.TarInfo("root/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "file.txt"
        stream.addfile(info)
    with pytest.raises(ValueError, match="links and special"):
        TOURNAMENT._safe_extract_tar(
            archive, tmp_path / "extract", expected_prefix="root"
        )


def _build_manifest():
    variants = []
    for mode in TOURNAMENT.MODES:
        variants.append({
            "mode": mode,
            "archived_source_sha256": {
                "source_fallback/my_optimizer.py": "a" * 64,
            },
            "package": {
                "path": f"packages/{mode}.tar.gz",
                "sha256": "b" * 64,
                "size_bytes": 100,
            },
            "audit": {"status": "PASS"},
        })
    return {
        "schema_version": 1,
        "mode": TOURNAMENT.BUILD_MODE,
        "source": {
            "commit": "c" * 40,
            "tree": "d" * 40,
            "git_archive_sha256": "e" * 64,
            "base_optimizer_sha256": "f" * 64,
        },
        "contract": {
            "modes": list(TOURNAMENT.MODES),
            "timing_design": [
                list(row) for row in TOURNAMENT.williams_sequences()
            ],
        },
        "tooling": {
            "orchestrator_sha256": hashlib.sha256(
                (ROOT / "scripts/package_mode_tournament.py").read_bytes()
            ).hexdigest(),
            "build_submission_sha256": "1" * 64,
            "package_audit_sha256": "2" * 64,
            "package_self_test_sha256": "3" * 64,
            "official_sources_sha256": "4" * 64,
            "organizer_wrapper_sha256": "5" * 64,
        },
        "build_host": {},
        "variants": variants,
    }


def test_build_manifest_requires_all_four_ordered_variants(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_build_manifest()))
    _raw, _manifest, variants = TOURNAMENT._validate_build_manifest(path)
    assert tuple(variants) == TOURNAMENT.MODES

    payload = _build_manifest()
    payload["variants"] = list(reversed(payload["variants"]))
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="order changed"):
        TOURNAMENT._validate_build_manifest(path)


def test_build_manifest_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        TOURNAMENT._validate_build_manifest(path)
