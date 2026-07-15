import copy
import hashlib
import importlib.util
import json
import math
import os
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
    component_names = TOURNAMENT._component_names(ROOT)
    solver_names = [f"contest_solution/{name}" for name in component_names]
    support = {
        "packaging/torch_stub.py": "6" * 64,
        "packaging/eval_stub.py": "7" * 64,
        "packaging/solver_main.py": "8" * 64,
    }
    variants = []
    for mode in TOURNAMENT.MODES:
        after = ("f" if mode == "replacement" else str(TOURNAMENT.MODES.index(mode))) * 64
        solver = {name: "9" * 64 for name in solver_names}
        solver["contest_solution/my_optimizer.py"] = after
        archived = {
            **{f"source_fallback/{Path(name).name}": digest for name, digest in solver.items()},
            "source_fallback/torch.py": support["packaging/torch_stub.py"],
            "source_fallback/iccad2026_evaluate.py": support["packaging/eval_stub.py"],
            "source_fallback/solver_main.py": support["packaging/solver_main.py"],
        }
        variants.append({
            "mode": mode,
            "source_patch": {
                "path": "contest_solution/my_optimizer.py",
                "assignment_before": 'self._learned_order_mode = "replacement"',
                "assignment_after": f'self._learned_order_mode = "{mode}"',
                "sha256_before": "f" * 64,
                "sha256_after": after,
                "changed": mode != "replacement",
            },
            "solver_components": solver,
            "package_support_sources": dict(support),
            "archived_source_sha256": archived,
            "binary": {
                "path": TOURNAMENT.INTERNAL_BINARY_PATH,
                "sha256": "a" * 64,
                "size_bytes": 200,
            },
            "wrapper": {
                "path": TOURNAMENT.INTERNAL_WRAPPER_PATH,
                "sha256": "5" * 64,
                "size_bytes": 300,
            },
            "package": {
                "path": f"packages/iccad2026_submission_{mode}.tar.gz",
                "sha256": "b" * 64,
                "size_bytes": 100,
            },
            "audit": {
                "status": "PASS",
                "details": {
                    "archive_sha256": "b" * 64,
                    "members": "10 expanded_bytes=100",
                    "elf_machine": "AMD64",
                    "max_glibc": "2.41",
                    "smoke": "PASS",
                    "default_mode": mode,
                },
                "log": {
                    "path": f"logs/{mode}.audit.log",
                    "sha256": "c" * 64,
                    "size_bytes": 10,
                },
            },
            "build_log": {
                "path": f"logs/{mode}.build.log",
                "sha256": "d" * 64,
                "size_bytes": 11,
            },
        })
    return {
        "schema_version": TOURNAMENT.BUILD_SCHEMA_VERSION,
        "mode": TOURNAMENT.BUILD_MODE,
        "source": {
            "commit": "c" * 40,
            "tree": "d" * 40,
            "git_archive_sha256": "e" * 64,
            "base_optimizer_sha256": "f" * 64,
        },
        "contract": {
            "modes": list(TOURNAMENT.MODES),
            "source_mutation": "frozen isolated patch",
            "build_environment": "AMD64 container",
            "audit": "strict smoke",
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
            "solver_registry_sha256": "6" * 64,
            "official_source_checker_sha256": "7" * 64,
        },
        "build_host": {"platform": "Linux", "machine": "x86_64", "python": "3.13"},
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


def test_schema_one_build_manifests_are_intentionally_rejected(tmp_path):
    payload = _build_manifest()
    payload["schema_version"] = 1
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="schema-1 manifests are intentionally rejected"):
        TOURNAMENT._validate_build_manifest(path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value["variants"][0]["source_patch"].__setitem__("assignment_after", 'self._learned_order_mode = "replacement"'), "source patch contract"),
        (lambda value: value["variants"][0]["solver_components"].pop("contest_solution/dissect.py"), "solver registry source inventory"),
        (lambda value: value["variants"][0]["wrapper"].__setitem__("sha256", "0" * 64), "wrapper differs"),
        (lambda value: value["variants"][0]["binary"].__setitem__("path", "wrong/binary"), "binary path"),
        (lambda value: value["variants"][0]["audit"]["log"].pop("size_bytes"), "must have exactly"),
        (lambda value: value["variants"][0]["audit"]["details"].__setitem__("default_mode", "replacement"), "audit attestation"),
    ],
)
def test_build_manifest_rejects_tampered_bound_fields(tmp_path, mutate, message):
    payload = _build_manifest()
    mutate(payload)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=message):
        TOURNAMENT._validate_build_manifest(path)


def _native_probe(cpuinfo=None, platform_text="Linux-6.8-x86_64-with-glibc2.39"):
    return {
        "platform_machine": "x86_64",
        "platform_platform": platform_text,
        "uname_machine": "x86_64",
        "uname_sysname": "Linux",
        "uname_release": "6.8.0",
        "cpuinfo": cpuinfo or (
            "processor : 0\n"
            "vendor_id : GenuineIntel\n"
            "model name : AMD EPYC GitHub Runner\n"
            "flags : fpu sse2 lm hypervisor avx2\n"
        ),
    }


def test_native_amd64_attestation_allows_virtualized_github_runner():
    result = TOURNAMENT._native_amd64_attestation(_native_probe(), {})
    assert result["status"] == "PASS"
    assert result["hypervisor_flag"] is True
    assert result["cpu_vendors"] == ["GenuineIntel"]


@pytest.mark.parametrize(
    ("probe", "environment", "message"),
    [
        (_native_probe(cpuinfo="vendor_id : GenuineIntel\nmodel name : QEMU Virtual CPU\nflags : lm sse2\n"), {}, "QEMU"),
        (_native_probe(platform_text="Linux-aarch64"), {}, "non-x86"),
        (_native_probe(), {"QEMU_CPU": "max"}, "environment markers"),
        ({**_native_probe(), "uname_machine": "aarch64"}, {}, "agreeing AMD64"),
    ],
)
def test_native_amd64_attestation_rejects_emulation_contradictions(
    probe, environment, message
):
    with pytest.raises(ValueError, match=message):
        TOURNAMENT._native_amd64_attestation(probe, environment)


def test_timing_environment_is_allowlisted_and_thread_pinned():
    result = TOURNAMENT._sanitized_timing_environment({
        "PATH": "/usr/bin",
        "HOME": "/tmp/home",
        "MY_OPT_BIN": "/attacker",
        "PYTHONPATH": "/attacker",
        "SOLVER_DEBUG": "1",
        "QEMU_CPU": "max",
        "SECRET_TOKEN": "secret",
    })
    assert result["PATH"] == "/usr/bin"
    assert result["PYTHONHASHSEED"] == "0"
    assert result["OMP_NUM_THREADS"] == "1"
    assert not ({"MY_OPT_BIN", "PYTHONPATH", "SOLVER_DEBUG", "QEMU_CPU", "SECRET_TOKEN"} & set(result))


def test_rotated_williams_starts_change_by_case_and_cycle_and_remain_complete():
    assert TOURNAMENT._rotated_sequence_indices(0, 0) == (0, 1, 2, 3)
    assert TOURNAMENT._rotated_sequence_indices(1, 0) == (1, 2, 3, 0)
    assert TOURNAMENT._rotated_sequence_indices(1, 1) == (2, 3, 0, 1)
    assert all(
        set(TOURNAMENT._rotated_sequence_indices(case, cycle)) == {0, 1, 2, 3}
        for case in range(5)
        for cycle in range(5)
    )


def test_discarded_warmup_is_one_balanced_pass_per_case():
    for case_index in range(8):
        schedule = TOURNAMENT._discarded_warmup_schedule(case_index)
        assert len(schedule) == len(TOURNAMENT.MODES)
        assert {mode for _sequence, _period, mode in schedule} == set(TOURNAMENT.MODES)
        assert {sequence for sequence, _period, _mode in schedule} == {
            case_index % len(TOURNAMENT.MODES)
        }
        assert [period for _sequence, period, _mode in schedule] == list(range(4))


def _scenario_runs(case_ids=(0, 1)):
    mode_runtime = {
        "off": 1.0,
        "replacement": 0.9,
        "additive": 0.8,
        "additive_first_pass": 0.7,
    }
    runs = []
    for mode in TOURNAMENT.MODES:
        for case_id in case_ids:
            quality = {
                "test_id": case_id,
                "block_count": 21 + case_id,
                "is_feasible": True,
                "hpwl_gap": -0.5 if case_id == 0 else 0.2,
                "area_gap": 0.1,
                "violations_relative": 0.05,
                "cost": 123.0,
                "positions_sha256": "a" * 64,
            }
            for delta in (-0.1, 0.1):
                runs.append({
                    "mode": mode,
                    "case_id": case_id,
                    "runtime_seconds": mode_runtime[mode] + delta,
                    "quality": quality,
                })
    artifacts = [
        {"test_id": case_id, "block_count": 21 + case_id}
        for case_id in case_ids
    ]
    return runs, artifacts


def test_runtime_frontier_uses_exact_official_formula_medians_and_partial_name():
    runs, artifacts = _scenario_runs()
    result = TOURNAMENT._runtime_scenario_frontier(runs, artifacts)
    assert result["panel"]["kind"] == "partial_report_only_panel"
    aggregate = result["panel"]["aggregate_name"]
    assert aggregate == "partial_panel_exp_n_over_12_weighted_cost_not_official_total_score"
    assert [row["field_median_runtime_seconds"] for row in result["scenarios"]] == [0.25, 0.5, 1.0, 2.0, 3.0]
    case = result["per_mode_case"]["off"]["0"]
    assert case["median_runtime_seconds"] == pytest.approx(1.0)
    expected = min(
        (1 + 0.5 * 0.1) * math.exp(2 * 0.05)
        * max(0.7, (1.0 / 0.5) ** 0.3),
        10 - 1e-6,
    )
    assert case["cost_by_field_median"]["0.5_seconds"] == pytest.approx(expected)
    for scenario in result["scenarios"]:
        scores = scenario[aggregate]
        assert scenario["delta_vs_off"]["off"] == 0.0
        assert scenario["delta_vs_replacement"]["replacement"] == 0.0
        assert "additive_first_pass" in scenario["winners"]
        assert scores["additive_first_pass"] <= scores["additive"] <= scores["replacement"] <= scores["off"]
    assert result["nondominated_modes"] == ["additive_first_pass"]
    assert result["all_scenario_winner_intersection"] == ["additive_first_pass"]


def test_official_scenario_cost_clamps_negative_gaps_and_caps_feasible_cost():
    base = {
        "is_feasible": True,
        "hpwl_gap": -100.0,
        "area_gap": -100.0,
        "violations_relative": 0.0,
    }
    assert TOURNAMENT._official_scenario_cost(base, 0.01, 3.0) == pytest.approx(0.7)
    huge = dict(base, hpwl_gap=100.0, area_gap=100.0, violations_relative=100.0)
    assert TOURNAMENT._official_scenario_cost(huge, 10.0, 0.25) == TOURNAMENT.FEASIBLE_COST_CAP


def test_complete_100_case_frontier_uses_official_total_score_name():
    runs, artifacts = _scenario_runs(tuple(range(100)))
    result = TOURNAMENT._runtime_scenario_frontier(runs, artifacts, (1.0,))
    assert result["panel"]["kind"] == "official_complete_100_case_panel"
    assert result["panel"]["aggregate_name"] == "official_100_case_total_score"


def test_post_timing_binding_mutation_is_rejected():
    before = {"evaluator": {"sha256": "a" * 64}, "packages": {"off": "b" * 64}}
    after = copy.deepcopy(before)
    TOURNAMENT._require_unchanged_bindings(before, after)
    after["packages"]["off"] = "c" * 64
    with pytest.raises(ValueError, match="changed between pre- and post-run rehash"):
        TOURNAMENT._require_unchanged_bindings(before, after)


def test_fake_evaluator_executes_exact_wrapper_command_contract(tmp_path):
    wrapper = tmp_path / "op_wrapper.py"
    wrapper.write_text("print('bound')\n")
    data = tmp_path / "data"
    data.mkdir()
    output = tmp_path / "result.json"
    evaluator = tmp_path / "fake_evaluator.py"
    evaluator.write_text(
        "import argparse, json, pathlib, subprocess, sys\n"
        "p=argparse.ArgumentParser(); p.add_argument('--evaluate'); "
        "p.add_argument('--data-path'); p.add_argument('--test-id', type=int); "
        "p.add_argument('--output'); a=p.parse_args()\n"
        "assert subprocess.check_output([sys.executable,a.evaluate],text=True) == 'bound\\n'\n"
        "assert pathlib.Path(a.data_path).is_dir() and a.test_id == 7\n"
        "n=28; row={'test_id':7,'block_count':n,'is_feasible':True,"
        "'hpwl_gap':0.1,'area_gap':0.2,'violations_relative':0.0,"
        "'runtime_seconds':0.25,'cost':1.0,'positions':[[0,0,1,1]]*n,'error':None}\n"
        "json.dump({'test_results':[row], 'summary':{'num_tests':1,'num_feasible':1}}, open(a.output,'w'))\n"
    )
    command = TOURNAMENT._official_evaluator_command(
        evaluator, wrapper, data, 7, output
    )
    completed = TOURNAMENT._run_checked(
        command,
        cwd=tmp_path,
        env=TOURNAMENT._sanitized_timing_environment(dict(os.environ)),
    )
    assert completed.returncode == 0
    validated = TOURNAMENT._validate_official_result(output, 7, 28)
    assert validated["quality"]["block_count"] == 28
