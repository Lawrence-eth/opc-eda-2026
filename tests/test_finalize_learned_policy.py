import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "finalize_learned_policy.py"
    spec = importlib.util.spec_from_file_location("finalize_learned_policy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FINAL = _load_module()
COMMIT = "a" * 40


def _digest(label):
    return hashlib.sha256(label.encode()).hexdigest()


def _components():
    return {
        name: hashlib.sha256((ROOT / "contest_solution" / name).read_bytes()).hexdigest()
        for name in FINAL._expected_components()
    }


def _descriptor(domain, fold):
    return {
        "path": f"{domain}-{fold}.json",
        "sha256": _digest(f"panel-{domain}-{fold}"),
        "fold": fold,
        "domain": domain,
        "manifest_sha256": _digest(f"manifest-{domain}"),
        "case_count": 105,
        "source_file_count": 105,
    }


def _derivation():
    selected = {
        str(size): {str(slot): 0 for slot in FINAL.SLOTS}
        for size in FINAL.HEAVY_SIZES
    }
    nonpreserved = {
        str(size): {str(slot): 1 for slot in FINAL.SLOTS}
        for size in FINAL.HEAVY_SIZES
    }
    derived = {"100": 1.0, "101": 0.9, "102": 1.2}
    for size, slot in derived.items():
        nonpreserved[size][str(slot)] = 0
    replacement = {"100": 1.0, "102": 1.2}
    development_abstain = list(range(103, 121))
    confirmed_abstain = [101, *development_abstain]
    return {
        "schema_version": 2,
        "mode": FINAL.DERIVATION_MODE,
        "provenance": {
            "derivation_harness": {
                "path": "scripts/derive_redundant_slot_map.py",
                "sha256": hashlib.sha256(FINAL.DERIVATION_HARNESS.read_bytes()).hexdigest(),
            }
        },
        "contract": {
            "derivation_folds": [0, 1, 2],
            "derivation_domains": ["mib_input_compatible", "raw_hash"],
            "confirmation_fold": 3,
            "confirmation_policy": FINAL.PUBLIC_POLICY,
            "uses_learned_outputs": False,
            "uses_golden_costs": False,
            "uses_public_cases": False,
            "invariant": (
                "observed removal preserves packed-byte final positions and "
                "inference-visible metrics on every derivation case"
            ),
            "comparison_stage": "complete_deployed_solver_output",
            "removal_method": "full solver rerun for every paid standard slot",
            "tie_rule": "ascending(total_selected_count, numeric_width_factor)",
            "required_development_support_per_size": 30,
        },
        "solver_binding": {
            "commit": COMMIT,
            "components": _components(),
            "audit_harness_sha256": hashlib.sha256(
                FINAL.SLOT_AUDIT_HARNESS.read_bytes()
            ).hexdigest(),
        },
        "development_artifacts": [
            _descriptor(domain, fold)
            for domain in ("mib_input_compatible", "raw_hash")
            for fold in (0, 1, 2)
        ],
        "confirmation_artifacts": [
            _descriptor(domain, 3)
            for domain in ("mib_input_compatible", "raw_hash")
        ],
        "total_selected_counts": {str(slot): 0 for slot in FINAL.SLOTS},
        "tie_preference": list(FINAL.SLOTS),
        "selected_counts_by_size": selected,
        "nonpreserved_counts_by_size": nonpreserved,
        "derivation_case_counts_by_size": {
            str(size): 30 for size in FINAL.HEAVY_SIZES
        },
        "derivation_unique_source_counts_by_size": {
            str(size): 15 for size in FINAL.HEAVY_SIZES
        },
        "derived_replacement_wf_by_size": derived,
        "development_abstain_sizes": development_abstain,
        "confirmation_rejections": [
            {
                "block_count": 101,
                "derived_width_factor": 0.9,
                "failures": [
                    {
                        "domain": "raw_hash",
                        "case_id": "source.th#7",
                        "final_positions_equal": False,
                        "visible_metrics_equal": True,
                    }
                ],
            }
        ],
        "confirmation_rejected_sizes": [101],
        "replacement_wf_by_size": replacement,
        "abstain_sizes": confirmed_abstain,
    }


def _metrics(hpwl=100.0):
    return {
        "feasible": True,
        "hpwl": hpwl,
        "area": 200.0,
        "soft_violations": 1,
    }


def _case(test_id, size, slot, *, preserved):
    control = _digest(f"control-{size}")
    removed = control if preserved else _digest(f"removed-{size}")
    return {
        "test_id": test_id,
        "block_count": size,
        "removed_width_factor": slot,
        "control_positions_sha256": control,
        "removed_positions_sha256": removed,
        "final_positions_equal": preserved,
        "control_visible_metrics": _metrics(),
        "removed_visible_metrics": _metrics(),
        "visible_metrics_equal": True,
        "final_preserved": preserved,
    }


def _audit(slot_sha):
    return {
        "schema_version": 1,
        "mode": FINAL.PUBLIC_AUDIT_MODE,
        "config": {
            "solver_dir": "contest_solution",
            "slot_map_sha256": slot_sha,
            "uses_golden_costs": False,
            "reads_stored_golden_metrics": False,
            "computes_golden_hpwl_or_area": False,
            "golden_geometry_use": FINAL.GOLDEN_GEOMETRY_USE,
            "policy": FINAL.PUBLIC_POLICY,
        },
        "provenance": {
            "harness_sha256": hashlib.sha256(
                FINAL.PUBLIC_AUDIT_HARNESS.read_bytes()
            ).hexdigest(),
            "solver_components": _components(),
            "solver_git": {
                "commit": COMMIT,
                "dirty": False,
                "tracked_dirty": False,
                "has_untracked": False,
            },
        },
        "summary": {
            "mapped_cases": 2,
            "preserved_cases": 1,
            "rejected_sizes": [102],
        },
        "cases": [
            _case(79, 100, 1.0, preserved=True),
            _case(81, 102, 1.2, preserved=False),
        ],
    }


def _write_inputs(tmp_path, derivation=None, mutate_audit=None):
    derivation = _derivation() if derivation is None else derivation
    slot_path = tmp_path / "derived.json"
    slot_path.write_text(json.dumps(derivation, indent=2) + "\n")
    slot_sha = hashlib.sha256(slot_path.read_bytes()).hexdigest()
    audit = _audit(slot_sha)
    if mutate_audit is not None:
        mutate_audit(audit)
    audit_path = tmp_path / "public.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    return slot_path, audit_path, derivation, audit


def test_finalizer_is_provenance_bound_and_public_is_rejection_only(tmp_path):
    slot_path, audit_path, derivation, _audit_payload = _write_inputs(tmp_path)
    result = FINAL.finalize_policy(slot_path, audit_path)

    assert result["schema_version"] == 2
    assert result["evidence_status"] == FINAL.EVIDENCE_STATUS
    assert result["public_rejected_sizes"] == [102]
    assert result["replacement_wf_by_size"] == {"100": 1.0}
    assert result["abstain_sizes"] == list(range(101, 121))
    assert result["public_audit_summary"] == {
        "mapped_cases": 2,
        "mapped_sizes": [100, 102],
        "preserved_cases": 1,
    }
    assert result["derived_replacement_wf_by_size"] == derivation[
        "derived_replacement_wf_by_size"
    ]
    assert result["provenance"]["input_artifacts"]["derivation"]["sha256"] == (
        hashlib.sha256(slot_path.read_bytes()).hexdigest()
    )
    assert result["provenance"]["input_artifacts"]["public_audit"]["sha256"] == (
        hashlib.sha256(audit_path.read_bytes()).hexdigest()
    )
    assert result["provenance"]["finalizer"]["sha256"] == hashlib.sha256(
        FINAL.FINALIZER_PATH.read_bytes()
    ).hexdigest()
    assert result["contract"]["public_validation"] == {
        "uses_public_cases": True,
        "uses_public_golden_costs": False,
        "reads_stored_golden_metrics": False,
        "computes_golden_hpwl_or_area": False,
        "golden_geometry_use": FINAL.GOLDEN_GEOMETRY_USE,
        "policy": FINAL.PUBLIC_POLICY,
        "required_cases_per_mapped_size": 1,
    }
    assert result["provenance"]["solver_snapshots"] == {
        "derivation_commit": COMMIT,
        "public_audit_commit": COMMIT,
        "live_components_identical": True,
    }


def test_finalizer_allows_clean_nonlive_commit_drift(tmp_path):
    slot_path, audit_path, _derivation_payload, _audit_payload = _write_inputs(
        tmp_path,
        mutate_audit=lambda audit: audit["provenance"]["solver_git"].__setitem__(
            "commit", "b" * 40
        ),
    )

    result = FINAL.finalize_policy(slot_path, audit_path)

    assert result["provenance"]["solver_snapshots"] == {
        "derivation_commit": COMMIT,
        "public_audit_commit": "b" * 40,
        "live_components_identical": True,
    }


def test_finalizer_rejects_public_audit_for_different_derivation_bytes(tmp_path):
    slot_path, audit_path, _derivation_payload, _audit_payload = _write_inputs(tmp_path)
    payload = json.loads(slot_path.read_text())
    slot_path.write_text(json.dumps(payload, separators=(",", ":")))
    with pytest.raises(ValueError, match="exact derivation input bytes"):
        FINAL.finalize_policy(slot_path, audit_path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda audit: audit["provenance"]["solver_git"].__setitem__(
                "dirty", True
            ),
            "not clean",
        ),
        (
            lambda audit: audit["provenance"]["solver_components"].__setitem__(
                next(iter(audit["provenance"]["solver_components"])), _digest("drift")
            ),
            "different solver components",
        ),
        (
            lambda audit: audit["cases"][1].__setitem__("removed_width_factor", 1.1),
            "retuned",
        ),
        (
            lambda audit: audit["cases"].pop(),
            "one case for every confirmed mapping",
        ),
        (
            lambda audit: audit["cases"][1].__setitem__("block_count", 100),
            "duplicate",
        ),
        (
            lambda audit: audit["summary"].__setitem__("mapped_cases", 1),
            "mapped case count",
        ),
        (
            lambda audit: audit["summary"].__setitem__("rejected_sizes", []),
            "rejected-size summary",
        ),
        (
            lambda audit: audit["cases"][1].__setitem__("final_preserved", True),
            "final verdict",
        ),
    ],
)
def test_finalizer_rejects_public_audit_drift(tmp_path, mutate, message):
    slot_path, audit_path, _derivation_payload, _audit_payload = _write_inputs(
        tmp_path, mutate_audit=mutate
    )
    with pytest.raises(ValueError, match=message):
        FINAL.finalize_policy(slot_path, audit_path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["replacement_wf_by_size"].__setitem__("100", 0.8),
            "remove, never add or retune",
        ),
        (
            lambda payload: payload["replacement_wf_by_size"].__setitem__("103", 1.0),
            "remove, never add or retune",
        ),
        (
            lambda payload: payload.__setitem__("abstain_sizes", list(range(102, 121))),
            "complement",
        ),
        (
            lambda payload: payload["nonpreserved_counts_by_size"]["100"].__setitem__(
                "0.8", 0
            ),
            "recorded counts",
        ),
        (
            lambda payload: payload["solver_binding"]["components"].pop(
                next(iter(payload["solver_binding"]["components"]))
            ),
            "incomplete live component",
        ),
    ],
)
def test_finalizer_rejects_inconsistent_derivation(tmp_path, mutate, message):
    derivation = _derivation()
    mutate(derivation)
    slot_path, audit_path, _derivation_payload, _audit_payload = _write_inputs(
        tmp_path, derivation=derivation
    )
    with pytest.raises(ValueError, match=message):
        FINAL.finalize_policy(slot_path, audit_path)


def test_json_loader_rejects_duplicate_keys_and_nonfinite_numbers(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version": 2, "schema_version": 2}')
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        FINAL._load_object(duplicate, "test")

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value": NaN}')
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        FINAL._load_object(nonfinite, "test")


def test_atomic_writer_refuses_overwrite_without_permission(tmp_path):
    output = tmp_path / "final.json"
    FINAL._write_json_atomic(output, {"value": 1}, overwrite=False)
    with pytest.raises(FileExistsError, match="--overwrite"):
        FINAL._write_json_atomic(output, {"value": 2}, overwrite=False)
    assert json.loads(output.read_text()) == {"value": 1}
    FINAL._write_json_atomic(output, {"value": 3}, overwrite=True)
    assert json.loads(output.read_text()) == {"value": 3}


def test_validation_does_not_mutate_either_input_object(tmp_path):
    slot_path, audit_path, derivation, audit = _write_inputs(tmp_path)
    derivation_before = copy.deepcopy(derivation)
    audit_before = copy.deepcopy(audit)
    FINAL.finalize_policy(slot_path, audit_path)
    assert derivation == derivation_before
    assert audit == audit_before
