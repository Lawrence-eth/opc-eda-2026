import importlib.util
import json
from pathlib import Path

import torch
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_holdout_folds", ROOT / "scripts" / "build_holdout_folds.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

EVAL_SPEC = importlib.util.spec_from_file_location(
    "evaluate_training_holdout", ROOT / "scripts" / "evaluate_training_holdout.py"
)
EVAL_MODULE = importlib.util.module_from_spec(EVAL_SPEC)
EVAL_SPEC.loader.exec_module(EVAL_MODULE)


def test_holdout_harness_loads_the_pinned_official_evaluator_by_path():
    expected = (
        ROOT
        / "external"
        / "FloorSet"
        / "iccad2026contest"
        / "iccad2026_evaluate.py"
    ).resolve()
    assert Path(EVAL_MODULE.evaluate_solution.__code__.co_filename).resolve() == expected


def test_file_fold_is_stable_and_seeded():
    first = MODULE._fold_for_file("worker_1/layouts_42.th", 7, 5)
    assert first == MODULE._fold_for_file("worker_1/layouts_42.th", 7, 5)
    assert 0 <= first < 5
    assert any(
        MODULE._fold_for_file("worker_1/layouts_42.th", seed, 5) != first
        for seed in range(8, 32)
    )


def test_hash_offset_is_stable_bounded_and_seeded():
    first = MODULE._hash_offset("floorset_lite/worker_1/layouts_42.th", 7, 112)
    assert first == MODULE._hash_offset(
        "floorset_lite/worker_1/layouts_42.th", 7, 112
    )
    assert 0 <= first < 112
    assert any(
        MODULE._hash_offset("floorset_lite/worker_1/layouts_42.th", seed, 112)
        != first
        for seed in range(8, 32)
    )


def test_mib_input_compatibility_uses_area_intervals_and_hard_targets_only():
    constraints = torch.tensor([[0, 0, 1, 0, 0], [0, 0, 1, 0, 0]], dtype=torch.float)
    clean_fp = torch.tensor([[2.0, 5.0, 0, 0], [2.0, 5.0, 2, 0]])
    assert MODULE._mib_is_input_compatible(
        torch.tensor([10.0, 10.15]), constraints, clean_fp, 2
    )
    assert not MODULE._mib_is_input_compatible(
        torch.tensor([10.0, 12.0]), constraints, clean_fp, 2
    )
    # Free-block golden shapes are labels, not admission features.
    corrupt_fp = torch.tensor([[2.0, 5.0, 0, 0], [1.0, 10.0, 2, 0]])
    assert MODULE._mib_is_input_compatible(
        torch.tensor([10.0, 10.0]), constraints, corrupt_fp, 2
    )
    hard_constraints = torch.tensor(
        [[1, 0, 1, 0, 0], [0, 1, 1, 0, 0]], dtype=torch.float
    )
    assert not MODULE._mib_is_input_compatible(
        torch.tensor([10.0, 10.0]), hard_constraints, corrupt_fp, 2
    )

    # Fixed/preplaced blocks are excluded from the official soft-block area
    # tolerance.  Their input dimensions, not their area_target entries,
    # constrain a common MIB shape.
    one_hard = torch.tensor(
        [[1, 0, 1, 0, 0], [0, 0, 1, 0, 0]], dtype=torch.float
    )
    hard_shape = torch.tensor([[2.0, 2.0, 0, 0], [9.0, 9.0, 0, 0]])
    assert MODULE._mib_is_input_compatible(
        torch.tensor([100.0, 4.0]), one_hard, hard_shape, 2
    )
    assert not MODULE._mib_is_input_compatible(
        torch.tensor([100.0, 9.0]), one_hard, hard_shape, 2
    )
    all_hard = torch.tensor(
        [[1, 0, 1, 0, 0], [0, 1, 1, 0, 0]], dtype=torch.float
    )
    same_hard_shapes = torch.tensor([[2.0, 2.0, 0, 0], [2.0, 2.0, 3, 0]])
    assert MODULE._mib_is_input_compatible(
        torch.tensor([100.0, 200.0]), all_hard, same_hard_shapes, 2
    )


def test_fold_manifest_indices_select_exact_fold(tmp_path):
    path = tmp_path / "folds.json"
    path.write_text(
        json.dumps(
            {
                "manifests": [
                    {"fold": 0, "cases": [{"sample_index": 11}]},
                    {
                        "fold": 1,
                        "cases": [{"sample_index": 20}, {"sample_index": 21}],
                    },
                ]
            }
        )
    )
    assert EVAL_MODULE._manifest_indices(path, 1) == [20, 21]


def test_manifest_resolution_binds_complete_input_and_scoring_labels(
    tmp_path, monkeypatch
):
    data_root = tmp_path / "data"
    source = data_root / "floorset_lite" / "worker_0" / "layouts_0.th"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"inventory identity")
    relative = source.relative_to(data_root).as_posix()
    sample = {
        "input": (
            torch.tensor([4.0, 9.0]),
            torch.tensor([[0.0, 1.0, 1.0]]),
            torch.empty((0, 3)),
            torch.empty((0, 2)),
            torch.tensor([[1.0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
        ),
        "label": (
            None,
            torch.tensor([[2.0, 2.0, 0.0, 0.0], [3.0, 3.0, 2.0, 0.0]]),
            None,
        ),
    }

    class Dataset:
        all_files = [str(source)]
        layouts_per_file = 1
        cached_file_idx = -1

        def __getitem__(self, index):
            assert index == 0
            return sample

    dataset = Dataset()
    fold = MODULE._fold_for_file(relative, 17, 5)
    case = MODULE._case_metadata(sample, 999, relative, 0, 2)
    manifest = {
        "schema_version": 3,
        "split_unit": "source_file",
        "num_folds": 5,
        "dataset": {
            "official_floorset_commit": "abc123",
            "layouts_per_file": 1,
            "source_file_count": 1,
            "source_inventory_sha256": MODULE._inventory_sha256(dataset, data_root),
        },
        "generation": {
            "min_blocks": 2,
            "max_blocks": 2,
            "num_folds": 5,
            "per_size": 1,
            "seed": 17,
        },
        "manifests": [
            {
                "fold": fold,
                "case_count": 1,
                "source_file_count": 1,
                "cases": [case],
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(EVAL_MODULE, "_git_commit", lambda _path: "abc123")
    selected, provenance = EVAL_MODULE._resolve_manifest_cases(
        dataset, data_root, path, fold
    )
    assert selected[0][0] == 0  # stored diagnostic sample_index=999 is ignored
    assert selected[0][2]["case_id"] == f"{relative}#0"
    assert provenance["resolved_inventory_sha256"] == manifest["dataset"][
        "source_inventory_sha256"
    ]
    assert provenance["resolved_official_floorset_commit"] == "abc123"

    monkeypatch.setattr(EVAL_MODULE, "_git_commit", lambda _path: "different")
    with pytest.raises(ValueError, match="FloorSet commit does not match"):
        EVAL_MODULE._resolve_manifest_cases(dataset, data_root, path, fold)
    monkeypatch.setattr(EVAL_MODULE, "_git_commit", lambda _path: "abc123")

    original_input = sample["input"]
    sample["input"] = (torch.tensor([4.0, 10.0]),) + sample["input"][1:]
    with pytest.raises(ValueError, match="input digest changed"):
        EVAL_MODULE._resolve_manifest_cases(dataset, data_root, path, fold)
    sample["input"] = original_input

    fp_sol = sample["label"][1]
    original_width = float(fp_sol[0, 0])
    fp_sol[0, 0] = original_width + 0.5
    with pytest.raises(ValueError, match="optimizer_target_sha256"):
        EVAL_MODULE._resolve_manifest_cases(dataset, data_root, path, fold)
    fp_sol[0, 0] = original_width

    original_x = float(fp_sol[1, 2])
    fp_sol[1, 2] = original_x + 0.5
    with pytest.raises(ValueError, match="scoring_label_sha256"):
        EVAL_MODULE._resolve_manifest_cases(dataset, data_root, path, fold)


def test_solver_output_validation_requires_exact_finite_positive_rectangles():
    assert EVAL_MODULE._validate_solver_positions(
        [(0, 1, 2, 3), torch.tensor([4.0, 5.0, 6.0, 7.0])], 2
    ) == [(0.0, 1.0, 2.0, 3.0), (4.0, 5.0, 6.0, 7.0)]

    invalid = [
        (None, 1),
        ([(0, 0, 1, 1)], 2),
        ([(0, 0, 1)], 1),
        ([(float("nan"), 0, 1, 1)], 1),
        ([(0, float("inf"), 1, 1)], 1),
        ([(0, 0, 0, 1)], 1),
        ([(0, 0, 1, -1)], 1),
        (["not-a-rectangle"], 1),
    ]
    for positions, block_count in invalid:
        with pytest.raises(ValueError):
            EVAL_MODULE._validate_solver_positions(positions, block_count)
