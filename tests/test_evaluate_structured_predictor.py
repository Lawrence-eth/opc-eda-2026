import importlib.util
import json
from pathlib import Path
import sys

import pytest

from contest_solution.dual_parent_decoder import DualParentLabels, HorizontalRelation


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_structured_predictor",
    ROOT / "scripts" / "evaluate_structured_predictor.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _labels():
    return DualParentLabels(
        root=0,
        dimensions=((2.0, 2.0), (2.0, 2.0)),
        shape_options=(((2.0, 2.0),), ((2.0, 2.0),)),
        selected_shape_indices=(0, 0),
        horizontal=(HorizontalRelation(0, 1, 0),),
        vertical_supports=(None, None),
    )


def test_label_metrics_require_every_structural_head_for_full_exactness():
    expected = _labels()
    exact = MODULE._labels_equal(expected, expected)
    assert exact["full_exact"]
    assert exact["horizontal_correct"] == 1

    wrong = DualParentLabels(
        root=expected.root,
        dimensions=expected.dimensions,
        shape_options=expected.shape_options,
        selected_shape_indices=expected.selected_shape_indices,
        horizontal=expected.horizontal,
        vertical_supports=(None, 0),
    )
    result = MODULE._labels_equal(wrong, expected)
    assert not result["full_exact"]
    assert result["vertical_correct"] == 1


def test_development_harness_refuses_sealed_manifest_without_reading_it(tmp_path):
    path = tmp_path / "heavy_sealed_v2.json"
    path.write_text("this must not be parsed")
    with pytest.raises(ValueError, match="sealed manifests"):
        MODULE._selected_cases(object(), tmp_path, path, {0})


def test_development_harness_refuses_sealed_role(tmp_path):
    path = tmp_path / "development.json"
    path.write_text(
        json.dumps(
            {
                "split_unit": "source_file",
                "manifests": [{"fold": 0, "role": "final_sealed", "cases": []}],
            }
        )
    )

    class Dataset:
        all_files = []
        layouts_per_file = 112

    with pytest.raises(ValueError, match="sealed manifest roles"):
        MODULE._selected_cases(Dataset(), tmp_path, path, {0})
