import subprocess
import sys
from pathlib import Path

import pytest

from scripts.solver_components import (
    LIVE_SOLVER_COMPONENTS,
    SOLVER_ENTRYPOINT,
    validate_live_solver_components,
)


ROOT = Path(__file__).resolve().parents[1]


def test_live_solver_registry_is_unique_complete_and_present():
    assert LIVE_SOLVER_COMPONENTS[0] == SOLVER_ENTRYPOINT
    assert len(LIVE_SOLVER_COMPONENTS) == len(set(LIVE_SOLVER_COMPONENTS))
    assert {
        "learned_order.py",
        "order_model_v5b.py",
        "golden_plus_repair.py",
    }.issubset(LIVE_SOLVER_COMPONENTS)
    assert all(
        (ROOT / "contest_solution" / component).is_file()
        for component in LIVE_SOLVER_COMPONENTS
    )


def test_live_solver_registry_cli_matches_python_contract():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "solver_components.py")],
        check=True,
        capture_output=True,
        text=True,
    )

    assert tuple(completed.stdout.splitlines()) == LIVE_SOLVER_COMPONENTS


@pytest.mark.parametrize(
    "component",
    ("nested/solver.py", "solver.txt", "not-importable.py", ""),
)
def test_live_solver_registry_rejects_invalid_component_names(component):
    with pytest.raises(ValueError, match="component"):
        validate_live_solver_components((SOLVER_ENTRYPOINT, component))


def test_live_solver_registry_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        validate_live_solver_components((SOLVER_ENTRYPOINT, SOLVER_ENTRYPOINT))


def test_live_solver_registry_rejects_package_support_name_collision():
    with pytest.raises(ValueError, match="collides with a package support source"):
        validate_live_solver_components((SOLVER_ENTRYPOINT, "torch.py"))


def test_live_solver_registry_fails_closed_on_missing_source(tmp_path):
    (tmp_path / SOLVER_ENTRYPOINT).write_text("# fixture\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing"):
        validate_live_solver_components(
            (SOLVER_ENTRYPOINT, "missing.py"), source_dir=tmp_path
        )
