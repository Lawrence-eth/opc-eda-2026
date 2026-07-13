import subprocess
import sys
from pathlib import Path

from scripts.solver_components import LIVE_SOLVER_COMPONENTS, SOLVER_ENTRYPOINT


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
