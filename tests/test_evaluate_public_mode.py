from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import evaluate_public_mode as public_mode
from scripts.solver_components import LIVE_SOLVER_COMPONENTS


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class _FakeRow:
    test_id: int
    block_count: int = 1
    is_feasible: bool = True
    hpwl_gap: float = 0.0
    area_gap: float = 0.0
    violations_relative: float = 0.0
    runtime_seconds: float = 0.01
    cost: float = 1.0
    positions: list | None = None
    error: str | None = None


@dataclass
class _FakeResult:
    submission_name: str
    timestamp: str
    total_score: float
    test_results: list
    summary: dict


class _FakeOptimizer:
    def __init__(self):
        self._learned_order_mode = "replacement"
        self._learned_order_enabled = True
        self._baselines_by_n = None


def _fake_evaluator_module(*, mutate=None):
    class FakeEvaluator:
        configured_optimizer = None

        def __init__(self, data_path, verbose):
            self.data_path = data_path
            self.verbose = verbose

        def _load_optimizer(self, path):
            return _FakeOptimizer()

        def evaluate(self, optimizer_path, test_ids=None):
            optimizer = self._load_optimizer(optimizer_path)
            type(self).configured_optimizer = optimizer
            if mutate is not None:
                mutate()
            ids = list(public_mode.PUBLIC_CASE_IDS) if test_ids is None else test_ids
            rows = [
                _FakeRow(
                    test_id=test_id,
                    positions=[[float(test_id), 0.0, 1.0, 1.0]],
                )
                for test_id in ids
            ]
            return _FakeResult(
                submission_name="official",
                timestamp="2026-07-14T00:00:00",
                total_score=1.0,
                test_results=rows,
                summary={
                    "num_tests": len(rows),
                    "num_feasible": len(rows),
                    "avg_cost": 1.0,
                    "avg_runtime": 0.01,
                },
            )

    return SimpleNamespace(
        ContestEvaluator=FakeEvaluator,
        compute_total_score=lambda costs, _blocks: sum(costs) / len(costs),
    )


def _filesystem_fixture(tmp_path: Path):
    solver_dir = tmp_path / "solver"
    solver_dir.mkdir()
    for component in LIVE_SOLVER_COMPONENTS:
        (solver_dir / component).write_text(
            f"# fixture {component}\n", encoding="utf-8"
        )
    data_root = tmp_path / "FloorSet"
    evaluator = data_root / "iccad2026contest" / "iccad2026_evaluate.py"
    evaluator.parent.mkdir(parents=True)
    evaluator.write_text("# verified evaluator fixture\n", encoding="utf-8")
    official_sources = tmp_path / "official_sources.json"
    official_sources.write_text("{}\n", encoding="utf-8")
    return solver_dir, data_root, official_sources


def _allow_official_verification(monkeypatch):
    monkeypatch.setattr(
        public_mode,
        "verify_official_sources",
        lambda *args, **kwargs: (True, [], ["fixture verification"]),
    )


def test_normalize_test_ids_is_fail_closed():
    assert public_mode._normalize_test_ids(None) is None
    assert public_mode._normalize_test_ids([99, 0]) == [99, 0]

    with pytest.raises(ValueError, match="unique"):
        public_mode._normalize_test_ids([4, 4])
    with pytest.raises(ValueError, match="0..99"):
        public_mode._normalize_test_ids([-1])
    with pytest.raises(ValueError, match="0..99"):
        public_mode._normalize_test_ids([100])


def test_configured_loader_applies_only_the_requested_mode(tmp_path):
    entrypoint = tmp_path / "my_optimizer.py"
    entrypoint.write_text("# fixture\n", encoding="utf-8")

    class Evaluator:
        def _load_optimizer(self, _path):
            return _FakeOptimizer()

    evaluator = Evaluator()
    loaded = public_mode._configure_optimizer_loader(
        evaluator, entrypoint, "off"
    )
    optimizer = evaluator._load_optimizer(str(entrypoint))

    assert len(loaded) == 1
    assert optimizer._learned_order_mode == "off"
    assert optimizer._learned_order_enabled is False
    assert optimizer._baselines_by_n == {}
    with pytest.raises(RuntimeError, match="multiple optimizers"):
        evaluator._load_optimizer(str(entrypoint))


def test_configured_loader_rejects_wrong_path_and_missing_switches(tmp_path):
    entrypoint = tmp_path / "my_optimizer.py"
    entrypoint.write_text("# fixture\n", encoding="utf-8")

    class Evaluator:
        def __init__(self, optimizer):
            self.optimizer = optimizer

        def _load_optimizer(self, _path):
            return self.optimizer

    wrong_path_evaluator = Evaluator(_FakeOptimizer())
    public_mode._configure_optimizer_loader(
        wrong_path_evaluator, entrypoint, "replacement"
    )
    with pytest.raises(RuntimeError, match="unexpected optimizer"):
        wrong_path_evaluator._load_optimizer(str(tmp_path / "other.py"))

    missing_switch_evaluator = Evaluator(object())
    public_mode._configure_optimizer_loader(
        missing_switch_evaluator, entrypoint, "replacement"
    )
    with pytest.raises(RuntimeError, match="research switches"):
        missing_switch_evaluator._load_optimizer(str(entrypoint))


def test_evaluate_public_mode_records_complete_provenance(tmp_path, monkeypatch):
    solver_dir, data_root, official_sources = _filesystem_fixture(tmp_path)
    _allow_official_verification(monkeypatch)
    fake_module = _fake_evaluator_module()
    monkeypatch.setattr(
        public_mode, "_load_official_evaluator", lambda _path: fake_module
    )

    payload = public_mode.evaluate_public_mode(
        data_root=data_root,
        solver_dir=solver_dir,
        official_sources=official_sources,
        learned_mode="additive",
        test_ids=[7],
    )

    config = payload["research_config"]
    optimizer = fake_module.ContestEvaluator.configured_optimizer
    assert payload["submission_name"] == "my_optimizer-additive"
    assert config["learned_mode"] == "additive"
    assert config["test_ids"] == [7]
    assert config["requested_test_ids"] == [7]
    assert config["evaluated_test_ids"] == [7]
    assert config["runtime_factor_mode"] == "neutral_rf_1"
    assert config["uses_environment_override"] is False
    assert config["official_verification_notes"] == ["fixture verification"]
    assert set(config["solver_components"]) == set(LIVE_SOLVER_COMPONENTS)
    assert len(config["official_evaluator"]["sha256"]) == 64
    assert optimizer._learned_order_mode == "additive"
    assert optimizer._learned_order_enabled is True
    assert optimizer._baselines_by_n == {}


def test_evaluate_public_mode_rejects_live_source_mutation(tmp_path, monkeypatch):
    solver_dir, data_root, official_sources = _filesystem_fixture(tmp_path)
    _allow_official_verification(monkeypatch)

    def mutate():
        (solver_dir / "dissect.py").write_text("# mutated\n", encoding="utf-8")

    fake_module = _fake_evaluator_module(mutate=mutate)
    monkeypatch.setattr(
        public_mode, "_load_official_evaluator", lambda _path: fake_module
    )

    with pytest.raises(RuntimeError, match="changed during public evaluation"):
        public_mode.evaluate_public_mode(
            data_root=data_root,
            solver_dir=solver_dir,
            official_sources=official_sources,
            learned_mode="replacement",
            test_ids=[99],
        )


def test_evaluate_public_mode_rejects_failed_official_pin(tmp_path, monkeypatch):
    solver_dir, data_root, official_sources = _filesystem_fixture(tmp_path)
    monkeypatch.setattr(
        public_mode,
        "verify_official_sources",
        lambda *args, **kwargs: (False, ["wrong commit"], []),
    )
    monkeypatch.setattr(
        public_mode,
        "_load_official_evaluator",
        lambda _path: _fake_evaluator_module(),
    )

    with pytest.raises(RuntimeError, match="wrong commit"):
        public_mode.evaluate_public_mode(
            data_root=data_root,
            solver_dir=solver_dir,
            official_sources=official_sources,
            learned_mode="replacement",
            test_ids=[99],
        )


def test_result_validation_rejects_case_and_score_drift():
    module = _fake_evaluator_module()
    evaluator = module.ContestEvaluator("unused", False)
    entrypoint = Path("/tmp/my_optimizer.py")
    evaluator._load_optimizer = lambda _path: _FakeOptimizer()
    result = evaluator.evaluate(str(entrypoint), test_ids=[3])
    payload = public_mode._result_payload(result)

    payload["test_results"][0]["test_id"] = 4
    with pytest.raises(ValueError, match="case IDs"):
        public_mode._validate_result_payload(payload, [3], module.compute_total_score)

    payload = public_mode._result_payload(result)
    payload["total_score"] = 2.0
    with pytest.raises(ValueError, match="total_score"):
        public_mode._validate_result_payload(payload, [3], module.compute_total_score)


def test_result_validation_requires_saved_finite_positions():
    module = _fake_evaluator_module()
    evaluator = module.ContestEvaluator("unused", False)
    evaluator._load_optimizer = lambda _path: _FakeOptimizer()
    payload = public_mode._result_payload(
        evaluator.evaluate("/tmp/my_optimizer.py", test_ids=[3])
    )
    payload["test_results"][0]["positions"] = [[float("nan"), 0.0, 1.0, 1.0]]

    with pytest.raises(ValueError, match="valid saved positions"):
        public_mode._validate_result_payload(payload, [3], module.compute_total_score)


def test_result_validation_preserves_well_formed_official_error_rows():
    module = _fake_evaluator_module()
    evaluator = module.ContestEvaluator("unused", False)
    evaluator._load_optimizer = lambda _path: _FakeOptimizer()
    payload = public_mode._result_payload(
        evaluator.evaluate("/tmp/my_optimizer.py", test_ids=[3])
    )
    row = payload["test_results"][0]
    row.update(
        block_count=0,
        is_feasible=False,
        cost=10.0,
        positions=None,
        error="solver failed",
    )
    payload["summary"].update(num_feasible=0, avg_cost=10.0)
    payload["total_score"] = 10.0

    assert public_mode._validate_result_payload(
        payload, [3], module.compute_total_score
    ) == [3]

    row["is_feasible"] = True
    with pytest.raises(ValueError, match="inconsistent error metadata"):
        public_mode._validate_result_payload(payload, [3], module.compute_total_score)


def test_atomic_writer_refuses_overwrite_without_explicit_permission(tmp_path):
    output = tmp_path / "result.json"
    public_mode._write_json_atomic(output, {"value": 1}, overwrite=False)

    with pytest.raises(FileExistsError, match="--overwrite"):
        public_mode._write_json_atomic(output, {"value": 2}, overwrite=False)

    public_mode._write_json_atomic(output, {"value": 3}, overwrite=True)
    assert json.loads(output.read_text(encoding="utf-8")) == {"value": 3}
    assert not list(tmp_path.glob("*.tmp"))


def test_help_does_not_require_loading_the_official_evaluator():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "evaluate_public_mode.py"), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--learned-mode" in completed.stdout
    assert "--overwrite" in completed.stdout
