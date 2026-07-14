#!/usr/bin/env python3
"""Evaluate one learned-policy mode with the pinned official public scorer.

The production optimizer deliberately ignores environment variables. This
research harness configures one freshly loaded in-memory optimizer, leaving
every live solver source byte-for-byte unchanged. Results retain the official
JSON fields and add a provenance-only ``research_config`` object.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
DEFAULT_OFFICIAL_ROOT = ROOT / "external" / "FloorSet"
DEFAULT_SOLVER_DIR = ROOT / "contest_solution"
DEFAULT_OFFICIAL_SOURCES = ROOT / "docs" / "official_sources.json"
PUBLIC_CASE_IDS = tuple(range(100))
LEARNED_MODES = ("off", "replacement", "additive", "additive_first_pass")

sys.path.insert(0, str(SCRIPTS_DIR))

from check_official_sources import verify_official_sources  # noqa: E402
from solver_components import LIVE_SOLVER_COMPONENTS  # noqa: E402


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _solver_component_hashes(path: Path) -> dict[str, str]:
    if not LIVE_SOLVER_COMPONENTS or len(LIVE_SOLVER_COMPONENTS) != len(
        set(LIVE_SOLVER_COMPONENTS)
    ):
        raise ValueError("live solver component registry is empty or duplicated")
    missing = [name for name in LIVE_SOLVER_COMPONENTS if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"live solver components are missing from {path}: {missing}"
        )
    return {name: _file_sha256(path / name) for name in LIVE_SOLVER_COMPONENTS}


def _git_state(path: Path) -> dict[str, Any]:
    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(path), *arguments],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip())
        return completed.stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        status = git("status", "--porcelain")
        tracked_status = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, RuntimeError):
        return {"commit": None, "dirty": None}
    return {
        "commit": commit,
        "dirty": bool(status),
        "tracked_dirty": bool(tracked_status),
        "has_untracked": any(line.startswith("??") for line in status.splitlines()),
    }


def _normalize_test_ids(values: list[int] | None) -> list[int] | None:
    if values is None:
        return None
    if not values:
        raise ValueError("at least one --test-id is required when selecting cases")
    if len(values) != len(set(values)):
        raise ValueError("--test-id values must be unique")
    invalid = [value for value in values if value not in PUBLIC_CASE_IDS]
    if invalid:
        raise ValueError(
            "--test-id values must be public case IDs in 0..99: "
            + ", ".join(map(str, invalid))
        )
    return list(values)


def _verify_official_checkout(
    data_root: Path,
    official_sources: Path,
) -> list[str]:
    ok, errors, notes = verify_official_sources(
        official_sources,
        root=ROOT,
        floorset_path=data_root,
        materials_dir=None,
        require_floorset=True,
        release_manifest_path=None,
    )
    if not ok:
        raise RuntimeError(
            "official FloorSet verification failed:\n  " + "\n  ".join(errors)
        )
    return notes


def _load_official_evaluator(evaluator_path: Path):
    """Load the verified evaluator by path, independent of cached modules."""

    if not evaluator_path.is_file():
        raise FileNotFoundError(f"official evaluator is missing: {evaluator_path}")
    contest_dir = evaluator_path.parent
    floorset_root = contest_dir.parent
    sys.path[:0] = [str(contest_dir), str(floorset_root)]
    module_name = f"_public_mode_official_{_file_sha256(evaluator_path)[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, evaluator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot create import spec for {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    return module


def _configure_optimizer_loader(
    evaluator: Any,
    entrypoint: Path,
    learned_mode: str,
) -> list[dict[str, Any]]:
    """Install the one-load in-memory research configuration contract."""

    if learned_mode not in LEARNED_MODES:
        raise ValueError(f"unsupported learned mode: {learned_mode!r}")
    official_loader = getattr(evaluator, "_load_optimizer", None)
    if not callable(official_loader):
        raise RuntimeError("official evaluator does not expose _load_optimizer")
    expected_entrypoint = entrypoint.resolve()
    loaded: list[dict[str, Any]] = []

    def load_configured_optimizer(path):
        observed_entrypoint = Path(path).resolve()
        if observed_entrypoint != expected_entrypoint:
            raise RuntimeError(
                "official evaluator requested an unexpected optimizer: "
                f"{observed_entrypoint}"
            )
        if loaded:
            raise RuntimeError("official evaluator attempted to load multiple optimizers")
        optimizer = official_loader(path)
        required = (
            "_learned_order_mode",
            "_learned_order_enabled",
            "_baselines_by_n",
        )
        missing = [name for name in required if not hasattr(optimizer, name)]
        if missing:
            raise RuntimeError(
                "solver does not expose required research switches: "
                + ", ".join(missing)
            )
        record = {
            "optimizer": optimizer,
            "initial_learned_mode": optimizer._learned_order_mode,
            "initial_learned_enabled": bool(optimizer._learned_order_enabled),
            "applied_learned_mode": learned_mode,
            "applied_learned_enabled": learned_mode != "off",
            "forces_empty_local_baselines": True,
        }
        optimizer._learned_order_mode = learned_mode
        optimizer._learned_order_enabled = learned_mode != "off"
        # Never discover repository-local golden baselines in a research run.
        optimizer._baselines_by_n = {}
        loaded.append(record)
        return optimizer

    evaluator._load_optimizer = load_configured_optimizer
    return loaded


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _valid_positions(value: Any, block_count: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == block_count
        and all(
            isinstance(row, (list, tuple))
            and len(row) == 4
            and all(_finite_number(item) for item in row)
            for row in value
        )
    )


def _result_payload(result: Any) -> dict[str, Any]:
    if is_dataclass(result) and not isinstance(result, type):
        payload = asdict(result)
    elif isinstance(result, dict):
        payload = dict(result)
    else:
        raise TypeError("official evaluator result is not a dataclass or object")
    if not isinstance(payload, dict):
        raise TypeError("official evaluator result did not become a JSON object")
    return payload


def _validate_result_payload(
    payload: dict[str, Any],
    requested_test_ids: list[int] | None,
    compute_total_score: Callable[[list[float], list[int]], float],
) -> list[int]:
    expected_ids = (
        list(PUBLIC_CASE_IDS) if requested_test_ids is None else requested_test_ids
    )
    rows = payload.get("test_results")
    if not isinstance(rows, list) or len(rows) != len(expected_ids):
        raise ValueError(
            f"official evaluator returned {len(rows) if isinstance(rows, list) else 'invalid'} "
            f"rows; expected {len(expected_ids)}"
        )
    observed_ids: list[int] = []
    costs: list[float] = []
    block_counts: list[int] = []
    feasible = 0
    for offset, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"official result row {offset} is not an object")
        test_id = row.get("test_id")
        if isinstance(test_id, bool) or not isinstance(test_id, int):
            raise ValueError(f"official result row {offset} has an invalid test_id")
        observed_ids.append(test_id)
        error = row.get("error")
        if error is not None and not isinstance(error, str):
            raise ValueError(f"official result case {test_id} has a malformed error")
        has_error = error is not None
        if has_error and not error:
            raise ValueError(f"official result case {test_id} has an empty error")
        block_count = row.get("block_count")
        if (
            isinstance(block_count, bool)
            or not isinstance(block_count, int)
            or block_count < (0 if has_error else 1)
        ):
            raise ValueError(f"official result case {test_id} has an invalid block_count")
        if not isinstance(row.get("is_feasible"), bool):
            raise ValueError(f"official result case {test_id} has invalid feasibility")
        if has_error and (row["is_feasible"] or block_count != 0):
            raise ValueError(
                f"official result case {test_id} has inconsistent error metadata"
            )
        if row["is_feasible"]:
            feasible += 1
        for field in (
            "hpwl_gap",
            "area_gap",
            "violations_relative",
            "runtime_seconds",
            "cost",
        ):
            if not _finite_number(row.get(field)):
                raise ValueError(f"official result case {test_id} has invalid {field}")
        if float(row["runtime_seconds"]) < 0.0 or float(row["cost"]) < 0.0:
            raise ValueError(f"official result case {test_id} has negative runtime or cost")
        if not has_error and not _valid_positions(row.get("positions"), block_count):
            raise ValueError(f"official result case {test_id} lacks valid saved positions")
        if has_error and row.get("positions") is not None:
            raise ValueError(
                f"official result case {test_id} unexpectedly saved error positions"
            )
        costs.append(float(row["cost"]))
        block_counts.append(block_count)

    if observed_ids != expected_ids:
        raise ValueError(
            f"official evaluator returned case IDs {observed_ids}; expected {expected_ids}"
        )
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("official evaluator result lacks a summary object")
    if summary.get("num_tests") != len(rows):
        raise ValueError("official summary.num_tests does not match result rows")
    if summary.get("num_feasible") != feasible:
        raise ValueError("official summary.num_feasible does not match result rows")
    for field in ("avg_cost", "avg_runtime"):
        if not _finite_number(summary.get(field)):
            raise ValueError(f"official summary.{field} is not finite")
    expected_avg_cost = sum(costs) / len(costs) if costs else 0.0
    if not math.isclose(
        float(summary["avg_cost"]), expected_avg_cost, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("official summary.avg_cost does not match result rows")
    if not _finite_number(payload.get("total_score")):
        raise ValueError("official total_score is not finite")
    reconstructed = float(compute_total_score(costs, block_counts))
    if not math.isclose(
        float(payload["total_score"]), reconstructed, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("official total_score does not match result rows")
    return observed_ids


def evaluate_public_mode(
    *,
    data_root: Path,
    solver_dir: Path,
    official_sources: Path,
    learned_mode: str,
    test_ids: list[int] | None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run one verified public-mode evaluation and return its result payload."""

    data_root = data_root.resolve()
    solver_dir = solver_dir.resolve()
    official_sources = official_sources.resolve()
    test_ids = _normalize_test_ids(test_ids)
    if learned_mode not in LEARNED_MODES:
        raise ValueError(f"unsupported learned mode: {learned_mode!r}")
    entrypoint = solver_dir / "my_optimizer.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(f"solver entrypoint is missing: {entrypoint}")
    evaluator_path = data_root / "iccad2026contest" / "iccad2026_evaluate.py"
    if not evaluator_path.is_file():
        raise FileNotFoundError(f"official evaluator is missing: {evaluator_path}")
    if not official_sources.is_file():
        raise FileNotFoundError(
            f"official-source manifest is missing: {official_sources}"
        )

    verification_notes = _verify_official_checkout(data_root, official_sources)
    official_sources_sha256_before = _file_sha256(official_sources)
    evaluator_sha256_before = _file_sha256(evaluator_path)
    solver_hashes_before = _solver_component_hashes(solver_dir)
    solver_git_before = _git_state(solver_dir)
    official_git_before = _git_state(data_root)
    official_module = _load_official_evaluator(evaluator_path)
    evaluator_type = getattr(official_module, "ContestEvaluator", None)
    compute_total_score = getattr(official_module, "compute_total_score", None)
    if not callable(evaluator_type) or not callable(compute_total_score):
        raise RuntimeError("official evaluator module does not expose its public contract")
    contest_evaluator = evaluator_type(
        data_path=str(data_root), verbose=verbose
    )
    loaded = _configure_optimizer_loader(
        contest_evaluator, entrypoint, learned_mode
    )
    try:
        result = contest_evaluator.evaluate(str(entrypoint), test_ids=test_ids)
    finally:
        # Run these even when evaluator loading or execution raises. A research
        # failure must never conceal mutation of a live or official source.
        solver_hashes_after = _solver_component_hashes(solver_dir)
        evaluator_sha256_after = _file_sha256(evaluator_path)
        official_sources_sha256_after = _file_sha256(official_sources)
        _verify_official_checkout(data_root, official_sources)
        if solver_hashes_after != solver_hashes_before:
            raise RuntimeError("live solver components changed during public evaluation")
        if evaluator_sha256_after != evaluator_sha256_before:
            raise RuntimeError("official evaluator changed during public evaluation")
        if official_sources_sha256_after != official_sources_sha256_before:
            raise RuntimeError(
                "official-source manifest changed during public evaluation"
            )
    if len(loaded) != 1:
        raise RuntimeError("official evaluator did not load exactly one optimizer")
    configured_optimizer = loaded[0]["optimizer"]
    if (
        configured_optimizer._learned_order_mode != learned_mode
        or bool(configured_optimizer._learned_order_enabled)
        != (learned_mode != "off")
        or configured_optimizer._baselines_by_n != {}
    ):
        raise RuntimeError("solver changed the applied research configuration")

    payload = _result_payload(result)
    observed_ids = _validate_result_payload(
        payload,
        test_ids,
        compute_total_score,
    )

    loader_record = {
        key: value for key, value in loaded[0].items() if key != "optimizer"
    }
    payload["submission_name"] = f"my_optimizer-{learned_mode}"
    payload["research_config"] = {
        "schema_version": 1,
        "learned_mode": learned_mode,
        "data_root": _portable_path(data_root),
        "solver_dir": _portable_path(solver_dir),
        "test_ids": test_ids,
        "requested_test_ids": test_ids,
        "evaluated_test_ids": observed_ids,
        "runtime_factor_mode": "neutral_rf_1",
        "uses_environment_override": False,
        "optimizer_configuration": loader_record,
        "harness_sha256": _file_sha256(Path(__file__)),
        "solver_components": solver_hashes_before,
        "solver_git": solver_git_before,
        "official_sources_manifest": {
            "path": _portable_path(official_sources),
            "sha256": official_sources_sha256_before,
        },
        "official_evaluator": {
            "path": _portable_path(evaluator_path),
            "sha256": evaluator_sha256_before,
        },
        "official_floorset_git": official_git_before,
        "official_verification_notes": verification_notes,
    }
    return payload


def _write_json_atomic(
    path: Path,
    payload: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o644)
        if overwrite:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"output already exists (pass --overwrite to replace it): {path}"
                ) from exc
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_OFFICIAL_ROOT)
    parser.add_argument("--solver-dir", type=Path, default=DEFAULT_SOLVER_DIR)
    parser.add_argument(
        "--official-sources", type=Path, default=DEFAULT_OFFICIAL_SOURCES
    )
    parser.add_argument("--learned-mode", choices=LEARNED_MODES, required=True)
    parser.add_argument(
        "--test-id",
        type=int,
        action="append",
        help="evaluate one public case ID; repeat for a bounded panel",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output artifact atomically",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    try:
        if args.output.exists() and not args.overwrite:
            raise FileExistsError(
                f"output already exists (pass --overwrite to replace it): {args.output}"
            )
        payload = evaluate_public_mode(
            data_root=args.data_root,
            solver_dir=args.solver_dir,
            official_sources=args.official_sources,
            learned_mode=args.learned_mode,
            test_ids=args.test_id,
            verbose=args.verbose,
        )
        _write_json_atomic(args.output, payload, overwrite=args.overwrite)
    except Exception as exc:
        print(f"Public-mode evaluation: FAIL\n  {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    rows = payload["test_results"]
    print(
        json.dumps(
            {
                "learned_mode": args.learned_mode,
                "total_score": payload["total_score"],
                "num_tests": payload["summary"]["num_tests"],
                "num_feasible": payload["summary"]["num_feasible"],
                "num_errors": sum(row.get("error") is not None for row in rows),
                "avg_runtime": payload["summary"]["avg_runtime"],
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
