#!/usr/bin/env python3
"""Fail closed before evaluating solver modules copied into FloorSet.

Run this with the official virtualenv after copying the registry-defined live
modules.  It checks the copied directory, not the repository originals.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
import sys

if __package__:
    from .solver_components import LIVE_SOLVER_COMPONENTS
else:
    from solver_components import LIVE_SOLVER_COMPONENTS


EXPECTED_V5B_PAYLOAD_SHA256 = (
    "c94b4af92a7088f04206a5fa20dfbf807f945d9bdd80d9ffcbdc0b8b45f18beb"
)
DEFAULT_POLICY_MIN_BLOCKS = 100
DEFAULT_POLICY_MAX_BLOCKS = 120
DEFAULT_ABSTAIN_SIZES = (101, 109, 112)
STANDARD_REPLACEMENT_WIDTHS = frozenset((0.8, 0.9, 1.0, 1.1, 1.2))


class PreflightError(RuntimeError):
    """A copied solver component violated the production contract."""


def _require(condition, message):
    if not condition:
        raise PreflightError(message)


def _origin(module, name):
    raw = getattr(module, "__file__", None)
    _require(raw, f"{name} has no importable file origin")
    return Path(raw).resolve()


def _require_real_torch(solver_dir):
    """Load Torch before making the copied solver directory importable."""

    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        raise PreflightError(f"real Torch import failed: {exc}") from exc
    origin = _origin(torch, "torch")
    _require(
        solver_dir != origin.parent and solver_dir not in origin.parents,
        f"Torch resolved inside copied solver tree: {origin}",
    )
    spec = getattr(torch, "__spec__", None)
    _require(
        spec is not None and spec.submodule_search_locations is not None,
        "Torch is not an installed package",
    )
    _require(
        isinstance(getattr(torch, "__version__", None), str),
        "Torch does not expose a real package version",
    )
    _require(
        all(hasattr(torch, name) for name in ("Tensor", "empty", "tensor")),
        "Torch lacks the tensor API required by the solver",
    )
    try:
        probe = torch.tensor([1.0, 2.0], dtype=torch.float64)
    except Exception as exc:
        raise PreflightError(f"Torch tensor probe failed: {exc}") from exc
    _require(
        isinstance(probe, torch.Tensor) and probe.tolist() == [1.0, 2.0],
        "Torch tensor probe returned an unexpected object",
    )
    return torch, origin


def _import_copied_modules(solver_dir):
    components = ("iccad2026_evaluate.py", *LIVE_SOLVER_COMPONENTS)
    _require(
        len(components) == len(set(components)),
        "live solver registry contains duplicate components",
    )
    modules = {}
    for component in components:
        relative = Path(component)
        _require(
            relative.name == component and relative.suffix == ".py",
            f"invalid live solver component name: {component!r}",
        )
        expected = (solver_dir / relative).resolve()
        _require(expected.is_file(), f"copied solver component is missing: {expected}")
        name = relative.stem
        existing = sys.modules.get(name)
        if existing is not None:
            _require(
                _origin(existing, name) == expected,
                f"{name} was already imported from the wrong location",
            )
        try:
            module = existing or importlib.import_module(name)
        except Exception as exc:
            raise PreflightError(f"copied module {name} failed to import: {exc}") from exc
        _require(
            _origin(module, name) == expected,
            f"copied module {name} did not resolve to {expected}",
        )
        modules[name] = module
    return modules


def check_v5b_integrity(learned_order, order_model, optimizer_module):
    model = getattr(order_model, "MODEL", None)
    _require(isinstance(model, dict), "order_model_v5b.MODEL is not a mapping")
    payload_sha256 = model.get("payload_sha256")
    _require(
        payload_sha256 == EXPECTED_V5B_PAYLOAD_SHA256,
        "v5b payload SHA mismatch: "
        f"expected {EXPECTED_V5B_PAYLOAD_SHA256}, got {payload_sha256!r}",
    )
    compile_artifact = getattr(learned_order, "compile_artifact", None)
    _require(callable(compile_artifact), "learned_order.compile_artifact is unavailable")
    try:
        compiled = compile_artifact(model)
    except Exception as exc:
        raise PreflightError(f"v5b canonical integrity/compilation failed: {exc}") from exc
    _require(
        isinstance(compiled, dict) and compiled,
        "v5b compilation returned an empty or malformed payload",
    )
    live_compile = getattr(optimizer_module, "_compiled_learned_model", None)
    _require(callable(live_compile), "optimizer has no v5b compilation path")
    try:
        live_compiled = live_compile()
    except Exception as exc:
        raise PreflightError(f"optimizer v5b compilation path raised: {exc}") from exc
    _require(
        live_compiled == compiled,
        "optimizer v5b compilation does not match direct compilation",
    )
    return {
        "payload_sha256": payload_sha256,
        "message_steps": compiled.get("message_steps"),
        "feature_count": len(compiled.get("coefficients", ())),
    }


def check_replacement_policy(
    optimizer_module,
    *,
    minimum_blocks,
    maximum_blocks,
    expected_abstain_sizes,
):
    _require(minimum_blocks <= maximum_blocks, "replacement policy range is empty")
    abstentions = tuple(expected_abstain_sizes)
    _require(abstentions, "at least one known replacement abstention is required")
    _require(
        len(abstentions) == len(set(abstentions)),
        "known replacement abstentions contain duplicates",
    )
    domain = set(range(minimum_blocks, maximum_blocks + 1))
    abstention_set = set(abstentions)
    _require(
        abstention_set <= domain,
        "known replacement abstention falls outside policy range",
    )

    policy = getattr(optimizer_module, "_LEARNED_REPLACEMENT_WF", None)
    _require(
        isinstance(policy, dict) and policy,
        "production replacement policy is missing or empty",
    )
    for block_count, width_factor in policy.items():
        _require(
            isinstance(block_count, int) and not isinstance(block_count, bool),
            "replacement policy block counts must be integers",
        )
        _require(
            isinstance(width_factor, (int, float))
            and not isinstance(width_factor, bool),
            "replacement policy widths must be numeric",
        )
        numeric_width = float(width_factor)
        _require(
            math.isfinite(numeric_width)
            and numeric_width in STANDARD_REPLACEMENT_WIDTHS,
            f"invalid replacement width {numeric_width!r} for n={block_count}",
        )

    expected_mapped = domain - abstention_set
    actual_mapped = set(policy)
    _require(
        actual_mapped == expected_mapped,
        "replacement policy coverage mismatch: "
        f"missing={sorted(expected_mapped - actual_mapped)}, "
        f"unexpected={sorted(actual_mapped - expected_mapped)}",
    )
    optimizer_class = getattr(optimizer_module, "MyOptimizer", None)
    _require(isinstance(optimizer_class, type), "copied solver has no MyOptimizer")
    try:
        optimizer = optimizer_class(verbose=False)
    except Exception as exc:
        raise PreflightError(f"MyOptimizer construction failed: {exc}") from exc
    _require(
        getattr(optimizer, "_learned_order_enabled", None) is True,
        "learned ordering is not enabled in production",
    )
    _require(
        getattr(optimizer, "_learned_order_mode", None) == "replacement",
        "production learned-order mode is not replacement",
    )
    return {
        "mode": "replacement",
        "minimum_blocks": minimum_blocks,
        "maximum_blocks": maximum_blocks,
        "mapped_sizes": len(policy),
        "abstain_sizes": sorted(abstentions),
    }


def exercise_safe_mib_repair(repair_module, torch):
    repair = getattr(repair_module, "repair_fixed_topology", None)
    config_class = getattr(repair_module, "RepairConfig", None)
    _require(
        callable(repair) and isinstance(config_class, type),
        "golden_plus_repair does not expose the production API",
    )
    square = math.sqrt(40.0)
    positions = [
        (0.0, 0.0, square, square),
        (20.0, 0.0, square, square),
        (40.0, 0.0, 4.0, 10.0),
        (0.0, 20.0, 1.0, 1.0),
    ]
    constraints = [
        (0, 0, 1, 0, 0),
        (0, 0, 1, 0, 0),
        (0, 0, 1, 0, 0),
        (0, 0, 0, 0, 0),
    ]
    config = config_class(
        enable_boundary=False,
        enable_mib=True,
        enable_grouping=False,
        require_safe_mib_pattern=True,
    )

    def invoke():
        return repair(
            torch.tensor(positions, dtype=torch.float64),
            torch.tensor([40.0, 40.0, 40.0, 1.0], dtype=torch.float64),
            torch.empty((0, 3), dtype=torch.float64),
            torch.empty((0, 3), dtype=torch.float64),
            torch.empty((0, 2), dtype=torch.float64),
            torch.tensor(constraints, dtype=torch.float64),
            torch.full((4, 4), -1.0, dtype=torch.float64),
            config=config,
            return_report=True,
        )

    try:
        (first, first_report), (second, second_report) = invoke(), invoke()
        first = tuple(tuple(float(item) for item in row) for row in first)
        second = tuple(tuple(float(item) for item in row) for row in second)
    except Exception as exc:
        raise PreflightError(f"safe-MIB repair probe failed: {exc}") from exc
    _require(first != tuple(positions), "safe-MIB repair did not change output")
    _require(first == second, "safe-MIB repair probe is not deterministic")
    _require(
        first_report.changed and second_report.changed,
        "safe-MIB repair report did not record a change",
    )
    _require(
        first_report.accepted.get("mib") == 1,
        "safe-MIB repair did not accept exactly one group",
    )
    expected_shape = (4.0, 10.0)
    _require(
        tuple(row[2:] for row in first[:3]) == (expected_shape,) * 3,
        "safe-MIB repair did not produce the expected common shape",
    )
    return {"changed": True, "common_shape": [4.0, 10.0], "accepted_groups": 1}


def run_preflight(
    solver_dir,
    *,
    minimum_blocks=DEFAULT_POLICY_MIN_BLOCKS,
    maximum_blocks=DEFAULT_POLICY_MAX_BLOCKS,
    expected_abstain_sizes=DEFAULT_ABSTAIN_SIZES,
):
    solver_dir = Path(solver_dir).resolve()
    _require(solver_dir.is_dir(), f"copied solver directory is missing: {solver_dir}")
    torch, torch_origin = _require_real_torch(solver_dir)
    original_sys_path = list(sys.path)
    try:
        sys.path.insert(0, str(solver_dir))
        importlib.invalidate_caches()
        modules = _import_copied_modules(solver_dir)
        result = {
            "status": "PASS",
            "solver_dir": str(solver_dir),
            "torch": {"version": torch.__version__, "origin": str(torch_origin)},
            "imported_modules": {
                name: str(_origin(module, name))
                for name, module in sorted(modules.items())
            },
            "v5b": check_v5b_integrity(
                modules["learned_order"],
                modules["order_model_v5b"],
                modules["my_optimizer"],
            ),
            "replacement_policy": check_replacement_policy(
                modules["my_optimizer"],
                minimum_blocks=minimum_blocks,
                maximum_blocks=maximum_blocks,
                expected_abstain_sizes=expected_abstain_sizes,
            ),
            "safe_mib_repair": exercise_safe_mib_repair(
                modules["golden_plus_repair"], torch
            ),
        }
    finally:
        sys.path[:] = original_sys_path
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-dir", type=Path, required=True)
    parser.add_argument("--policy-min-blocks", type=int, default=100)
    parser.add_argument("--policy-max-blocks", type=int, default=120)
    parser.add_argument(
        "--expected-abstain-size",
        action="append",
        type=int,
        help="Expected abstention; repeat as needed (default: 101,109,112).",
    )
    args = parser.parse_args(argv)
    abstentions = (
        tuple(args.expected_abstain_size)
        if args.expected_abstain_size is not None
        else DEFAULT_ABSTAIN_SIZES
    )
    try:
        result = run_preflight(
            args.solver_dir,
            minimum_blocks=args.policy_min_blocks,
            maximum_blocks=args.policy_max_blocks,
            expected_abstain_sizes=abstentions,
        )
    except Exception as exc:
        print(f"ERROR: official live-solver preflight failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
