#!/usr/bin/env python3
"""Authoritative list of source modules used by the live solver.

The official-copy scripts, synchronization audit, and holdout provenance all
consume this registry.  Keep implementation modules and generated deployment
artifacts adjacent here when adding a new live model family.
"""

from __future__ import annotations

from pathlib import Path


SOLVER_ENTRYPOINT = "my_optimizer.py"

LIVE_SOLVER_COMPONENTS = (
    SOLVER_ENTRYPOINT,
    "dissect.py",
    "topology_polish.py",
    "learned_order.py",
    "order_model_v5b.py",
    "golden_plus_repair.py",
)

# Package-only sources deliberately remain separate from the live contest
# module registry.  Their archive names differ from their repository names,
# and they must be provenance-bound without being copied into FloorSet.
PACKAGE_SUPPORT_SOURCE_BINDINGS = {
    "packaging/torch_stub.py": "source_fallback/torch.py",
    "packaging/eval_stub.py": "source_fallback/iccad2026_evaluate.py",
    "packaging/solver_main.py": "source_fallback/solver_main.py",
}


def validate_live_solver_components(
    components=LIVE_SOLVER_COMPONENTS,
    *,
    source_dir: Path | None = None,
) -> tuple[str, ...]:
    """Validate and return the deployable module registry.

    Deployment consumers deliberately accept only importable, flat Python
    module filenames.  A nested path, duplicate, or non-Python artifact could
    otherwise be copied under one name and imported under another.  When a
    source directory is supplied, every registry entry must also exist as a
    regular file.
    """

    validated: list[str] = []
    modules: set[str] = set()
    names: set[str] = set()
    reserved_archive_names = {
        Path(packaged_name).name
        for packaged_name in PACKAGE_SUPPORT_SOURCE_BINDINGS.values()
    }
    for component in components:
        if not isinstance(component, str) or not component:
            raise ValueError("live solver component names must be non-empty strings")
        path = Path(component)
        if (
            path.name != component
            or "/" in component
            or "\\" in component
            or path.suffix != ".py"
        ):
            raise ValueError(
                f"live solver component must be a flat .py filename: {component!r}"
            )
        module = path.stem
        if not module.isidentifier():
            raise ValueError(
                f"live solver component is not an importable module: {component!r}"
            )
        if component in reserved_archive_names:
            raise ValueError(
                "live solver component collides with a package support source: "
                f"{component!r}"
            )
        if component in names or module in modules:
            raise ValueError(f"duplicate live solver component: {component!r}")
        if source_dir is not None and not (source_dir / component).is_file():
            raise ValueError(
                f"live solver component is missing: {source_dir / component}"
            )
        names.add(component)
        modules.add(module)
        validated.append(component)

    if not validated:
        raise ValueError("live solver component registry must not be empty")
    if validated[0] != SOLVER_ENTRYPOINT:
        raise ValueError("live solver entrypoint must be the first registry component")
    return tuple(validated)


def main() -> None:
    """Print one component per line for the Bash copy workflows."""

    source_dir = Path(__file__).resolve().parents[1] / "contest_solution"
    try:
        components = validate_live_solver_components(source_dir=source_dir)
    except ValueError as exc:
        raise SystemExit(f"invalid live solver registry: {exc}") from exc
    print("\n".join(components))


if __name__ == "__main__":
    main()
