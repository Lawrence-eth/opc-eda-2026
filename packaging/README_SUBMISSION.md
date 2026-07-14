# ICCAD 2026 CAD Contest — Problem C (FloorSet Challenge) Submission

Deterministic heuristic floorplanning optimizer, packaged per the Problem C
submission guidelines (2026-06-16) and Q&A (2026-06-18).

## Contents

```
dist/my_optimizer/my_optimizer   x86-64 Debian 13 PyInstaller executable
op_wrapper.py                    the organizers' example wrapper, verbatim
source_fallback/                 pure-Python source (stdlib only), same solver
requirements.txt                 nothing to install for the executable path
THIRD_PARTY_NOTICES.md           provenance and dataset attribution
LICENSES/Apache-2.0.txt          FloorSet's Apache-2.0 license
README.md                        this file
```

## How to run (matches the official evaluation command)

```bash
tar xzf iccad2026_submission.tar.gz
cd <FloorSet>/iccad2026contest
cp -r <extracted>/dist .
cp <extracted>/op_wrapper.py .
python iccad2026_evaluate.py --evaluate op_wrapper.py
```

`op_wrapper.py` resolves the executable at `dist/my_optimizer/my_optimizer`
relative to itself (its default candidate path). `MY_OPT_BIN` may be set to
an alternative path if the layout differs.

## Executable interface

Exactly the `solve()` schema of `optimizer_template.py`, transported as JSON
(as in the provided `op_wrapper.py`):

- stdin: one JSON object — `block_count`, `area_targets`,
  `b2b_connectivity`, `p2b_connectivity`, `pins_pos`, `constraints`,
  `target_positions` (or null)
- stdout: `{"positions": [[x, y, w, h], ...]}` with exactly `block_count`
  entries

## Properties

- Single-threaded, CPU-only, deterministic (fixed seeds); no GPU, no network,
  no filesystem writes.
- No external dependencies: the x86-64 executable bundles a minimal Python runtime;
  the source fallback runs on any Python ≥3.10 with the standard library only
  (it does not import real torch — inputs arrive as JSON lists).
- Never crashes: on any internal error it emits a conservative feasible
  layout rather than failing the case.
- Typical runtime ≈0.2–0.4 s per case end-to-end (including process start),
  ≤1.5 s on the largest cases.

## Source fallback

If the executable cannot be used, `source_fallback/solver_main.py` is the
same solver as plain Python (reads the same JSON on stdin; `my_optimizer.py`
+ `dissect.py` + `topology_polish.py` + `learned_order.py` +
`order_model_v5b.py` + `golden_plus_repair.py` are the live solver;
`torch.py`/`iccad2026_evaluate.py` are
self-contained stand-ins for the imports). It can also be
adapted to the import-based interface: `source_fallback/my_optimizer.py`
contains the `MyOptimizer(FloorplanOptimizer)` class with the standard
`solve()` signature and works unmodified against the official
`iccad2026_evaluate.py` when real torch is available.
