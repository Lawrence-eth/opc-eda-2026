# Golden-plus fixed-topology repair v1

## Decision

Promote the **MIB-only safe gate** as a final post-solve repair. It improves the
saved v32 candidate on both allowed validation panels, remains feasible on all
205 cases, and has zero official-metric mismatches. Keep the broader boundary
and grouping searches disabled in production for now.

This is a generic, fail-closed transformation. It uses only the submitted
instance, its candidate placement, and published constraints. It contains no
case ID, source-file, generator, or dataset-identity branch.

## Safety contract

Every accepted repair must:

- preserve fixed/preplaced geometry and exact block area;
- preserve a frozen non-overlap separation for every block pair;
- preserve existing grouping-component contact through a spanning forest;
- reduce the total soft-violation count without increasing any category;
- not increase HPWL or bounding-box area beyond `1e-7` numerical tolerance;
- pass a final independent hard-feasibility check.

Any exception, infeasible input, unavailable safe pattern, or failed gate
returns the original placement unchanged.

The deployable fast path additionally requires one three-member equal-area MIB
group with exactly one legal oriented factor shape already present, at least
two unoriented factor choices, square fallback shapes on movable outliers, and
shape compatibility for every locked member. Only the lower-right anchored
reshape is considered. This makes almost every case an allocation-light O(n)
no-op and subjects a rare candidate to exact acceptance checks.

## Official-score evidence

All scores were independently recomputed with the pinned official evaluator;
stored metrics were not trusted. Scope was public plus the explicitly named
`heavy_clean_v1` fold 0. Sealed-v2 was not accessed.

| Candidate / panel | RF1 before | RF1 after | Delta | Accepted | Violations removed | Feasible | Max HPWL delta | Max bbox delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v32 + MIB gate, public | 1.615378775 | 1.615181467 | -0.000197308 | 1/100 | 1 MIB | 100/100 | 0 | 0 |
| v32 + MIB gate, clean fold 0 | 1.790580526 | 1.790097469 | -0.000483057 | 1/105 | 1 MIB | 105/105 | 0 | 0 |
| supplied golden + combined research, public | 1.107940199 | 1.106350377 | -0.001589822 | 11/100 | 14 boundary | 100/100 | 0 | 0 |
| supplied golden + combined research, clean fold 0 | 1.107727046 | 1.103831342 | -0.003895704 | 11/105 | 11 boundary | 105/105 | 0 | 0 |

The golden experiment proves the supplied golden placement is not a hard
optimum: a generic boundary repair beats its aggregate score on both panels.
That broader repair did not improve v32, so it is retained as research rather
than deployed. Golden MIB-only and grouping-only ablations produced no wins;
combined equalled boundary-only.

The v32 MIB gate also reduced aggregate HPWL by `0.058048100` on public and
`0.097488866` on clean fold 0 while leaving aggregate bounding-box area
unchanged. There were zero official rejections and zero metric mismatches on
all 205 cases.

## Runtime

A 20-repeat no-report benchmark using plain JSON/list inputs (matching the
packaged path) measured:

| Panel | Mean/case | p95/case | Mean full sweep | Changed cases |
|---|---:|---:|---:|---:|
| public | 0.0898 ms | 0.1030 ms | 9.046 ms / 100 | 1 |
| clean fold 0 | 0.1292 ms | 0.1168 ms | 13.674 ms / 105 | 1 |

After charging this overhead to observed solver runtime, the repaired candidate
beats v32 at every evaluated common field-median runtime from 0.1 through 3.0
seconds on both panels.

## Production integration

Run the gate after the final candidate selector and topology polish, immediately
before returning the placement:

```python
from golden_plus_repair import RepairConfig, repair_fixed_topology

_GOLDEN_PLUS_MIB_CONFIG = RepairConfig(
    enable_boundary=False,
    enable_mib=True,
    enable_grouping=False,
    require_safe_mib_pattern=True,
)

best_positions = repair_fixed_topology(
    best_positions,
    area_targets,
    b2b_edges,
    p2b_edges,
    pins_l,
    constraints,
    target_positions,
    config=_GOLDEN_PLUS_MIB_CONFIG,
)
```

Use the original tensor-or-list inputs for area, constraints, and target
positions: the dissection-local `areas_l`, `con_l`, and `tp_l` variables may
not exist when an earlier dissection step fails. The optional
`target_positions` argument enforces fixed/preplaced geometry; omit it only if
those targets are unavailable and the incumbent itself is the required
reference.

For packaging, copy `contest_solution/golden_plus_repair.py` into
`submission/src`, add `golden_plus_repair` to hidden imports, and include it in
the source fallback archive. Run a packaged import/smoke test before promotion.

## Reproduction

```bash
python scripts/evaluate_golden_plus_repair.py --v32-only
python scripts/evaluate_golden_plus_repair.py --v32-ablation
python scripts/benchmark_golden_plus_fast_path.py
pytest -q tests/test_golden_plus_repair.py
```

Exact source/data/evaluator and raw-artifact hashes are recorded in
`results/research/golden_plus_repair_v1.json`.
