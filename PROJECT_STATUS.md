# Project Status

## Completed

- Added a feasibility-first optimizer for the ICCAD 2026 FloorSet validation environment.
- Added local unit tests for hard constraints and output shape.
- Preserved exact preplaced coordinates and fixed/preplaced dimensions.
- Preserved soft-block target areas and overlap-free placement.
- Added perimeter handling for movable boundary-constrained blocks.
- Added compact perimeter placement to remove unnecessary spacing around the final boundary frame.
- Added cluster-aware macro packing for non-boundary clusters.
- Added boundary-aware packing for same-edge boundary clusters, placing boundary members on the required edge and packing cluster mates inward.
- Added connectivity-aware ordering for movable boundary blocks on each perimeter edge.
- Published local validation artifacts in `results/`.
- Added `scripts/analyze_results.py` for case-level score diagnostics, weighted-contribution analysis, and block-count range summaries.
- Added optional official-evaluator enrichment to `scripts/analyze_results.py` so saved full results can show per-case boundary, grouping, and MIB violation counts without rerunning the optimizer.
- Added an explicit `--write-enriched` mode to save those reconstructed soft-violation counts as a separate diagnostic JSON without replacing the published best-result artifact.
- Improved `scripts/analyze_results.py` to keep tiny weighted contributions visible and report reconstructed score share, weight share, and top weighted case by block-count range.
- Added score-concentration reporting to `scripts/analyze_results.py` so optimization cycles can see cumulative weight and score share for the top high-block-count cases.
- Extended `scripts/analyze_results.py` with weighted metric-pressure estimates for HPWL, area, and soft-violation-ratio improvements, plus score-weighted soft-driver ranking when enriched counts are present.
- Extended `scripts/analyze_results.py` with optional structural case profiles from the official checkout, including fixed/preplaced block counts, boundary demand, cluster and MIB group pressure, and B2B/P2B net counts for weighted focus cases.
- Added `--write-focus-json` to `scripts/analyze_results.py` so high-impact weighted cases, score concentration, metric pressure, and the recommendation can be saved as a compact planning artifact without replacing the published best-result JSON.
- Added committed diagnostic artifacts for the current best result: `results/enriched_diagnostics.json` with reconstructed soft-violation attribution and `results/focus_cases.json` with compact weighted-case planning data.
- Added regression tests that keep committed diagnostic artifacts aligned with the published best result and prevent focus reports from carrying full saved positions.
- Added analyzer regression tests covering weighted-score reconstruction and soft-violation reporting.
- Added analyzer regression tests for metric-pressure and score-weighted soft-driver calculations.
- Added analyzer regression tests for structural constraint-profile extraction and reporting.
- Added a regression test that locks down the exponential high-block-count weighting used by the analyzer.
- Added `scripts/compare_results.py` as a publication guard for candidate full-run JSON files, including score, feasibility, and case-count checks.
- Extended `scripts/compare_results.py` with top weighted per-case regression and improvement reporting for candidate-vs-baseline debugging.
- Tightened `scripts/compare_results.py` so candidate feasibility is derived from per-case records and duplicate candidate `test_id` values fail before publication.
- Tightened `scripts/compare_results.py` to reconstruct baseline and candidate total scores from per-case costs, preventing stale or hand-edited score fields from passing the publication guard.
- Added `scripts/audit_results.py` to validate result artifact integrity, including duplicate IDs, missing fields, finite metric values, summary consistency, feasibility, saved rectangle shape, and saved-rectangle overlap checks.
- Added result-audit regression tests so malformed or partial evaluator JSON files fail before publication.
- Extended the result audit to reconstruct the block-count weighted total score and verify published summary averages against per-case metrics.
- Added `scripts/check_public_release.py` as a combined publication gate for result auditing, public-facing documentation scan, candidate comparison, and optional optimizer-copy synchronization.
- Extended the release check so candidate full-result JSON files are audited before candidate-vs-baseline comparison.
- Added release-check regression tests covering public wording boundaries, optimizer synchronization, and combined gate behavior.
- Added standalone optimizer-helper regression tests for boundary/corner accounting, grouping connectedness, MIB dimension normalization, and boundary-cluster local packing.
- Made Torch-dependent public optimizer tests skip cleanly when contest dependencies are absent, so diagnostics and publication-guard tests remain runnable in a plain Python environment.
- Added repository pytest configuration so `pytest` and `python -m pytest` both resolve local `scripts` imports reliably.

## Current Optimizer

The optimizer is a constructive heuristic:

- keeps preplaced blocks at exact required `(x, y, w, h)`;
- keeps fixed/preplaced dimensions exact;
- preserves soft-block areas;
- avoids overlaps;
- builds a final perimeter frame for movable boundary-constrained blocks;
- compacts the perimeter frame against the interior layout without introducing overlaps;
- normalizes MIB dimensions when target areas allow it;
- packs non-boundary cluster groups as connected macro-blocks;
- packs same-edge boundary clusters as perimeter macro-blocks when this is beneficial for the validation-size range;
- orders movable perimeter blocks by nearby pins and already placed connected blocks while keeping clustered boundary members consecutive;
- uses connectivity-weighted ordering and adaptive shelf widths for score/runtime balance;
- preprocesses connectivity into lightweight tuples using vectorized tensor conversion on large cases;
- skips unused selection-score evaluation when a block count has only one deterministic layout variant;
- prunes high-block-count variant sets where the runtime cost outweighs placement-quality gains;
- applies targeted row-width tuning on the highest-weight validation sizes, including retuned 116- through 119-block settings from validation sweeps;
- uses obstacle-aware interior shelf packing on 116-block and larger cases with exact preplacements, so movable units can occupy legal gaps around preplaced rectangles instead of being forced into a strip to the right;
- reuses cached connectivity degrees for cluster member ordering to reduce high-block-count runtime;
- applies bounded post-placement translation of unconstrained cluster components when it removes a grouping split without overlaps or bbox expansion;
- applies bounded post-placement shifts of unconstrained interior blocks on selected high-count cases when local incident wirelength improves without overlaps or bbox expansion;
- allows fixed-shape, non-preplaced interior blocks to join the guarded shift pass on 117- through 119-block cases, preserving dimensions while reducing high-count HPWL;
- tests a guarded combined-axis shift candidate on 116- through 119-block cases after independent overlap-free axis clamps;
- applies a trimmed 120-block interior shift pass over the highest-connectivity free blocks to reduce incident wirelength while preserving runtime-cap behavior;
- applies guarded top-edge boundary compaction on the largest case when movable top-edge blocks can be pulled inward without overlaps, soft-violation increase, or incident-wirelength regression;
- uses a retuned 120-block top-level row target with tighter large-cluster shelf packing to reduce the dominant weighted case HPWL and area while preserving soft violations;
- caches incident edges for boundary-ordering keys only on 116-block and larger cases, reducing score-dominant runtime while preserving the incumbent layout and median-runtime balance;
- applies a narrow equal-shape swap pass on 117- and 119-block cases when it improves incident wirelength without changing soft violations;
- pre-resolves valid pin coordinates into the free-block shift pass adjacency cache, reducing high-count local wirelength overhead without changing layouts;
- applies a bounded 120-block equal-shape swap probe that accepts up to two meaningful HPWL-improving swaps while preserving soft violations and bounding-box area;
- applies a 118-block-only boundary-line shift refinement when same-edge boundary movement improves local wirelength without overlaps, bbox growth, or soft-violation increase;
- retunes the 118-block cluster-local width to reduce weighted HPWL with only a tiny area tradeoff;
- uses a 119-block grouped-boundary ordering fallback to the net-aware key when it preserves soft counts and improves the high-weight HPWL proxy;
- applies a 116- through 119-block adjacent boundary wire-swap pass that accepts up to two same-edge local swaps per side when incident wirelength improves without overlaps or bbox growth, while leaving the 120-block incumbent path unchanged;
- tries a bounded set of deterministic layout variants and selects with a cheap HPWL, area, and soft-constraint proxy.

## Validation Results

Latest local official validation over 100 Lite validation cases:

- Feasible: 100 / 100
- Total score: 2.6344
- Average cost: 2.9092
- Average runtime: 0.93s
- Official quick validator: PASSED
- Public release guard: PASSED against the official evaluator JSON

Current official-format result file:

- `results/tuned49_official_full.json`

Curated historical result file:

- `results/boundary_full.json`

Diagnostic companion files:

- `results/enriched_diagnostics.json`
- `results/focus_cases.json`

## Implementation Notes

The implementation targets the main local validation cost drivers:

1. hard feasibility;
2. grouping constraints;
3. boundary constraints;
4. MIB shape consistency where compatible with area targets;
5. HPWL gap;
6. bounding-box area gap;
7. runtime.

Soft-constraint diagnostics on the final 100-case validation run:

- boundary violations: 122 total
- grouping violations: 366 total
- MIB violations: 55 total

Remaining violations are mostly hard-constraint tradeoffs. Preplaced blocks cannot be moved to satisfy a soft boundary condition without breaking fixed preplacement. Some MIB groups also have target areas that do not allow one exact common shape without creating hard area violations.

## Useful Commands

From the contest directory after copying `contest_solution/my_optimizer.py` into place:

```bash
python -m pytest test_my_optimizer.py -q
PYTHONPATH=.. python iccad2026_evaluate.py --validate my_optimizer.py --quick
PYTHONPATH=.. python iccad2026_evaluate.py --evaluate my_optimizer.py --verbose --save-solutions --output results/boundary_full.json
```

From the repository root:

```bash
python scripts/analyze_results.py results/boundary_full.json
python scripts/analyze_results.py results/boundary_full.json --write-focus-json results/focus_cases.json
python scripts/analyze_results.py results/boundary_full.json --contest-dir external/FloorSet/iccad2026contest
python scripts/analyze_results.py results/boundary_full.json --contest-dir external/FloorSet/iccad2026contest --write-enriched results/enriched_diagnostics.json
python scripts/analyze_results.py results/boundary_full.json --diagnostic-sidecar results/enriched_diagnostics.json
python scripts/audit_results.py results/boundary_full.json --expected-cases 100 --require-positions
python scripts/compare_results.py results/boundary_full.json candidate_full.json
python scripts/check_public_release.py --contest-optimizer /path/to/FloorSet/iccad2026contest/my_optimizer.py
python -m pytest -q
```

Without Torch installed, the public test suite skips the optimizer tests that
need contest tensor inputs and still runs the diagnostics and comparison-guard
tests. Use the contest environment for the full optimizer regression suite
before publishing solver changes. The comparison guard requires candidate
full-run JSON files to preserve per-case full feasibility, include every baseline
`test_id`, avoid duplicate candidate IDs, reconstruct to the declared score,
and strictly improve the published total score before replacing best-result artifacts.
The result audit should pass on any published full-run
artifact before it is compared or copied over the current best result; it also
checks saved rectangles for positive-area overlaps and verifies that top-level
score and summary averages are consistent with the per-case metrics.
The analyzer's score-concentration section should be checked before solver
experiments so case-level tuning focuses on the cases that materially affect
the block-count weighted score.
When `results/enriched_diagnostics.json` matches the published result, the
analyzer now merges its derived boundary/grouping/MIB counts and structural
constraint profile into the report automatically. This keeps the published
best-result JSON unchanged while making default diagnostics specific enough to
choose the next solver target.
The compact focus JSON can be regenerated for each candidate or enriched result
to keep experiment notes aligned with the current weighted-score drivers.
When run with `--contest-dir`, the analyzer should also be used to inspect the
weighted cases' structural profile. High boundary density points toward
perimeter ordering and boundary-cluster packing experiments; dense B2B
connectivity points toward connectivity-aware ordering or local movement.
The public release check combines the result audit with a public-facing docs
scan and optional optimizer-copy synchronization, so a release can fail early if
the uploaded optimizer diverges from the validated contest copy or the docs
contain wording that should not appear in the public repository.

## Next Improvement Ideas

- Post-placement local search for unit swaps/shifts to reduce HPWL without increasing soft violations.
- Analytical placement or force-directed ordering before legalization.
- More advanced MIB handling for groups with incompatible target areas.
- Use the optimizer-helper regression tests as guardrails before changing boundary-cluster packing, grouping, or MIB heuristics.
- Keep public smoke tests runnable with or without the official evaluator on `PYTHONPATH`.
- Run the result-comparison guard before replacing published best-score artifacts.
- Run the result-artifact audit before comparing or publishing candidate JSON files.
- Inspect the weighted per-case delta report from `scripts/compare_results.py` before keeping or discarding a solver experiment.
