# FloorSet ICCAD-2026 — Next Iteration Plan (from 2.5443)

Verified baseline: **2.5443**, 100/100 feasible, reproduces from live `contest_solution/my_optimizer.py` (re-evaluated, internally consistent). Runtime sum 60.5s, max 3.3s. Sprint-1/2 foundation work is solid — keep it.

## Diagnosis (from sprint2_v6, exp(n/12) official weighting)

- Weighted **quality factor `1+0.5*(hpwl_gap+area_gap)` = 2.07**; weighted **soft factor `exp(2*vrel)` = 1.23**. Quality is ~80% of the score.
- **HPWL gap is the dominant enemy: ~1.36** across 81–115 (you are ~136% over the golden wirelength). Area gap ~1.06. Soft is now minor.
- 81–120 = **96.5%** of total weight. 101–115 alone = 50% of the score.

## THE job: a real global-placement core (this was NOT actually tried)

Everything in the report's "what didn't work" list (force-directed, centroid sort, compaction, real-cost) failed for **one** reason: it was applied as a *post-pass on a frozen, overlap-locked shelf layout* ("blocks couldn't move; converged in 1–2 iterations"). That is local nudging on a bad structure, not global placement. The shelf packer places connected blocks in different rows; no legal micro-shift fixes that.

Build the pipeline below to **replace the shelf packer as the seed** (keep shelf packer as a fallback — see guardrail).

### Pipeline
1. **Global placement, overlaps ALLOWED.** Per axis independently, minimize weighted HPWL with pins as fixed anchors.
   - Easy/robust: iterative weighted-centroid relaxation (Gauss–Seidel). For each free block, set its center to the connectivity-weighted average of neighbor centers + connected pin positions; sweep ~30–50× (or until movement < eps). Preplaced/fixed-position blocks are anchors and never move.
   - Better: true quadratic placement — build the b2b+p2b weighted Laplacian, solve `L x = b` (and `L y = b`) per axis with conjugate gradient (`scipy.sparse.linalg.cg`). Pins/preplaced go to the RHS as fixed terms. Fast even at n=120.
2. **Legalization (single pass, minimal displacement).** Sort blocks by analytical position; place with the existing contour/shelf packer biased to the analytical target so wirelength structure survives. Respect: preplaced exact (x,y,w,h); fixed-shape exact (w,h); soft-block area within 1% (only adjust aspect ratio, never area).
3. **Compaction** toward origin, both axes, overlap-safe (process in sorted order, slide each block until it touches a neighbor/edge). Directly cuts area_gap. (Prior "5.15" was a buggy overlap-creating version — gate every move on no-overlap + bbox-not-worse.)
4. **Detailed SA on the REAL contest cost** (see below), strict ≤1.0s budget, seeded from the legalized layout. Allow relocation moves (block → empty region / swap rows), not just in-gap nudges.

### Guardrail (non-negotiable — makes the rewrite zero-risk)
Per case, run BOTH the new analytical path and the current shelf path; keep whichever has lower **real contest cost**. Worst case the total ties 2.54; best case it drops. Never regress the baseline.

### Use the REAL cost everywhere you select/anneal
Stop using `hpwl + 0.01*area`. Reuse the official `iccad2026_evaluate.evaluate_solution` (or replicate exactly) to compute the true per-case cost = `(1+0.5*(max(0,hpwl_gap)+max(0,area_gap))) * exp(2*vrel)` against that case's **baseline metrics** (precompute baselines once via `--baseline` / `_extract_baseline`). Variant selection AND SA acceptance must use this, or you optimize the wrong landscape (the report's "real cost function hurt" was because only *selection* switched while the rest stayed proxy-tuned — switch all of it together).

## Target
hpwl_gap/area_gap from 1.36/1.06 → ~0.5/0.4 ⇒ quality factor ~1.45 ⇒ **total ~1.5–1.8** (winning-tier). Stretch toward ~1.2 with strong SA + grouping cleanup.

## Two standing warnings
- **Overfitting / hidden test:** the `_layout_variants` `tuned{}` table and `==118/119/120` refine gates are fit to THESE 100 validation cases. Final ranking is a DIFFERENT hidden 100 (same 21–120 range). Once the analytical core beats 2.54, **delete the per-count special-casing** and re-measure — a clean general placer transfers better. Do not carry overfit tuning forward.
- **Runtime is a live leaderboard risk** (local eval forces RuntimeFactor=1.0; the real one is `your_rt / cross-submission median`, penalty uncapped). Quadratic placement is fast, so this rewrite should *lower* runtime. Keep max per-case ≤ ~1–2s; watch the 90–99 band.

## Optional parallel track (high effort, high upside): ML warm-start
1M training samples with golden solutions + differentiable proxy loss ship with the contest (`get_training_dataloader`, `compute_training_loss_differentiable`); inference target is A100 80GB. A GNN over the b2b/p2b graph predicting per-block (x,y) gives a near-instant, high-quality seed that you then legalize+compact+SA — fixes runtime AND quality simultaneously. Doesn't block the analytical core; pursue if you want to push past ~1.5.

## Hygiene still open from the prior plan
- Regenerate/delete the stale `results/summary.json` (still claims 1.50 for a 9.69 file). Make `audit_results.py` fail CI when a file's top-level score disagrees with the recompute from its per-case costs.
- After the core lands, run the leave-one-out ablation on the ~15 refine passes (now cheap); delete any that don't earn a net win.

## Commands
```bash
cd /home/ubuntu/EDA
cp contest_solution/my_optimizer.py external/FloorSet/iccad2026contest/
cd external/FloorSet/iccad2026contest
PYTHONPATH=.. /home/ubuntu/EDA/.venv/bin/python3 iccad2026_evaluate.py --evaluate my_optimizer.py --output /home/ubuntu/EDA/results/next.json
cd /home/ubuntu/EDA && python3 scripts/analyze_results.py results/next.json --top 20
```
