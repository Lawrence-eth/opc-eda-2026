# FloorSet ICCAD-2026 — Comprehensive Plan Execution Log

### 2026-07-07 clamp-branch segmented-fill experiment — ❌ REVERTED
- Hypothesis: rows crossing obstacle edges should fill around obstacles at natural height instead of
  jumping to the edge (case-70 whitespace). Result: dissect-only 1.9393 -> 1.9797 (n>=100 ag 0.249->0.294),
  case 70 UNCHANGED (its dead zone forms below py1 through a different path). Reverted via git checkout.
- Open lead (case 70, n=91, util 0.49): interior fill strands y in [0, py1] region almost empty when
  preplaced obstacles are scattered low; blocks pile above py1=206 -> H 407 vs ideal ~201. Needs a real
  trace of fill_region's y-progression, not another guess.

### 2026-07-07 integrated v10 — cluster edge stacks + band free-width + clamp re-queue
- Clusters with multiple same-edge boundary members: L/R members now form vertical stacks on the
  cluster's outer side (all touch the edge). Band height re-derived from obstacle-free width.
  Clamped rows always re-queue L/R heads/tails (right-alignment preserved).
- dissect-only 1.9393 (91/100 wins), n>=100 vr 0.110 (G3 V_rel gate met); ag regressed 0.18->0.25
  (band heights; case 70 outlier util 0.49 — shielded by best-of; open lead)
- Integrated official **1.8074**, 100/100, rt avg 0.18s; radj@{0.5,1,2,3}s = 1.602/1.353/1.266/1.265 — ✅ kept

### 2026-07-07 integrated v9 — MIB forcing + segment L/R injection + edge retouch
- MIB groups split across units -> forced shared square dims (slack traded for shape consistency)
- L/R-required units injected into die-edge obstacle segments; final edge-retouch pass (live clash check)
- BUG caught by full eval: retouch moved PREPLACED blocks toward soft boundary edges -> 9 hard-infeasible
  cases (dims/position mismatch). Preplaced now excluded (Q&A Q5). Lesson: every retouch/move pass must
  skip preplaced explicitly.
- Result: dissect-only 2.0000 (93/100 wins); integrated official **1.8975**, 100/100, rt avg 0.20s;
  radj@{0.5,1,2,3}s = 1.728/1.444/1.333/1.328 — ✅ kept

### 2026-07-07 integrated v4-v8 — portfolio/runtime discipline round
- v4 (15-candidate wf×pin_scale grid): RF=1 1.9409 but radj@1s 1.926 — ❌ REVERTED (runtime > quality; I.3)
- v5 (+0.25s order-refinement rebuild search): RF=1 1.9460, radj@1s 1.917 — ❌ dormant behind _REFINE_BUDGET=0 (needs ≥10% gain to pay its multiplier; got ~1%)
- v6 (two-pass construction, pass-2 reorders from pass-1 positions): RF=1 1.9352 — ✅ kept (free quality)
- v7/v8 (fast-first: no-SA shelf as reference on n≥50; full-SA shelf only if it wins selection — it never did): rt avg 0.27→0.18s, max 0.64s — ✅ kept
- Net: **RF=1 1.9352, 100/100, rt avg 0.18s; radj@{0.5,1,2,3}s = 1.718/1.451/1.356/1.355** (v9: 2.549/2.110/1.924/1.903)

### 2026-07-07 dissect_v2 + integration — CAMPAIGN_GOLDEN G2-G4 (see docs/CAMPAIGN_GOLDEN.md)
- Hypothesis: golden = near-perfect tessellation ⇒ exact-area dissection closes area_gap structurally.
- Runtime-adjusted Total @ median {0.5,1,2,3,5}s: 2.260 / 1.849 / 1.584 / 1.508 / 1.484  (v9: 2.549 / 2.110 / 1.924 / 1.903 / 1.903)
- Local RF=1 Total: 2.1204 | Feasible: 100/100 | runtime avg/max: 0.27s / 1.35s
- n≥100 hpwl_gap ~0.89, area_gap ~0.21, V_rel ~0.176 (dissect-only stats)
- Verdict: ✅ kept — integrated behind feasibility-gated best-of (93/100 cases improved)
- New dead-end: clamped gaps in a self-normalized selector censor improvements (use SIGNED gaps when baselines absent)

> **Purpose:** Complete experiment history, dead-ends, architecture, and strategic assessment.  
> **Read this before touching code.** Every future agent must read Part I (invariants) and Part II (dead-ends) before acting.  
> **Last updated:** 2026-05-30, after M4' surgical hybrid (reverted).

---

## HEAD Result

| Field | Value |
|-------|-------|
| **Best committed** | `sprint5_v9` = **2.7182** (local RF=1.0) |
| **Feasible** | 100/100 |
| **Runtime sum / max** | 18.1s / 0.9s |
| **Runtime-adjusted (median=1.0s)** | **2.135** |
| **Runtime-adjusted (median=2.0s)** | **1.932** |
| **Runtime-adjusted (median=3.0s)** | 1.903 |
| **vs baseline (quadratic_v1) at median=1.0s** | **−0.517** (we win) |

**Score formula:**
```
cost_i = (1 + 0.5·max(0,hpwl_gap) + 0.5·max(0,area_gap)) · exp(2·V_rel) · max(0.7, (rt/median)^0.3)
Total  = Σ cost_i · exp(n_i/12) / Σ exp(n_j/12)
```

---

## Part I — Invariants (never change; judge everything against these)

1. **Feasibility 100/100 is non-negotiable.** One infeasible case = cost 10 ≈ catastrophic.
2. **Judge only on runtime-adjusted total** (median 1–3s), not local RF=1.0.
3. **De-overfit:** ranking set is a different 100. Per-count tuned tables don't transfer.
4. **Never regress HEAD.** Always leave committed, 100/100, best runtime-adjusted config.
5. **Runtime is first-class.** At rt ≤ 0.305·median → hard 0.7 floor (30% cut, unbeatable). Above median → uncapped penalty. **Never trade quality for speed once at the floor.**
6. **Winning first, never rush, no busywork.** Every experiment needs a hypothesis. Don't ship partial/unverified work.

---

## Part II — Score Trajectory

| Tag | Local Score | Runtime avg | Δ vs prev | Key change | Verdict |
|-----|-------------|-------------|-----------|------------|---------|
| tuned52 (original) | 2.6326 | 1.03s | — | Per-count tuned shelf-SA | Overfit |
| sprint2_v6 | 2.5443 | 0.57s | −3.4% | Tuple conversion, refine extension, SA bound | ✅ |
| analytical_v6 | 2.5211 | 0.70s | −0.9% | Analytical refinement, generalized swaps | ✅ |
| quadratic_v1 | 2.4658 | 0.69s | −2.1% | Quadratic placement (CG solver) | ✅ |
| portfolio_v1 | 2.3977 | 6.96s | −2.8% | Parallel multi-start (6 configs + SA) | ❌ runtime regression |
| **sprint5_v9** | **2.7182** | **0.18s** | — | **Reverted portfolio; SA for n≥100 only** | **✅ Current best** |
| s6_p0_v4 | 2.4167 | 2.78s | — | Persistent pool + shelf-SA | ❌ runtime regression |
| s6_engine_v6 | 2.4542 | 1.54s | — | SA + correctness-first polish (1s) | ❌ too slow |
| s7_m4_hybrid | 2.7011 | 0.69s | — | SP-SA interior replacement | ❌ runtime penalty dominates |

**Key insight:** Every attempt to improve quality via portfolio/contour/skyline/polish added runtime that outweighed quality gains at median≤2.0s. The shelf packer with SA for n≥100 (sprint5_v9) remains the best committed result.

---

## Part III — P2 Milestones (Sequence-Pair Topological SA)

| Milestone | Gate | Result | Verdict |
|-----------|------|--------|---------|
| **M1: SP packer** | Valid non-overlapping on case 99 movable | 5/5 random SPs = 0 overlaps | ✅ PASS |
| **M2: SP-SA proof-of-concept** | Util > 0.80 on case 99 movable | **Util = 0.828** (7849 moves, 30s) | ✅ PASS |
| **M3: Preplaced obstacles** | Exact feasibility on all 21 big cases | 18/20 feasible, util 0.705 | ⚠️ Partial |
| **M4: Soft constraints** | V_rel ≤ 0.10, util > 0.70 | V_rel=0.712, util=0.667 | ❌ FAIL |
| **M4': Surgical hybrid** | Runtime-adjusted beats v9 | 2.7011 local, 0.69s avg | ❌ WORSE |
| **N1: Boundary-edge encoding** | Boundary blocks on required edge | 36/36 boundary OK, 0.665 util | ✅ PASS |
| **N2: Cluster + MIB + shape** | V_rel drops, util ≥ 0.70 | **Util = 0.737**, boundary 36/36, cluster=16, MIB=0 | ✅ PASS |

### M4 Pivot Finding (decisive)
The SP-SA **cannot survive soft constraints**. With soft penalties:
- V_rel = 0.712 (target ≤ 0.10) — massive structural violations
- Util dropped 0.705 → 0.667 — soft penalties fight area optimization
- Boundary (30% of blocks must touch bbox edge): SA places blocks randomly
- Cluster abutment: SA places blocks independently
- MIB equal shape: SA picks random shapes

### M4' Surgical Hybrid Finding
Replacing only the interior packer with SP-SA while keeping v9's boundary/cluster/MIB:
- Local score 2.7011 (slight improvement over v9's 2.7182)
- Runtime 0.69s (vs v9's 0.18s — 3.8× slower)
- Runtime-adjusted: **WORSE at every median** — runtime penalty dominates
- Root cause: SP-SA runs on ALL n≥80 cases even when it doesn't improve

---

## Part IV — Golden Mining Results

| Metric | Golden | Ours (shelf) | Gap |
|--------|--------|-------------|-----|
| Area utilization | **0.971** | 0.52–0.59 | **0.38–0.45** |
| Aspect ratio (median) | **1.45** | 1.00 | 0.45 |
| Aspect ratio (p90) | **2.50** | 1.00 | 1.50 |
| Aspect ratio (max) | **3.00** | 1.00 | 2.00 |
| Boundary blocks/case | 24 | ~24 | ~0 |
| Preplaced blocks/case | 2.6 | ~2.6 | ~0 |

**Key insight:** Golden uses aspect ratios up to 3:1 (median 1.45). We use 1:1. The evaluator does NOT check aspect ratio — only area. Shape is a free variable bounded only by our own choice.

---

## Part V — Complete Experiment Log (50+ experiments)

### Sprint 1–2 (Previous Agent)
| # | Approach | Score | Verdict |
|---|----------|-------|---------|
| 1 | Tuple conversion for ALL block counts | 2.5443 | ✅ |
| 2 | Extend refine passes to 80-99 band | 2.5443 | ✅ |
| 3 | Simplify 100+ refine paths | 2.5443 | ✅ |
| 4 | SA budget bounding (3s→1s) | 2.5443 | ✅ |
| 5 | Interior obstacles for 80+ | 2.5443 | ✅ |
| 6 | BFS ordering | — | ❌ |
| 7 | Multi-start SA | — | ❌ |
| 8 | Force-directed refinement | — | ❌ |
| 9 | Centroid sorting | — | ❌ |
| 10 | Real contest cost for variant selection | — | ❌ |
| 11 | Position swaps | — | ❌ |
| 12 | Two-axis compaction | — | ❌ |

### Sprint 3 (First agent pass)
| # | Approach | Score | Verdict |
|---|----------|-------|---------|
| 13 | Analytical global placement (CG) | 2.4658 | ✅ |
| 14 | Analytical-target refinement | 2.5211 | ✅ |
| 15 | Analytical ordering WITHOUT super-blocks | — | ❌ (25-36 soft violations) |
| 16 | Compaction toward origin | — | ❌ (boundary violations) |
| 17 | Generalized equal-shape swaps | 2.5218 | ✅ |
| 18 | Generalized boundary wire swaps | 2.5218 | ✅ |
| 19 | Aggressive analytical refinement | 2.5211 | ✅ |
| 20 | Real contest cost in SA | — | ≈ Neutral |
| 21 | Quadratic placement (CG) | 2.4658 | ✅ |
| 22 | QP + centroid relaxation hybrid | — | ❌ |
| 23 | Iterative QP + spreading | — | ❌ |
| 24 | Density-spread QP in analytical | — | ❌ |

### Sprint 4 (Portfolio era)
| # | Approach | Score | Verdict |
|---|----------|-------|---------|
| 25 | Parallel multi-start (6 configs + SA) | 2.3977 | ❌ runtime 6.96s |
| 26 | Persistent pool portfolio | 2.4167 | ❌ runtime 2.78s |
| 27 | De-overfit per-count tuning | 2.4167 | ✅ better transfer |
| 28–30 | Non-square shapes on shelf (1.2/1.5/2:1) | — | ❌ 15/100 feasible |
| 31 | Post-pack shape optimization | — | ≈ no-op |
| 32 | Post-pack compaction | — | ❌ breaks soft constraints |
| 33 | Abacus-style legalization | 2.4030 | ✅ in portfolio |
| 34 | Correctness-first polish | — | ❌ +0.16s for ~nil gain |

### Sprint 5 (Runtime-first)
| # | Approach | Score | Verdict |
|---|----------|-------|---------|
| 35 | **SA for n≥100 only** | **2.7182** | **✅ Current best** |
| 36 | SA budget reduction | 2.7182 | ≈ same |
| 37 | Fast local search (incremental HPWL) | — | ❌ 83 worsened |
| 38 | SA with relocation moves | 2.7182 | ≈ neutral |

### Sprint 6 (Packer + engine)
| # | Approach | Score | Verdict |
|---|----------|-------|---------|
| 40 | Skyline packer (contour/BLF) | 0.047 util | ❌ wide-empty |
| 41 | Variable-height shelf packer | 0.274 util | ❌ too many rows |
| 42 | Bbox-area scoring | 0.591 util | ⏸️ best standalone |
| 43 | Shape selection (score=th) | 0.048 util | ❌ picks 5:1 shapes |
| 44 | Skyline packer integration | 2.7177 | ❌ adds runtime |
| 45 | Correctness-first polish (0.2s) | 2.7182 | ≈ no improvement |
| 46 | Real contest cost in SA | 2.7182 | ≈ no improvement |

### Sprint 7 (M4' surgical hybrid)
| # | Approach | Score | Verdict |
|---|----------|-------|---------|
| 47 | SP-SA interior replacement (2s budget) | 2.7014 | ❌ runtime 1.34s |
| 48 | SP-SA interior replacement (0.5s budget) | 2.7011 | ❌ runtime 0.69s |

---

## Part VI — Consolidated Dead-Ends (do NOT retry)

1. BFS ordering
2. Multi-start SA as built
3. Force-directed refinement under hard overlap-reject
4. Centroid sorting
5. Equal-area/overlap-reject position swaps
6. Compaction-to-origin
7. Analytical x-ordering without cluster super-blocks (41–58 soft violations)
8. Linear QP without spreading
9. QP+relaxation hybrid
10. Real-cost in the tuned variant loop
11. 200 relaxation sweeps
12. Per-case process-pool creation
13. Portfolio of heavy-SA members (runtime blowup)
14. Density-spread QP only in losing analytical path
15. Incremental-HPWL local search built incrementally-first (broke, 83 worsened)
16. Aspect-ratio fitting bolted onto weak SA (no effect)
17. Sacrificing quality for sub-floor speed
18. Construction-only portfolio for quality (2.67 vs 2.72)
19. Non-square shapes on shelf packer (15/100 feasible)
20. Full-recompute polish at scale (too slow)
21. Skyline packer standalone (0.047–0.591 util, never hit 0.70)
22. Compaction (breaks soft constraints)
23. SA with ripple-repair (broke 3 cases)
24. Iterative QP + spreading (no improvement)
25. Aspect-ratio refinement (no effect)
26. Portfolio with 3+ configs (overhead > quality gain)
27. SA relocation moves (not accepted)
28. Analytical ordering for non-cluster blocks (hurt score)
29. SP-SA with soft constraints (V_rel=0.712 — SA can't satisfy structural constraints)
30. SP-SA interior replacement (runtime penalty dominates)
31. Correctness-first polish (too slow for quality gain)

---

## Part VII — Architecture Map

```
contest_solution/my_optimizer.py (4421 lines, 87+ methods)
├── solve()                         entry; shelf-SA for n≥100
├── _construct_layout()             degree-ordering shelf packer + refine + SA
│   ├── _choose_dimensions()        near-square for soft blocks
│   ├── _pack_interior_units()      shelf row packer (degree ordering)
│   ├── _place_boundary_items()     perimeter frame
│   ├── _refine_*()                 8 refinement passes
│   └── _sa_post_optimization()     SA with swap/shift moves
├── _skyline_pack()                 contour packer with shape selection (unused)
├── _correctness_first_polish()     greedy descent on true cost (wired in, ~no effect)
├── _true_contest_cost()            exact contest cost, feasibility-gated
├── _n_soft() / _is_feasible()      helpers
├── _analytical_global_placement()  quadratic placement (CG solver)
└── _analytical_construct_layout()  QP + contour + refinement (loses to shelf)

contest_solution/sequence_pair_sa.py (P2 module, dormant)
├── sp_pack()                       SP encode/decode + longest-path packing
├── sp_sa_movable_only()            SA over SP (M2: util 0.828)
├── sp_sa_with_obstacles()          SA with preplaced + soft constraints (M3/M4)
└── compute_soft_violations()       standalone soft violation counter

scripts/score_real.py               runtime-adjusted scoring tool
scripts/mine_golden.py              golden solution mining
scripts/analyze_results.py          per-case analysis
MASTER_PLAYBOOK.md                  strategy document
PLAN_EXECUTION_LOG.md               this file
```

---

## Part VIII — Constraint Reference

**Hard constraints** (violation → cost 10.0):
1. No overlaps (touching edges OK, tolerance 1e-6)
2. Soft-block area: `|w*h − target| / target ≤ 0.01` (symmetric, ±1%)
3. Fixed-shape blocks: exact (w,h) from input (tolerance 1e-4)
4. Preplaced blocks: exact (x,y,w,h) from input (tolerance 1e-4)

**Soft constraints** (penalized via `exp(2·V_rel)`):
1. **Boundary:** block must touch specified bbox edge/corner (bitmask: 1=left, 2=right, 4=top, 8=bottom)
2. **Grouping:** blocks in same cluster must abut (share edge)
3. **MIB:** blocks in same MIB group must have identical (w,h)

**N_soft** = (#boundary blocks) + Σ(|cluster_group|−1) + Σ(|mib_group|−1)

---

## Part IX — Key Files

| File | Purpose |
|------|---------|
| `contest_solution/my_optimizer.py` | Optimizer (4421 lines) |
| `contest_solution/sequence_pair_sa.py` | SP-SA module (dormant) |
| `results/sprint5_v9.json` | Current best (2.7182) |
| `results/quadratic_v1.json` | Baseline (2.4658) |
| `results/_baselines.json` | Golden hpwl/area for all 100 cases |
| `scripts/score_real.py` | Runtime-adjusted scoring |
| `scripts/mine_golden.py` | Golden solution mining |
| `MASTER_PLAYBOOK.md` | Strategy document |

---

## Part X — Strategic Assessment

### What we know
1. **Shelf packer is the ceiling** — every attempt to improve quality via portfolio/contour/skyline/polish either broke feasibility or added too much runtime.
2. **SP-SA works on movable-only** (0.828 util) but **collapses with soft constraints** (V_rel=0.712).
3. **Golden achieves 97% util** with aspect ratios up to 3:1 — the shape lever is real but the shelf packer can't exploit it.
4. **Runtime floor is achievable** (0.18s avg) — being fast is free quality on the leaderboard.
5. **Quality gap is huge** — we're ~2× golden on both wirelength and area.

### What's needed to win
1. **A packer that can exploit shape diversity** — either B*-tree/sequence-pair SA (P2) or a skyline packer that actually works with obstacles/soft constraints.
2. **Constraint-aware moves** in the SA — reshape-to-touch-boundary, abutment-repair, MIB-equalize.
3. **Either of the above is a multi-day build** with significant research risk.

### What NOT to do
1. Don't keep tweaking the shelf packer — it's structurally capped.
2. Don't keep trying portfolio diversity — construction-only members don't help quality.
3. Don't keep trying incremental polish — blocks are too tightly packed for centroid moves.
4. Don't sacrifice runtime for quality — being at the floor is free quality.

### Decision point for the operator
The playbook says: "If M2 fails (SA can't exceed ~0.70 util): escalate — the remaining bet is IV.P3 (learned placement)." M2 passed (0.828), but M4 failed (V_rel=0.712). The quality win does NOT survive the constraints. The remaining options are:
1. **P2 proper** — full B*-tree/sequence-pair SA with constraint-aware moves (complex, high-effort, high-risk)
2. **P3.2** — learned seeding via GNN (the contest's stated theme, A100 available)
3. **Accept v9 ceiling** and optimize runtime for the 0.7 floor bonus
