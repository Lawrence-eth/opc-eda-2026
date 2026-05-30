# FloorSet ICCAD-2026 — Plan Execution Log

> **Purpose:** Every experiment, every decision, every dead-end — recorded so no future agent re-derives or re-tries. Updated after every meaningful change. Read this before touching code.

---

## HEAD Result (verified, internally consistent)

| Field | Value |
|-------|-------|
| **Best committed** | `sprint5_v9` = **2.7182** (local RF=1.0) |
| **Feasible** | 100/100 |
| **Runtime sum / max** | 18.1s / 0.9s |
| **Runtime-adjusted (median=1.0s)** | **2.135** (beats quadratic_v1 baseline 2.652) |
| **Runtime-adjusted (median=2.0s)** | **1.932** (beats baseline 2.160) |
| **Runtime-adjusted (median=3.0s)** | 1.903 (baseline 1.941 — we win) |
| **Runtime-adjusted (median=5.0s)** | 1.903 (baseline 1.795 — baseline wins) |
| **SP-SA M4 util** | 0.667 mean (18/20 feasible, V_rel=0.712) |

**Score formula:**
```
cost_i = (1 + 0.5·max(0,hpwl_gap) + 0.5·max(0,area_gap)) · exp(2·V_rel) · max(0.7, (rt/median)^0.3)
Total  = Σ cost_i · exp(n_i/12) / Σ exp(n_j/12)
```

---

## Score Trajectory (all verified, internally consistent)

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
| s6_engine_v9 | 2.4664 | 0.74s | — | SA n≥100 + polish (0.1s) | ❌ still too slow |
| s6_engine_v15 | 2.7182 | 0.18s | — | Polish-only (no refinement) | ❌ no improvement |

**Key insight:** Every attempt to improve quality via portfolio/contour/skyline/polish added runtime that outweighed quality gains at median≤2.0s. The shelf packer with SA for n≥100 (sprint5_v9) remains the best committed result.

---

## P2 Milestones (Sequence-Pair Topological SA)

| Milestone | Gate | Result | Verdict |
|-----------|------|--------|---------|
| **M1: SP packer** | Valid non-overlapping on case 99 movable | 5/5 random SPs = 0 overlaps | ✅ PASS |
| **M2: SP-SA proof-of-concept** | Util > 0.80 on case 99 movable | **Util = 0.828** (7849 moves, 30s) | ✅ PASS |
| **M3: Preplaced obstacles** | Exact feasibility on all 21 big cases | 18/20 feasible, util 0.705 | ⚠️ Partial (2 cases infeasible) |
| **M4: Soft constraints** | V_rel ≤ 0.10, util > 0.70 | V_rel=0.712, util=0.667 | ❌ FAIL |
| **M4': Surgical hybrid** | Runtime-adjusted beats v9 | 2.7011 local (vs 2.7182), 0.69s avg | ❌ WORSE — SP-SA adds runtime without quality |
| M5: Speed + numba | Not started | — | — |
| M6: Integrate + gate | Not started | — | — |

### M4 Pivot Finding (decisive)

The SP-SA **cannot survive soft constraints**. With soft penalties in the cost:
- **V_rel = 0.712** (target ≤ 0.10) — every case has massive soft violations
- **Util dropped 0.705 → 0.667** — soft penalties fight area optimization
- **Boundary (30% of blocks must touch bbox edge):** SA places blocks randomly, no boundary awareness
- **Cluster abutment:** SA places blocks independently, no abutment logic
- **MIB equal shape:** SA picks random shapes per block

**Root cause:** the SP-SA has no move type that can satisfy soft constraints. Reshaping blocks to touch edges, abutting cluster members, equalizing MIB shapes all require **constraint-aware moves** that don't exist in the SA.

**Decision point:** M4 failed. Either:
1. Build constraint-aware SA moves (complex, high-risk)
2. Accept shelf packer ceiling (2.7182 local / ~1.9 at median=3s)
3. Pivot to ML route (P3.2 learned seeding)

---

## Golden Mining Results

| Metric | Golden | Ours (shelf) | Gap |
|--------|--------|-------------|-----|
| Area utilization | 0.971 | 0.52–0.59 | **0.38–0.45** |
| Aspect ratio (median) | 1.45 | 1.00 | 0.45 |
| Aspect ratio (p90) | 2.50 | 1.00 | 1.50 |
| Aspect ratio (max) | 3.00 | 1.00 | 2.00 |
| Boundary blocks/case | 24 | ~24 | ~0 |
| Preplaced blocks/case | 2.6 | ~2.6 | ~0 |

**Key insight:** Golden uses aspect ratios up to 3:1 (median 1.45). We use 1:1. The evaluator does NOT check aspect ratio — only area. Shape is a free variable.

---

## What Was Tried — Complete Experiment Log (46 experiments)

### Sprint 1–2 (Previous Agent)

| # | Approach | Score | Verdict | Why |
|---|----------|-------|---------|-----|
| 1 | Tuple conversion for ALL block counts | 2.5443 | ✅ | 80-99 band iterating torch tensors |
| 2 | Extend refine passes to 80-99 band | 2.5443 | ✅ | 80-99 had cost 5-9 |
| 3 | Simplify 100+ refine paths | 2.5443 | ✅ | All 100+ cases get full treatment |
| 4 | SA budget bounding (3s→1s) | 2.5443 | ✅ | Reduced runtime |
| 5 | Interior obstacles for 80+ | 2.5443 | ✅ | Pack around preplaced |
| 6 | BFS ordering | — | ❌ | Disrupted degree ordering |
| 7 | Multi-start SA | — | ❌ | Same convergence, 2× runtime |
| 8 | Force-directed refinement | — | ❌ | Blocks can't move (overlap rejection) |
| 9 | Centroid sorting | — | ❌ | Shelf row structure dominates |
| 10 | Real contest cost for variant selection | — | ❌ | Tuned params for proxy cost |
| 11 | Position swaps | — | ❌ | Different-dimension blocks |
| 12 | Two-axis compaction | — | ❌ | Created overlaps |

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
| 22 | QP + centroid relaxation hybrid | — | ❌ (CG disrupted) |
| 23 | Iterative QP + spreading | — | ❌ (no improvement) |
| 24 | Density-spread QP in analytical | — | ❌ (doesn't affect shelf) |

### Sprint 4 (Portfolio era)

| # | Approach | Score | Verdict |
|---|----------|-------|---------|
| 25 | Parallel multi-start (6 configs + SA) | 2.3977 | ❌ runtime 6.96s |
| 26 | Persistent pool portfolio | 2.4167 | ❌ runtime 2.78s |
| 27 | De-overfit per-count tuning | 2.4167 | ✅ better transfer |
| 28 | 1.2:1 aspect ratio on shelf | 2.7540 | ❌ worse |
| 29 | 1.5:1 aspect ratio on shelf | — | ❌ 15/100 feasible |
| 30 | 2:1 aspect ratio on shelf | — | ❌ 16/100 feasible |
| 31 | Post-pack shape optimization | — | ≈ no-op (blocks within ±1%) |
| 32 | Post-pack compaction | — | ❌ breaks soft constraints |
| 33 | Abacus-style legalization | 2.4030 | ✅ in portfolio |
| 34 | Correctness-first polish (full) | — | ❌ +0.16s for ~nil gain |

### Sprint 5 (Runtime-first)

| # | Approach | Score | Verdict |
|---|----------|-------|---------|
| 35 | **SA for n≥100 only** | **2.7182** | **✅ Current best** |
| 36 | SA budget reduction | 2.7182 | ≈ same |
| 37 | Fast local search (incremental HPWL) | — | ❌ 83 worsened |
| 38 | SA with relocation moves | 2.7182 | ≈ neutral |
| 39 | Increased SA budget | 2.7182 | ≈ neutral |

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

---

## Dead-Ends (do NOT retry)

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
25. Aspect-ratio refinement (no effect — blocks already within ±1%)
26. Portfolio with 3+ configs (overhead > quality gain)
27. SA relocation moves (not accepted — creates overlaps)
28. Analytical ordering for non-cluster blocks (hurt score)
29. **SP-SA with soft constraints (V_rel=0.712 — SA can't satisfy structural soft constraints)**

---

## Architecture Map

```
contest_solution/my_optimizer.py (4413 lines, 87+ methods)
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
```

---

## Constraint Reference

**Hard constraints** (violation → cost 10.0):
1. No overlaps (touching edges OK, tolerance 1e-6)
2. Soft-block area: `|w*h − target| / target ≤ 0.01` (symmetric, ±1%)
3. Fixed-shape blocks: exact (w,h) from input (tolerance 1e-4)
4. Preplaced blocks: exact (x,y,w,h) from input (tolerance 1e-4)

**Soft constraints** (penalized via `exp(2·V_rel)`):
1. **Boundary:** block must touch specified bbox edge/corner (bitmask: 1=left, 2=right, 4=top, 8=bottom)
2. **Grouping:** blocks in same cluster must abut (share edge)
3. **MIB:** blocks in same MIB group must have identical (w,h)

---

## Key Files

| File | Purpose |
|------|---------|
| `contest_solution/my_optimizer.py` | Optimizer (4413 lines) |
| `contest_solution/sequence_pair_sa.py` | SP-SA module (dormant) |
| `results/sprint5_v9.json` | Current best (2.7182) |
| `results/quadratic_v1.json` | Baseline |
| `results/_baselines.json` | Golden hpwl/area for all 100 cases |
| `scripts/score_real.py` | Runtime-adjusted scoring |
| `scripts/mine_golden.py` | Golden solution mining |
| `MASTER_PLAYBOOK.md` | Strategy document |
| `PLAN_EXECUTION_LOG.md` | This file |
