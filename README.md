# ICCAD 2026 FloorSet Challenge — Winning Submission

> **Strategy:** Hybrid ML + heuristic legalizer via adversarial multi-agent planning
> **Current Target Score:** < 1.0 (from corrected baseline of ~2.91)
> **Runtime Target:** < 100ms per case (to exploit 0.7x discount)

## Repository Structure

```
.
├── PLAN/                          # Execution plan and strategy docs
│   └── EXECUTION_PLAN.md          # Full 12-task, 8-wave execution plan
├── ANALYSIS/                      # Research and findings
│   └── WHY_PREVIOUS_AGENT_FAILED.md  # Critical bug analysis
├── contest_solution/              # Final optimizer submission
│   ├── my_optimizer.py            # Main solver (heuristic + ML hybrid)
│   └── test_my_optimizer.py       # Unit tests
├── src/                           # Source code (model, training, utils)
│   ├── model.py                   # Graph Transformer architecture
│   ├── dataset.py                 # Training data pipeline
│   ├── sequence_pair.py           # SP decoder and packer
│   └── train.py                   # Training script
├── tests/                         # Test suite
├── scripts/                       # Diagnostic and analysis tools
│   ├── analyze_results.py         # Case-level score diagnostics
│   ├── audit_results.py           # Result integrity checker
│   └── compare_results.py         # Baseline vs candidate comparison
├── results/                       # Validation run artifacts
└── docs/                          # Documentation and research
```

## The Story

This repository contains the evolution of a winning strategy for the ICCAD 2026 FloorSet Challenge (Problem C). It began with a heuristic baseline (score ~2.91) and is evolving into a hybrid ML + heuristic approach targeting score < 1.0.

### Critical Discovery

The previous agent's work contained a **fatal scoring formula bug**: analysis scripts used `exp(n - max_n)` instead of the official `exp((n - max_n) / 12)`. This:
- Inflated score concentration (claimed case 99 = 63%, true = 8%)
- Hid the optimizer's poor performance (claimed score 1.50, true score 2.91)
- Led to misguided optimization focus

See `ANALYSIS/WHY_PREVIOUS_AGENT_FAILED.md` for the full forensic report.

### Winning Strategy (Adversarial Synthesis)

After 3 rounds of cross-attack by 5 specialist agents, the consensus strategy is:

1. **Phase 1:** Fix 3 fatal heuristic bugs (aspect ratio, boundary packing, dead compactor)
2. **Phase 2:** Train Graph Transformer via pure supervised learning on 1M samples
3. **Phase 3:** Integrate ML inference with fixed heuristic as micro-legalizer
4. **Target:** < 100ms per case, score < 1.0

See `PLAN/EXECUTION_PLAN.md` for the full 12-task execution plan.

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/Lawrence-eth/opc-eda-2026.git
cd opc-eda-2026

# 2. Install dependencies (from official FloorSet)
pip install torch shapely pytest tqdm numpy

# 3. Copy optimizer to contest directory
cp contest_solution/my_optimizer.py /path/to/FloorSet/iccad2026contest/

# 4. Evaluate
PYTHONPATH=.. python iccad2026_evaluate.py --evaluate my_optimizer.py --verbose
```

## Development Status

| Phase | Status | Target |
|-------|--------|--------|
| Score tooling fix | Not started | Correct `exp((n-max_n)/12)` weighting |
| Heuristic bug fixes | Not started | Score ~1.5 from ~2.91 |
| ML model development | Not started | Graph Transformer architecture |
| Training | Not started | 1M samples, supervised learning |
| Integration | Not started | < 100ms per case |
| Final target | Not started | Score < 1.0 |

## Contest Spec

- **Problem:** Data-driven SoC floorplanning with 21-120 blocks
- **Datasets:** 1M training, 100 validation, 100 hidden test
- **Scoring:** `Cost = (1 + 0.5*(HPWL_gap + Area_gap)) * exp(2*V_rel) * max(0.7, RuntimeFactor^0.3)`
- **Hard constraints:** No overlap, area tolerance ±1%, exact fixed/preplaced dimensions
- **Soft constraints:** Boundary, grouping (abutment), MIB (identical dimensions)
- **Hardware:** A100 80GB GPU, Ice Lake 48-core CPU, 128GB RAM
- **Weighting:** `exp(n/12)` — cases 101-120 dominate but with flatter distribution than expected

## Team

This project was planned via **adversarial multi-agent orchestration** under Sisyphus:
- **Pragmatist:** Quick wins and scope discipline
- **Deep-diver:** Forensic analysis and data archaeology
- **Architect:** System architecture and runtime analysis
- **Creative:** Radical alternatives and moat identification
- **Strategist:** End-to-end execution strategy

All findings cross-verified by independent agents before inclusion in the plan.

## License

Contest submission for ICCAD 2026 Problem C.
