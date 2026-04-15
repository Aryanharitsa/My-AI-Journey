# Golden Datasets — Analysis

## Evolution
Gold-10 evolved through 4 versions to calibrate difficulty against the solver's actual performance.

| Version | Problems | Baseline Score | Issue |
|---|---|---|---|
| v1 | 10 (AIME 2026 #11-15) | 10/10 | Too easy — model aces all AIME-hard |
| v2 | 10 (mixed AIME + custom) | 9/10 | Still too easy — only P10 (3×n tiling) failed |
| v3 | 10 (failure-class targeted) | 8/10 | Good calibration — P1 (generation miss) + P10 (attractor) |
| v4 | 7 (discriminative focus) | 5/7 | Well-calibrated — P4 (generation miss) + P5 (attractor) |

## Key Insight
AIME-hard problems are trivial for GPT-OSS-120B with TIR. The model scores ~97.9% on AIME 2025 with tools. True discrimination requires problems that exploit specific failure modes — attractor traps, generation-miss walls, long-horizon case analysis.

## Problem Design Principles
1. Expected baseline 2-5/8 correct (the useful zone for measuring interventions)
2. Post-cutoff sources (no training contamination)
3. All failure classes represented
4. Mix of answer magnitudes (avoid clustering)
