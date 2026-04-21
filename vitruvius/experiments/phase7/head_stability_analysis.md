# Phase 7 — Cross-dataset head-importance stability

Spearman rank correlation between per-head importance vectors (flattened across (layer, head)) across the 3 BEIR subsets. High ρ means the same heads matter on different datasets — evidence for a **universal retrieval-essential head set** per encoder that a structured-pruning recipe could target. Low ρ means important heads are domain-specific.

| Encoder | ρ(nfcorpus,scifact) | ρ(nfcorpus,fiqa) | ρ(scifact,fiqa) | Mean ρ |
|---|---:|---:|---:|---:|
| `minilm-l6-v2` | 0.2732 | 0.3066 | -0.0375 | **0.1808** |
| `bert-base` | 0.2194 | 0.1739 | 0.2795 | **0.2243** |
| `gte-small` | 0.1104 | — | — | **0.1104** |

## Interpretation

- `minilm-l6-v2` (mean ρ = 0.1808): **low** cross-dataset stability — head-importance is largely domain-specific; a universal pruning recipe would under-serve at least one dataset.
- `bert-base` (mean ρ = 0.2243): **low** cross-dataset stability — head-importance is largely domain-specific; a universal pruning recipe would under-serve at least one dataset.
- `gte-small` (mean ρ = 0.1104): **low** cross-dataset stability — head-importance is largely domain-specific; a universal pruning recipe would under-serve at least one dataset.

