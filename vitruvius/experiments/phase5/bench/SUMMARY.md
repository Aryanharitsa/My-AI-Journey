# Phase 3 — 3×3 encoder × BEIR sweep

Primary metric: nDCG@10 (from-scratch `retrieval_metrics.evaluate`).
References from handoff §3.4 (approximate BEIR leaderboard).
Tolerance: ±0.03 per cell.

## Measured nDCG@10 (delta vs reference in parentheses)

| Encoder \ Dataset | nfcorpus | scifact | fiqa |
|---|---|---|---|
| `lstm-retriever` | 0.1901 (no ref) | 0.3606 (no ref) | 0.0886 (no ref) |
| `conv-retriever` | 0.1400 (no ref) | 0.1978 (no ref) | 0.0370 (no ref) |
| `mamba-retriever-fs` | 0.2083 (no ref) | 0.3752 (no ref) | 0.0863 (no ref) |

## Per-cell breakdown

- `lstm-retriever` × `nfcorpus`: measured 0.1901 (no leaderboard reference). pytrec_eval nDCG@10=0.1897 (|Δ|=4.83e-04).
- `lstm-retriever` × `scifact`: measured 0.3606 (no leaderboard reference). pytrec_eval nDCG@10=0.3606 (|Δ|=0.00e+00).
- `lstm-retriever` × `fiqa`: measured 0.0886 (no leaderboard reference). pytrec_eval nDCG@10=0.0886 (|Δ|=0.00e+00).
- `conv-retriever` × `nfcorpus`: measured 0.1400 (no leaderboard reference). pytrec_eval nDCG@10=0.1389 (|Δ|=1.09e-03).
- `conv-retriever` × `scifact`: measured 0.1978 (no leaderboard reference). pytrec_eval nDCG@10=0.1978 (|Δ|=0.00e+00).
- `conv-retriever` × `fiqa`: measured 0.0370 (no leaderboard reference). pytrec_eval nDCG@10=0.0370 (|Δ|=0.00e+00).
- `mamba-retriever-fs` × `nfcorpus`: measured 0.2083 (no leaderboard reference). pytrec_eval nDCG@10=0.2072 (|Δ|=1.04e-03).
- `mamba-retriever-fs` × `scifact`: measured 0.3752 (no leaderboard reference). pytrec_eval nDCG@10=0.3752 (|Δ|=0.00e+00).
- `mamba-retriever-fs` × `fiqa`: measured 0.0863 (no leaderboard reference). pytrec_eval nDCG@10=0.0863 (|Δ|=0.00e+00).

## Cross-check status (from-scratch vs pytrec_eval)

| Cell | max|Δ| across nDCG@{1,5,10,100} | Recall@k bit-exact? |
|---|---:|:---:|
| `lstm-retriever` × `nfcorpus` | 5.68e-03 | ✅ |
| `lstm-retriever` × `scifact` | 0.00e+00 | ✅ |
| `lstm-retriever` × `fiqa` | 0.00e+00 | ✅ |
| `conv-retriever` × `nfcorpus` | 2.98e-03 | ✅ |
| `conv-retriever` × `scifact` | 0.00e+00 | ✅ |
| `conv-retriever` × `fiqa` | 0.00e+00 | ✅ |
| `mamba-retriever-fs` × `nfcorpus` | 5.16e-03 | ✅ |
| `mamba-retriever-fs` × `scifact` | 0.00e+00 | ✅ |
| `mamba-retriever-fs` × `fiqa` | 0.00e+00 | ✅ |

## Runtime (seconds, total per cell)

| Encoder \ Dataset | nfcorpus | scifact | fiqa |
|---|---|---|---|
| `lstm-retriever` | 4.11 | 5.23 | 39.24 |
| `conv-retriever` | 3.65 | 5.04 | 36.62 |
| `mamba-retriever-fs` | 67.65 | 7.72 | 58.23 |
