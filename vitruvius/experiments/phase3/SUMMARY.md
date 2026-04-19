# Phase 3 — 3×3 encoder × BEIR sweep

Primary metric: nDCG@10 (from-scratch `retrieval_metrics.evaluate`).
References from handoff §3.4 (approximate BEIR leaderboard).
Tolerance: ±0.03 per cell.

## Measured nDCG@10 (delta vs reference in parentheses)

| Encoder \ Dataset | nfcorpus | scifact | fiqa |
|---|---|---|---|
| `minilm-l6-v2` | 0.3165 (Δ+0.0165 vs 0.30) ✅ | 0.6451 (Δ+0.0051 vs 0.64) ✅ | 0.3687 (Δ+0.0087 vs 0.36) ✅ |
| `bert-base` | 0.3169 (Δ+0.0069 vs 0.31) ✅ | 0.6082 (Δ-0.0718 vs 0.68) ❌ | 0.3229 (Δ+0.0229 vs 0.30) ✅ |
| `gte-small` | 0.3492 (Δ+0.0092 vs 0.34) ✅ | 0.7269 (Δ-0.0031 vs 0.73) ✅ | 0.3937 (Δ-0.0263 vs 0.42) ✅ |

## Per-cell breakdown

- `minilm-l6-v2` × `nfcorpus`: measured 0.3165, ref 0.30, delta +0.0165, in-band. pytrec_eval nDCG@10=0.3159 (|Δ|=5.72e-04).
- `minilm-l6-v2` × `scifact`: measured 0.6451, ref 0.64, delta +0.0051, in-band. pytrec_eval nDCG@10=0.6451 (|Δ|=0.00e+00).
- `minilm-l6-v2` × `fiqa`: measured 0.3687, ref 0.36, delta +0.0087, in-band. pytrec_eval nDCG@10=0.3687 (|Δ|=0.00e+00).
- `bert-base` × `nfcorpus`: measured 0.3169, ref 0.31, delta +0.0069, in-band. pytrec_eval nDCG@10=0.3151 (|Δ|=1.81e-03).
- `bert-base` × `scifact`: measured 0.6082, ref 0.68, delta -0.0718, **OUT OF BAND**. pytrec_eval nDCG@10=0.6082 (|Δ|=0.00e+00).
- `bert-base` × `fiqa`: measured 0.3229, ref 0.30, delta +0.0229, in-band. pytrec_eval nDCG@10=0.3229 (|Δ|=0.00e+00).
- `gte-small` × `nfcorpus`: measured 0.3492, ref 0.34, delta +0.0092, in-band. pytrec_eval nDCG@10=0.3480 (|Δ|=1.17e-03).
- `gte-small` × `scifact`: measured 0.7269, ref 0.73, delta -0.0031, in-band. pytrec_eval nDCG@10=0.7269 (|Δ|=0.00e+00).
- `gte-small` × `fiqa`: measured 0.3937, ref 0.42, delta -0.0263, in-band. pytrec_eval nDCG@10=0.3937 (|Δ|=0.00e+00).

## Cross-check status (from-scratch vs pytrec_eval)

| Cell | max|Δ| across nDCG@{1,5,10,100} | Recall@k bit-exact? |
|---|---:|:---:|
| `minilm-l6-v2` × `nfcorpus` | 6.71e-03 | ✅ |
| `minilm-l6-v2` × `scifact` | 0.00e+00 | ✅ |
| `minilm-l6-v2` × `fiqa` | 0.00e+00 | ✅ |
| `bert-base` × `nfcorpus` | 6.19e-03 | ✅ |
| `bert-base` × `scifact` | 0.00e+00 | ✅ |
| `bert-base` × `fiqa` | 0.00e+00 | ✅ |
| `gte-small` × `nfcorpus` | 6.27e-03 | ✅ |
| `gte-small` × `scifact` | 0.00e+00 | ✅ |
| `gte-small` × `fiqa` | 0.00e+00 | ✅ |

## Runtime (seconds, total per cell)

| Encoder \ Dataset | nfcorpus | scifact | fiqa |
|---|---|---|---|
| `minilm-l6-v2` | 5.89 | 7.54 | 55.44 |
| `bert-base` | 21.21 | 28.42 | 197.44 |
| `gte-small` | 5.86 | 7.65 | 52.09 |
