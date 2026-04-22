# Phase 8 — Token-position shuffle & position sensitivity (90% milestone)

What this phase measures: for each of the 6 Pareto encoders × 3 BEIR test subsets × 3 shuffle modes,
re-encode docs and/or queries after permuting content-token positions (special tokens stay put).
Position sensitivity score = `(baseline_nDCG@10 − shuffled_nDCG@10) / baseline_nDCG@10`.
Higher = more sequentially-position-dependent; 0.0 = bag-of-concepts; 1.0 = full collapse.

Baselines are pulled from Phase 3 (transformers) and Phase 5 (from-scratch) — not re-run.
Shuffle seed 1729 is identical across all encoders for cross-encoder fairness.

## Headline — mean position sensitivity by architecture family (over 3 datasets × encoders in family)

| Family | docs-shuffled | queries-shuffled | both-shuffled |
|---|---:|---:|---:|
| transformer | 0.211 | 0.104 | 0.268 |
| recurrent | 0.166 | 0.064 | 0.224 |
| convolutional | 0.314 | 0.129 | 0.346 |
| ssm | 0.243 | 0.098 | 0.263 |

## Full table — position sensitivity per (encoder × dataset × mode)

Each cell is `(baseline − shuffled) / baseline` at nDCG@10.

### docs-shuffled

| Encoder \ Dataset | nfcorpus | scifact | fiqa |
|---|:---:|:---:|:---:|
| `minilm-l6-v2` | 0.107 | 0.062 | 0.490 |
| `bert-base` | 0.131 | 0.050 | 0.325 |
| `gte-small` | 0.167 | 0.113 | 0.451 |
| `lstm-retriever` | 0.126 | 0.130 | 0.244 |
| `conv-retriever` | 0.255 | 0.245 | 0.441 |
| `mamba-retriever-fs` | 0.150 | 0.277 | 0.303 |

### queries-shuffled

| Encoder \ Dataset | nfcorpus | scifact | fiqa |
|---|:---:|:---:|:---:|
| `minilm-l6-v2` | 0.039 | 0.069 | 0.235 |
| `bert-base` | 0.022 | 0.111 | 0.075 |
| `gte-small` | 0.073 | 0.122 | 0.186 |
| `lstm-retriever` | -0.005 | 0.047 | 0.150 |
| `conv-retriever` | 0.071 | 0.093 | 0.222 |
| `mamba-retriever-fs` | 0.047 | 0.112 | 0.135 |

### both-shuffled

| Encoder \ Dataset | nfcorpus | scifact | fiqa |
|---|:---:|:---:|:---:|
| `minilm-l6-v2` | 0.149 | 0.091 | 0.539 |
| `bert-base` | 0.150 | 0.167 | 0.379 |
| `gte-small` | 0.214 | 0.197 | 0.524 |
| `lstm-retriever` | 0.143 | 0.165 | 0.365 |
| `conv-retriever` | 0.292 | 0.251 | 0.494 |
| `mamba-retriever-fs` | 0.177 | 0.281 | 0.329 |

## Per-cell baseline vs shuffled (nDCG@10)

Sanity audit: every row shows the raw nDCG@10 that produced the sensitivity number.

| Encoder × Dataset | mode | baseline | shuffled | sensitivity |
|---|---|---:|---:|---:|
| `minilm-l6-v2` × `nfcorpus` | docs-shuffled | 0.3165 | 0.2825 | 0.107 |
| `minilm-l6-v2` × `nfcorpus` | queries-shuffled | 0.3165 | 0.3043 | 0.039 |
| `minilm-l6-v2` × `nfcorpus` | both-shuffled | 0.3165 | 0.2693 | 0.149 |
| `minilm-l6-v2` × `scifact` | docs-shuffled | 0.6451 | 0.6052 | 0.062 |
| `minilm-l6-v2` × `scifact` | queries-shuffled | 0.6451 | 0.6005 | 0.069 |
| `minilm-l6-v2` × `scifact` | both-shuffled | 0.6451 | 0.5864 | 0.091 |
| `minilm-l6-v2` × `fiqa` | docs-shuffled | 0.3687 | 0.1882 | 0.490 |
| `minilm-l6-v2` × `fiqa` | queries-shuffled | 0.3687 | 0.2821 | 0.235 |
| `minilm-l6-v2` × `fiqa` | both-shuffled | 0.3687 | 0.1699 | 0.539 |
| `bert-base` × `nfcorpus` | docs-shuffled | 0.3169 | 0.2754 | 0.131 |
| `bert-base` × `nfcorpus` | queries-shuffled | 0.3169 | 0.3098 | 0.022 |
| `bert-base` × `nfcorpus` | both-shuffled | 0.3169 | 0.2695 | 0.150 |
| `bert-base` × `scifact` | docs-shuffled | 0.6082 | 0.5780 | 0.050 |
| `bert-base` × `scifact` | queries-shuffled | 0.6082 | 0.5406 | 0.111 |
| `bert-base` × `scifact` | both-shuffled | 0.6082 | 0.5065 | 0.167 |
| `bert-base` × `fiqa` | docs-shuffled | 0.3229 | 0.2180 | 0.325 |
| `bert-base` × `fiqa` | queries-shuffled | 0.3229 | 0.2985 | 0.075 |
| `bert-base` × `fiqa` | both-shuffled | 0.3229 | 0.2006 | 0.379 |
| `gte-small` × `nfcorpus` | docs-shuffled | 0.3492 | 0.2909 | 0.167 |
| `gte-small` × `nfcorpus` | queries-shuffled | 0.3492 | 0.3238 | 0.073 |
| `gte-small` × `nfcorpus` | both-shuffled | 0.3492 | 0.2745 | 0.214 |
| `gte-small` × `scifact` | docs-shuffled | 0.7269 | 0.6447 | 0.113 |
| `gte-small` × `scifact` | queries-shuffled | 0.7269 | 0.6379 | 0.122 |
| `gte-small` × `scifact` | both-shuffled | 0.7269 | 0.5835 | 0.197 |
| `gte-small` × `fiqa` | docs-shuffled | 0.3937 | 0.2160 | 0.451 |
| `gte-small` × `fiqa` | queries-shuffled | 0.3937 | 0.3204 | 0.186 |
| `gte-small` × `fiqa` | both-shuffled | 0.3937 | 0.1876 | 0.524 |
| `lstm-retriever` × `nfcorpus` | docs-shuffled | 0.1901 | 0.1663 | 0.126 |
| `lstm-retriever` × `nfcorpus` | queries-shuffled | 0.1901 | 0.1910 | -0.005 |
| `lstm-retriever` × `nfcorpus` | both-shuffled | 0.1901 | 0.1630 | 0.143 |
| `lstm-retriever` × `scifact` | docs-shuffled | 0.3606 | 0.3138 | 0.130 |
| `lstm-retriever` × `scifact` | queries-shuffled | 0.3606 | 0.3438 | 0.047 |
| `lstm-retriever` × `scifact` | both-shuffled | 0.3606 | 0.3012 | 0.165 |
| `lstm-retriever` × `fiqa` | docs-shuffled | 0.0886 | 0.0670 | 0.244 |
| `lstm-retriever` × `fiqa` | queries-shuffled | 0.0886 | 0.0752 | 0.150 |
| `lstm-retriever` × `fiqa` | both-shuffled | 0.0886 | 0.0563 | 0.365 |
| `conv-retriever` × `nfcorpus` | docs-shuffled | 0.1400 | 0.1042 | 0.255 |
| `conv-retriever` × `nfcorpus` | queries-shuffled | 0.1400 | 0.1301 | 0.071 |
| `conv-retriever` × `nfcorpus` | both-shuffled | 0.1400 | 0.0990 | 0.292 |
| `conv-retriever` × `scifact` | docs-shuffled | 0.1978 | 0.1494 | 0.245 |
| `conv-retriever` × `scifact` | queries-shuffled | 0.1978 | 0.1793 | 0.093 |
| `conv-retriever` × `scifact` | both-shuffled | 0.1978 | 0.1481 | 0.251 |
| `conv-retriever` × `fiqa` | docs-shuffled | 0.0370 | 0.0207 | 0.441 |
| `conv-retriever` × `fiqa` | queries-shuffled | 0.0370 | 0.0287 | 0.222 |
| `conv-retriever` × `fiqa` | both-shuffled | 0.0370 | 0.0187 | 0.494 |
| `mamba-retriever-fs` × `nfcorpus` | docs-shuffled | 0.2083 | 0.1770 | 0.150 |
| `mamba-retriever-fs` × `nfcorpus` | queries-shuffled | 0.2083 | 0.1986 | 0.047 |
| `mamba-retriever-fs` × `nfcorpus` | both-shuffled | 0.2083 | 0.1713 | 0.177 |
| `mamba-retriever-fs` × `scifact` | docs-shuffled | 0.3752 | 0.2713 | 0.277 |
| `mamba-retriever-fs` × `scifact` | queries-shuffled | 0.3752 | 0.3330 | 0.112 |
| `mamba-retriever-fs` × `scifact` | both-shuffled | 0.3752 | 0.2696 | 0.281 |
| `mamba-retriever-fs` × `fiqa` | docs-shuffled | 0.0863 | 0.0602 | 0.303 |
| `mamba-retriever-fs` × `fiqa` | queries-shuffled | 0.0863 | 0.0746 | 0.135 |
| `mamba-retriever-fs` × `fiqa` | both-shuffled | 0.0863 | 0.0579 | 0.329 |

## Cross-phase synthesis — query-level shuffle damage

Per-query baseline vs shuffled nDCG@10 and hit@10 are preserved in
`experiments/phase8/query_level_sensitivity.parquet` (or `.json` fallback).
Phase 6 (failure taxonomy) cross-referencing is out of scope for this
pod session and can be layered in offline using the preserved per-query
records — `LEN-LONG` queries should be more position-sensitive than
`LEN-SHORT`; `PARAPHRASE` failures should be position-insensitive under
the bag-of-concepts hypothesis.

## Methodology

- Shuffle operates on `input_ids` AFTER tokenization and BEFORE encoding.
- Special tokens (`[CLS]`, `[SEP]`, `[PAD]`, any tokenizer-special ID) are
  pinned at their original positions. Padding is never touched.
- For a given (dataset, mode, sample), the shuffle permutation is the
  same regardless of which encoder is evaluating — this ensures the
  comparison across encoders is apples-to-apples and not confounded by
  shuffle-noise.
- Pre-trained transformer encoders use the default `sdpa` attention
  (faster than `eager` by ~1.7×). `head_mask` is not used in Phase 8
  so the Phase 7 `attn_implementation="eager"` requirement does not apply.
- From-scratch encoders load checkpoints from `models/<encoder>/best.pt`
  (trained in Session 03 / Phase 5).
- Graded-gain nDCG@10 from the from-scratch `retrieval_metrics.evaluate`
  is the primary metric; `pytrec_eval` runs as a cross-check.

## Limitations

- **Uniform random shuffle**. A local-shuffle variant (swap adjacent
  tokens only) would separate local-order sensitivity from global-order
  sensitivity; flagged as future work.
- **Fixed seed (1729)**. Averaging over multiple shuffle seeds would
  produce error bars but triples compute. Out of scope for Session 06.
- **Short queries are degenerate**. Queries with ≤2 content tokens
  cannot be meaningfully permuted. The `nfcorpus` median query length
  is documented in `experiments/phase3_5/dataset_length_stats.json`
  (computed in Phase 3.5).
- **No fine-tuning on shuffled data**. This measures how much already-
  trained encoders rely on position. "Can you train position-invariant
  encoders?" is a separate research question.

