# Phase 3.5 — Latency profile (30% milestone)

Query encoding latency is the production-critical number (one batch per retrieval request); document encoding throughput is the offline cost.

## Query encoding latency at batch size 1 — median ms

| Encoder \ Dataset | nfcorpus | scifact | fiqa |
|---|---|---|---|
| `lstm-retriever` | 0.964 | 1.452 | 1.543 |
| `conv-retriever` | 0.840 | 0.879 | 0.878 |
| `mamba-retriever-fs` | 11.819 | 11.956 | 11.874 |

## Query latency at batch 1 — percentiles (ms)

| Cell | median | p50 | p90 | p99 |
|---|---:|---:|---:|---:|
| `lstm-retriever` x `nfcorpus` | 0.964 | 0.964 | 0.977 | 1.004 |
| `lstm-retriever` x `scifact` | 1.452 | 1.451 | 1.489 | 1.728 |
| `lstm-retriever` x `fiqa` | 1.543 | 1.543 | 1.556 | 1.579 |
| `conv-retriever` x `nfcorpus` | 0.840 | 0.840 | 0.865 | 1.083 |
| `conv-retriever` x `scifact` | 0.879 | 0.879 | 0.893 | 0.918 |
| `conv-retriever` x `fiqa` | 0.878 | 0.878 | 0.897 | 1.072 |
| `mamba-retriever-fs` x `nfcorpus` | 11.819 | 11.818 | 11.985 | 13.555 |
| `mamba-retriever-fs` x `scifact` | 11.956 | 11.956 | 12.207 | 12.584 |
| `mamba-retriever-fs` x `fiqa` | 11.874 | 11.873 | 11.924 | 11.986 |

## All batch sizes — median ms

| Cell | bs=1 | bs=8 | bs=32 |
|---|---|---|---|
| `lstm-retriever` x `nfcorpus` | 0.964 | 1.391 | 2.340 |
| `lstm-retriever` x `scifact` | 1.452 | 3.326 | 5.447 |
| `lstm-retriever` x `fiqa` | 1.543 | 2.019 | 4.060 |
| `conv-retriever` x `nfcorpus` | 0.840 | 1.061 | 1.925 |
| `conv-retriever` x `scifact` | 0.879 | 1.574 | 3.697 |
| `conv-retriever` x `fiqa` | 0.878 | 1.276 | 3.002 |
| `mamba-retriever-fs` x `nfcorpus` | 11.819 | 12.129 | 13.253 |
| `mamba-retriever-fs` x `scifact` | 11.956 | 12.833 | 15.212 |
| `mamba-retriever-fs` x `fiqa` | 11.874 | 12.267 | 14.478 |

## Document encoding throughput at batch 32 — docs / second

| Encoder \ Dataset | nfcorpus | scifact | fiqa |
|---|---|---|---|
| `lstm-retriever` | 963.5 | 1008.1 | 1513.3 |
| `conv-retriever` | 1133.4 | 1203.2 | 2035.5 |
| `mamba-retriever-fs` | 110.4 | 762.7 | 1039.9 |

## Methodology

- Latency measured via `torch.cuda.Event` on CUDA (see `vitruvius.evaluation.latency_profiler`). 10 warmup + 100 measured passes per batch size. Percentiles are over the 100 measured times.
- Query samples: 200 queries sampled (`random.Random(seed=1729).sample`) from each dataset's test split qrels-having set.
- Document samples: 200 documents sampled (same seed, separate call) from each dataset's full corpus.
- Document encoding throughput: 3 warmup rounds then one wall-clock timed encode of all 200 sampled docs at batch size 32. Reported as `200 / wall_time`.
- Token length distributions (computed once, encoder-agnostic) are in `dataset_length_stats.json`. Canonical tokenizer: `sentence-transformers/all-MiniLM-L6-v2`.
- Seed 1729. Device: `cuda`. Hardware: see per-cell JSONs.
- These numbers are for this pod run only. Production latency is hardware-sensitive; treat these as within-study comparison numbers, not absolute benchmarks.

