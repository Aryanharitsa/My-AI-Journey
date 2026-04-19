# Phase 3.5 — Latency profile (30% milestone)

Query encoding latency is the production-critical number (one batch per retrieval request); document encoding throughput is the offline cost.

## Query encoding latency at batch size 1 — median ms

| Encoder \ Dataset | nfcorpus | scifact | fiqa |
|---|---|---|---|
| `minilm-l6-v2` | 4.177 | 4.303 | 4.304 |
| `bert-base` | 7.257 | 7.249 | 7.229 |
| `gte-small` | 7.601 | 7.596 | 7.646 |

## Query latency at batch 1 — percentiles (ms)

| Cell | median | p50 | p90 | p99 |
|---|---:|---:|---:|---:|
| `minilm-l6-v2` x `nfcorpus` | 4.177 | 4.177 | 4.271 | 4.529 |
| `minilm-l6-v2` x `scifact` | 4.303 | 4.303 | 4.365 | 4.505 |
| `minilm-l6-v2` x `fiqa` | 4.304 | 4.304 | 4.356 | 4.387 |
| `bert-base` x `nfcorpus` | 7.257 | 7.257 | 7.332 | 8.699 |
| `bert-base` x `scifact` | 7.249 | 7.249 | 7.362 | 7.508 |
| `bert-base` x `fiqa` | 7.229 | 7.229 | 7.349 | 7.434 |
| `gte-small` x `nfcorpus` | 7.601 | 7.601 | 7.685 | 7.999 |
| `gte-small` x `scifact` | 7.596 | 7.596 | 7.654 | 7.976 |
| `gte-small` x `fiqa` | 7.646 | 7.644 | 7.773 | 7.967 |

## All batch sizes — median ms

| Cell | bs=1 | bs=8 | bs=32 |
|---|---|---|---|
| `minilm-l6-v2` x `nfcorpus` | 4.177 | 5.151 | 6.059 |
| `minilm-l6-v2` x `scifact` | 4.303 | 5.602 | 8.595 |
| `minilm-l6-v2` x `fiqa` | 4.304 | 5.289 | 7.187 |
| `bert-base` x `nfcorpus` | 7.257 | 8.612 | 11.642 |
| `bert-base` x `scifact` | 7.249 | 9.908 | 24.733 |
| `bert-base` x `fiqa` | 7.229 | 8.501 | 17.847 |
| `gte-small` x `nfcorpus` | 7.601 | 9.100 | 9.768 |
| `gte-small` x `scifact` | 7.596 | 9.255 | 11.680 |
| `gte-small` x `fiqa` | 7.646 | 9.322 | 11.067 |

## Document encoding throughput at batch 32 — docs / second

| Encoder \ Dataset | nfcorpus | scifact | fiqa |
|---|---|---|---|
| `minilm-l6-v2` | 669.2 | 732.8 | 1167.9 |
| `bert-base` | 178.4 | 193.4 | 354.3 |
| `gte-small` | 670.2 | 723.8 | 1215.4 |

## Methodology

- Latency measured via `torch.cuda.Event` on CUDA (see `vitruvius.evaluation.latency_profiler`). 10 warmup + 100 measured passes per batch size. Percentiles are over the 100 measured times.
- Query samples: 200 queries sampled (`random.Random(seed=1729).sample`) from each dataset's test split qrels-having set.
- Document samples: 200 documents sampled (same seed, separate call) from each dataset's full corpus.
- Document encoding throughput: 3 warmup rounds then one wall-clock timed encode of all 200 sampled docs at batch size 32. Reported as `200 / wall_time`.
- Token length distributions (computed once, encoder-agnostic) are in `dataset_length_stats.json`. Canonical tokenizer: `sentence-transformers/all-MiniLM-L6-v2`.
- Seed 1729. Device: `cuda`. Hardware: see per-cell JSONs.
- These numbers are for this pod run only. Production latency is hardware-sensitive; treat these as within-study comparison numbers, not absolute benchmarks.

