# Phase 3.5 — 30% milestone

Latency profile of the three transformer encoders across three BEIR subsets and
three batch sizes (1, 8, 32). Query encoding latency is the production-critical
number (one batch per retrieval request); document encoding throughput is the
offline cost.

## What was measured

- Query encoding latency at batch sizes **1 / 8 / 32** — median + p50 / p90 / p99
  over 100 measured passes after 10 warmup, per cell.
- Document encoding throughput at **batch 32** — docs/second over a 200-doc
  sampled encode, wall-clock timed, 3 warmup rounds.
- Sampled inputs are 200 **real** BEIR queries + 200 **real** BEIR docs per
  dataset (not synthetic fixed-length strings — transformer latency scales
  non-linearly with sequence length).

## Headline

Batch-1 median ms (lower is better):

| Encoder | nfcorpus | scifact | fiqa |
|---|---:|---:|---:|
| `minilm-l6-v2` | 4.18 | 4.30 | 4.30 |
| `bert-base` | 7.26 | 7.25 | 7.23 |
| `gte-small` | 7.60 | 7.60 | 7.65 |

At batch 1, latency is essentially flat across datasets within each encoder —
the cost is dominated by kernel launch and fixed forward-pass overhead;
sequence-length variance is noise against that.

**Headline data point for Phase 5:** `bert-base × scifact` at batch 32 is 24.7 ms —
3.4× its batch-1 time and 2.1× its `nfcorpus` batch-32 time. That's the O(n²)
attention cost on longer scientific documents becoming visible, and it is exactly
the kind of gap the Phase 5 Pareto plot will interrogate (linear-time SSM /
recurrent / conv encoders vs quadratic attention).

See **[`SUMMARY.md`](SUMMARY.md)** for the full percentile tables (p50 / p90 / p99),
all nine cells at each batch size, document-encoding throughput, and the full
methodology. Per-dataset query/doc length distributions are in
`dataset_length_stats.json`. Raw stdout of the profile run is in `profile.log`.
Per-cell raw artifacts are in `<encoder>__<dataset>.json`.

## Reproduce

```bash
python -m vitruvius.cli profile \
  --encoders minilm-l6-v2 bert-base gte-small \
  --datasets nfcorpus scifact fiqa \
  --split test --batch-sizes 1 8 32 \
  --n-queries 200 --n-docs 200 \
  --warmup 10 --measured 100 \
  --device cuda \
  --output-dir experiments/phase3_5/
```

Seed 1729 (separate RNGs for query vs document sampling). Hardware for the
reference run: NVIDIA A100-SXM4-80GB, torch 2.4.1+cu124, FAISS 1.13.2,
Python 3.11.10. These are within-study comparison numbers on this specific
pod; production latency is hardware-sensitive.
