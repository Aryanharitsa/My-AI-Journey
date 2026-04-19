# Project Vitruvius

A comparative study of transformer and non-transformer encoder architectures
for dense retrieval. Where on the latency–accuracy Pareto frontier do alternative
architectures sit, and where specifically is the transformer's full bidirectional
attention earning its cost versus being wasted compute?

## Research Question

Dense retrieval systems rely on deep encoder models to produce embeddings for
similarity search. In production, these encoders operate under hard latency
constraints — and under that pressure, the transformer quietly becomes a
bottleneck. The standard response is distillation: smaller transformers that
trade accuracy for speed. But this accepts a framing we want to challenge:

**What if the transformer is the wrong inductive bias for retrieval in the first place?**

Retrieval differs from language modeling in ways that matter architecturally:

- **Asymmetric compute**: Document encoding is offline; query encoding is latency-critical.
- **Output modality**: The encoder produces a single fixed vector, not a token sequence.
- **Position sensitivity**: Retrieval may be more bag-of-concepts than sequence-order-dependent.
- **Task structure**: Compressing meaning into a dot-product-comparable vector is fundamentally different from next-token prediction.

This project empirically compares transformer, SSM (Mamba), recurrent (LSTM),
and convolutional (CNN) encoders on standard retrieval benchmarks, measuring
both accuracy AND latency, and asks: where on the Pareto frontier do
non-transformer architectures actually sit?

## Roadmap

| Milestone | Phase | Summary | Status |
|-----------|-------|---------|--------|
| 10%       | P1    | Local-only scaffold; CPU smoke test; module skeleton stands up. | done |
| 10%       | P2    | Reproduce a published BEIR number (MiniLM-L6-v2 nDCG@10 ~ 0.30 on NFCorpus). | done |
| 20%       | P3    | Benchmark 3 encoders x 3 BEIR subsets. Accuracy table. | done |
| 30%       | P3.5  | Latency profiler turned on. Batch sizes 1/8/32 timed for transformer encoders. | - |
| 40%       | P4    | Mamba Retriever integrated. First 4-point Pareto plot. | - |
| 55%       | P5    | LSTM + 1D-CNN encoders trained from scratch on MS MARCO subset. Full 6-encoder Pareto. | - |
| 70%       | P6    | Per-query failure analysis with taxonomy. | - |
| 80%       | P7    | Attention head pruning for retrieval specifically (not LM). | - |
| 90%       | P8    | Token-position shuffle. Position sensitivity per architecture. | - |
| 100%      | P9    | 5 to 7 page opinionated technical note, arXiv-ready. | - |

## Quickstart

```bash
cd vitruvius

# 1. Create a Python 3.11 venv (uv handles the interpreter)
make venv
source .venv/bin/activate

# 2. Install with dev extras
make dev

# 3. Run the smoke test (synthetic data, no model download)
python -m vitruvius.cli smoke --cpu --no-encoder

# 4. Run the test suite
make test

# 5. Lint
make lint
```

Expected smoke output (the `metrics` block):

```json
{
  "encoder": "hash-bag-of-features",
  "embedding_dim": 256,
  "metrics": {
    "MRR@10": 0.6083,
    "Recall@10": 0.7,
    "nDCG@10": 0.5965
  }
}
```

The hash-bag stand-in is intentionally crude — it is what runs when
`--no-encoder` is passed so the smoke test is fully offline. Drop the
`--no-encoder` flag to download MiniLM-L6-v2 and run the same path with a
real encoder.


### Phase 3 — 3×3 encoder × BEIR sweep (A100, 2026-04-19)

Primary metric: nDCG@10 (from-scratch `retrieval_metrics.evaluate`, graded-gain).
Tolerance: ±0.03 vs approximate BEIR leaderboard references.

| Encoder | `nfcorpus` | `scifact` | `fiqa` |
|---|---:|---:|---:|
| `minilm-l6-v2` (cosine) | **0.3165** (+0.017) ✅ | **0.6451** (+0.005) ✅ | **0.3687** (+0.009) ✅ |
| `bert-base` (dot)       | **0.3169** (+0.007) ✅ | **0.6082** (−0.072) ❌ | **0.3229** (+0.023) ✅ |
| `gte-small` (cosine)    | **0.3492** (+0.009) ✅ | **0.7269** (−0.003) ✅ | **0.3937** (−0.026) ✅ |

8/9 in band. The flagged cell (`bert-base × scifact`) is a measured
out-of-domain transfer gap of `msmarco-bert-base-dot-v5` on scientific-claim
retrieval — pytrec_eval agrees bit-exact, other encoders on the same
dataset are fine. Full discussion + methodology in
[`experiments/phase3/SUMMARY.md`](experiments/phase3/SUMMARY.md).

## Layout

```
vitruvius/
|-- pyproject.toml             # installable package
|-- Makefile                   # venv, dev, smoke, test, lint, format, ...
|-- src/vitruvius/
|   |-- cli.py                 # python -m vitruvius.cli {smoke,bench,profile,shuffle,prune}
|   |-- config.py              # pydantic experiment configs
|   |-- data/                  # synthetic + BEIR loader
|   |-- encoders/              # base + registry + MiniLM/BERT/GTE + Mamba/LSTM/CNN stubs
|   |-- evaluation/            # nDCG/Recall/MRR (from scratch), FAISS wrapper, latency profiler
|   |-- analysis/              # error analysis / pruning / position probes (Phases 6-8)
|   `-- utils/                 # logging, seed, device picker
|-- scripts/                   # download_beir.py, setup_pod.sh
|-- tests/                     # smoke, metrics, encoder interface
|-- experiments/               # run outputs (gitignored)
|-- figures/                   # plots (gitignored)
`-- notebooks/                 # walkthroughs
```

## Platform notes

On macOS, `faiss-cpu` and `torch` each ship their own `libomp`. Without a
workaround, calling `IndexFlatIP.search` after importing torch crashes the
process. Vitruvius applies two fixes automatically when imported:

1. Sets `KMP_DUPLICATE_LIB_OK=TRUE` (no-op on Linux pods).
2. Pins faiss to single-threaded mode on Darwin via `faiss.omp_set_num_threads(1)`.

These have no effect on Linux (the pod target uses `faiss-gpu` and a single
shared `libomp`).

## Connection to MSR India Research

This project was motivated by the MSR India research agenda on alternative
encoder architectures for dense retrieval. The question — *what assumptions
are we making when we reach for a transformer?* — is the starting point.

## References

See [LITERATURE.md](./LITERATURE.md) for the reading list and
[METHODOLOGY.md](./METHODOLOGY.md) for the experimental design.
