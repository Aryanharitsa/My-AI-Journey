# Phase 3 — 20% milestone

Three-encoder × three-BEIR-subset accuracy sweep. Primary metric: nDCG@10
(from-scratch graded-gain `2^rel − 1`, cross-checked against `pytrec_eval`).

## Encoders

| Registry key | HuggingFace checkpoint | Params | Similarity |
|---|---|---:|---|
| `minilm-l6-v2` | `sentence-transformers/all-MiniLM-L6-v2` | 22M | cosine |
| `bert-base` | `sentence-transformers/msmarco-bert-base-dot-v5` | 110M | dot |
| `gte-small` | `thenlper/gte-small` | 33M | cosine |

## Datasets

| BEIR subset | Corpus | Test queries | Max rel grade |
|---|---:|---:|---:|
| `nfcorpus` | 3,633 | 323 | 2 (graded) |
| `scifact` | 5,183 | 300 | 1 (binary) |
| `fiqa` | 57,638 | 648 | 1 (binary) |

## Headline

**8/9 cells in band** (±0.03 of approximate BEIR leaderboard references). One flagged:
`bert-base × scifact` at nDCG@10 = 0.6082 vs reference 0.68 (Δ = −0.072). Not a harness
bug (`pytrec_eval` agrees bit-exact, same dataset in-band on the other two encoders) —
it's a measured out-of-domain transfer gap of `msmarco-bert-base-dot-v5` on
scientific-claim retrieval. The flag stays; the number is not massaged.

See **[`SUMMARY.md`](SUMMARY.md)** for the full 3×3 grid, per-cell deltas, the
dot-vs-cosine investigation, the `pytrec_eval` cross-check table, and runtime
breakdown. Per-cell raw artifacts are in `<encoder>__<dataset>.json`. Raw stdout
of the sweep (both pre-fix cosine-forced run and post-fix dot run, interleaved
chronologically) is in `sweep.log`.

## Reproduce

```bash
python -m vitruvius.cli bench-sweep \
  --encoders minilm-l6-v2 bert-base gte-small \
  --datasets nfcorpus scifact fiqa \
  --split test --batch-size 128 --top-k 100 --device cuda \
  --output-dir experiments/phase3/
```

Seed 1729. Hardware for the reference run: NVIDIA A100-SXM4-80GB, torch 2.4.1+cu124,
FAISS 1.13.2, Python 3.11.10.
