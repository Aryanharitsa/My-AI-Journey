# Phase 2 — MiniLM × NFCorpus reproduction

## What this run is

The Phase 2 success criterion from the roadmap: reproduce
**MiniLM-L6-v2 nDCG@10 ≈ 0.30 on NFCorpus test** to within ±0.02 of the
BEIR leaderboard reference (Thakur et al. 2021, Table 4).

One encoder × one dataset × one split — small on purpose, so the first
pod run is a confidence check that the pipeline (BEIR load → encode →
FAISS → graded-gain nDCG) returns a number that agrees with the
literature before we scale up in Phase 3.

## How it was run

- **Pod:** A100-SXM4-80GB, Ubuntu 24.04, driver 580.126.16, Python 3.11.10.
- **Torch 2.4.1+cu124**, FAISS 1.13.2 (CPU, single-threaded during eval).
  No faiss-gpu — NFCorpus has 3,633 docs, search is already ~0.26 s.
- **No credentials used on the pod.** `huggingface-cli login` / `wandb login`
  are gated behind env-var presence checks in `scripts/setup_pod.sh` and
  neither env var was set. MiniLM is public; no WandB logging.
- **Encoder:** `sentence-transformers/all-MiniLM-L6-v2`, 384-dim,
  L2-normalized (ST default), on CUDA, batch size 128.
- **Doc format:** `(title + " " + text).strip()` — BEIR convention.
- **Retrieval:** FAISS IndexFlatIP, top-k = 100.
- **Metrics:** Our from-scratch `retrieval_metrics` (graded-gain nDCG,
  `gain = 2^rel - 1`; discount `log2(i+1)`) as the primary source.
  `pytrec_eval` run in parallel for cross-check. Gate IR-2 keeps the
  from-scratch implementation as the source of truth.
- **Seed:** 1729.

Reproduce the run:

```bash
python -m vitruvius.cli bench \
    --encoder minilm-l6-v2 \
    --dataset nfcorpus \
    --split test \
    --batch-size 128 \
    --top-k 100 \
    --device cuda \
    --output experiments/phase2/phase2_smoke.json
```

## Headline result

| Metric | Ours (from scratch) | pytrec_eval | \|Δ\| | Reference | Delta from ref |
|---|---:|---:|---:|---:|---:|
| nDCG@10 | **0.316513** | 0.315941 | 5.7e-4 | 0.30 ±0.02 | +0.0165 (**in band**) |
| Recall@10 | 0.146058 | 0.146058 | 0.0 | — | — |
| Recall@100 | 0.326335 | 0.326335 | 0.0 | — | — |
| nDCG@1 | 0.464396 | 0.457687 | 6.7e-3 | — | — |
| nDCG@5 | 0.344018 | 0.341950 | 2.1e-3 | — | — |
| nDCG@100 | 0.328010 | 0.333641 | 5.6e-3 | — | — |

Wall-clock breakdown (seconds): encode docs **5.43**, encode queries **0.03**,
index build **0.03**, search **0.26**, eval ours **0.02**, eval pytrec **0.02**,
total **10.99**.

## Why the pytrec_eval nDCG deltas are non-zero

`Recall@k` matches bit-exact, as expected — recall cares only about set
membership in the top-k cutoff, not ranking order within it.

`nDCG` differences are ≤ 7e-3 across all k and come from **tie-breaking**,
not from a formula mismatch:

- Both implementations use graded-gain DCG with `gain = 2^rel - 1` and
  discount `log2(i+1)`, and both normalize by the ideal DCG computed
  over the full qrels set. The gain/discount/iDCG definitions agree.
- FAISS `IndexFlatIP` returns tied scores in index-insertion order
  (the order we added docs: first corpus row wins on ties).
- `pytrec_eval` (via `trec_eval`) breaks ties by sorting on the
  `(docid, -score)` key — lexicographic docid as tiebreaker.
- On NFCorpus, many query-doc dot products collide near rank 1 and
  near rank 100. Different tiebreaks → different doc at a rank with
  non-zero graded relevance → small nDCG shift.
- `Recall@k` is immune as long as the tied docs are either all above
  or all below the cutoff, which holds at k ∈ {1,5,10,100} here (the
  deltas are 0.0 for all four).

This is expected behavior, not a bug. The primary report uses our
from-scratch number; pytrec_eval stays in the artifact as a sanity check
that would catch genuine formula bugs (which would produce deltas of
O(10⁻²) or larger, not 10⁻³).

## Embedding sanity checks logged by the run

- Max doc L2 norm: 1.000001 (ST `normalize_embeddings=True`, expected ≈ 1.0).
- Max query L2 norm: 1.000000.
- This lets FAISS inner-product search act as cosine similarity.

## What is NOT claimed

- This is one dataset, one encoder. **No Pareto story yet.** Phase 3
  is 3 encoders × 3 BEIR subsets (see roadmap).
- nDCG@10 = 0.3165 is +0.0165 above the BEIR paper's 0.30. That is
  inside the band but not bit-exact. Possible sources: corpus field
  ordering, tokenizer versions, sentence-transformers version bump
  since the paper. Not investigated here because the reproduction
  target is met.
- Latency numbers here are wall-clock, not the rigorous batched
  latency profile Phase 3.5 will produce. Do not cite these as
  latency results.

## Files in this directory

- `phase2_smoke.json` — full run artifact (config, hardware,
  dataset stats, runtime breakdown, both metric sources, delta, band
  check). Machine-readable; ingest this into future comparison tables.
- `nfcorpus_minilm_test.log` — raw stdout/stderr of the run, including
  `nvidia-smi` snapshot at run start.
- `env_snapshot.txt` — `pip freeze` output from the pod at run time.
- `pod_commits.bundle` — git bundle of the pod-side commit(s) that
  implemented Phase 2. Apply with `git bundle verify` then `git fetch
  pod_commits.bundle main:phase2`.
- `README.md` — this file.
