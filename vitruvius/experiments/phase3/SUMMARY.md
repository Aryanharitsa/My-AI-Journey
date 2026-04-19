# Phase 3 — 3 encoders × 3 BEIR subsets (20% milestone)

## Methodology

Primary metric: **nDCG@10** from `vitruvius.evaluation.retrieval_metrics.evaluate`
(from-scratch, `gain = 2^rel - 1`, discount `log2(i+1)`, iDCG over full qrels —
gate IR-2). `pytrec_eval` runs alongside as a cross-check, not a replacement.

Retrieval: FAISS `IndexFlatIP` over 384/768-dim embeddings. Similarity is
per-encoder, declared on the `Encoder` base class (commit `442fec0`):

- `minilm-l6-v2` → `similarity = "cosine"` → embeddings L2-normalized → IP = cosine.
- `gte-small` → `similarity = "cosine"` → embeddings L2-normalized → IP = cosine.
- `bert-base` → `similarity = "dot"` → embeddings **not** normalized → IP = raw dot.

Doc format: `(title + " " + text).strip()` (BEIR convention).
Seed 1729. Device: `cuda` (A100-SXM4-80GB). `top-k = 100`, batch size 128.

References from session-02 handoff §3.4 (approximate BEIR leaderboard).
Tolerance: ±0.03 per cell. Out-of-band cells are **flagged**, not massaged,
not silently re-run with different seeds.

## The dot-vs-cosine investigation (kept in the record)

The initial Phase 3 sweep forced L2-normalization on all encoders per the
session-02 handoff rule §4. Result on `bert-base`:

| Cell | Measured nDCG@10 (cosine, **wrong**) | Reference | Δ |
|---|---:|---:|---:|
| `bert-base` × `nfcorpus` | 0.2309 | 0.31 | **-0.0791** |
| `bert-base` × `scifact`  | 0.5691 | 0.68 | **-0.1109** |
| `bert-base` × `fiqa`     | 0.2205 | 0.30 | **-0.0795** |

All three out-of-band in the same direction, large, and consistent. The
`pytrec_eval` cross-check agreed with the from-scratch numbers to
`|Δ| ≤ 2.4e-3`, ruling out an evaluator bug.

Root cause (confirmed by inspecting `config_sentence_transformers.json` in the
HF cache): `sentence-transformers/msmarco-bert-base-dot-v5` declares
`"similarity_fn_name": "dot"` — it was trained with dot product on raw
(unnormalized) embeddings. Forcing L2-norm maps it onto cosine, a different
distance than its training objective. The −0.08 to −0.11 gap matches the
well-known dot-trained-eval'd-as-cosine degradation.

Fix (commits `442fec0` + `ab0bd98`):

1. Made `similarity` a required attribute on the `Encoder` base class. Every
   wrapper — current transformers, future Mamba, future from-scratch LSTM/CNN —
   must declare its training objective. The bench harness reads the attribute
   and decides whether to L2-normalize. Forgetting to declare it fails at
   class-creation time.
2. Set `BERTEncoder.similarity = "dot"` and `normalize_embeddings=False`.
   FAISS `IndexFlatIP` on non-normalized embeddings is raw dot product —
   exactly what the checkpoint was trained for.
3. Re-ran the sweep. Cosine cells are idempotent within tie-break noise; the
   three `bert-base` cells get the corrected numbers (§"Corrected results"
   below).

## A measured finding: msmarco-bert-base-dot-v5 on SciFact

After the dot fix, eight of nine cells are in band. The one remaining
out-of-band cell — `bert-base` × `scifact`, measured 0.6082 vs reference 0.68
(Δ = −0.072) — is a real accuracy characteristic of this specific checkpoint,
not a harness problem.

Evidence:
- `pytrec_eval` agrees bit-exact with the from-scratch score (|Δ| = 0.0); the
  evaluator is not the story.
- On the same SciFact split, `minilm-l6-v2` lands at 0.6451 and `gte-small` at
  0.7269 — both in band — so the dataset itself is well-behaved.
- `bert-base` on `nfcorpus` (+0.007) and `fiqa` (+0.023) are both in band.
  The checkpoint is not broken; it's just weaker specifically on SciFact.
- `sentence-transformers/msmarco-bert-base-dot-v5` was trained exclusively on
  MS MARCO (web QA). SciFact is scientific-claim retrieval — out-of-domain
  relative to the training data, and in a direction that appears to hurt this
  particular checkpoint more than it hurts the other two encoders in the
  lineup (MiniLM and GTE are also MS MARCO-adjacent but transfer better here,
  likely via their contrastive training recipes).

The cell is a **finding**, not a failure to reproduce. It says something
specific about the out-of-domain generalization of MSMARCO-BERT-dot on
scientific-claim retrieval, and it's a data point that will matter when the
Phase 4 Pareto plot puts this encoder next to a Mamba retriever trained with
a different objective. The flag stays in the table.


## Corrected results — measured nDCG@10 (post-fix, delta vs reference)

| Encoder \ Dataset | nfcorpus | scifact | fiqa |
|---|---|---|---|
| `minilm-l6-v2` | 0.3165 (Δ+0.0165 vs 0.30) ✅ | 0.6451 (Δ+0.0051 vs 0.64) ✅ | 0.3687 (Δ+0.0087 vs 0.36) ✅ |
| `bert-base` | 0.3169 (Δ+0.0069 vs 0.31) ✅ | 0.6082 (Δ-0.0718 vs 0.68) ❌ | 0.3229 (Δ+0.0229 vs 0.30) ✅ |
| `gte-small` | 0.3492 (Δ+0.0092 vs 0.34) ✅ | 0.7269 (Δ-0.0031 vs 0.73) ✅ | 0.3937 (Δ-0.0263 vs 0.42) ✅ |

## Per-cell breakdown

- `minilm-l6-v2` × `nfcorpus` (similarity=`cosine`): measured 0.3165, ref 0.30, delta +0.0165, in-band. pytrec_eval nDCG@10=0.3159 (|Δ|=5.72e-04).
- `minilm-l6-v2` × `scifact` (similarity=`cosine`): measured 0.6451, ref 0.64, delta +0.0051, in-band. pytrec_eval nDCG@10=0.6451 (|Δ|=0.00e+00).
- `minilm-l6-v2` × `fiqa` (similarity=`cosine`): measured 0.3687, ref 0.36, delta +0.0087, in-band. pytrec_eval nDCG@10=0.3687 (|Δ|=0.00e+00).
- `bert-base` × `nfcorpus` (similarity=`dot`): measured 0.3169, ref 0.31, delta +0.0069, in-band. pytrec_eval nDCG@10=0.3151 (|Δ|=1.81e-03).
- `bert-base` × `scifact` (similarity=`dot`): measured 0.6082, ref 0.68, delta -0.0718, **OUT OF BAND**. pytrec_eval nDCG@10=0.6082 (|Δ|=0.00e+00).
- `bert-base` × `fiqa` (similarity=`dot`): measured 0.3229, ref 0.30, delta +0.0229, in-band. pytrec_eval nDCG@10=0.3229 (|Δ|=0.00e+00).
- `gte-small` × `nfcorpus` (similarity=`cosine`): measured 0.3492, ref 0.34, delta +0.0092, in-band. pytrec_eval nDCG@10=0.3480 (|Δ|=1.17e-03).
- `gte-small` × `scifact` (similarity=`cosine`): measured 0.7269, ref 0.73, delta -0.0031, in-band. pytrec_eval nDCG@10=0.7269 (|Δ|=0.00e+00).
- `gte-small` × `fiqa` (similarity=`cosine`): measured 0.3937, ref 0.42, delta -0.0263, in-band. pytrec_eval nDCG@10=0.3937 (|Δ|=0.00e+00).

## Cross-check status (from-scratch vs pytrec_eval)

| Cell | max\|Δ\| across nDCG@{1,5,10,100} | Recall@k bit-exact? |
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
| `minilm-l6-v2` | 5.84 | 7.47 | 55.30 |
| `bert-base` | 21.21 | 28.46 | 196.24 |
| `gte-small` | 6.19 | 8.14 | 52.29 |

## Artifacts

- Nine per-cell JSONs (`<encoder>__<dataset>.json`) — full config, dataset stats, hardware, runtime breakdown, both metric sources, delta, band check. Machine-readable; ingest into Phase 4 comparison tables.
- `sweep.log` — raw stdout/stderr of the initial (cosine-forced) and the post-fix sweep, interleaved chronologically. Preserves the pre-fix bert-base numbers in the log even though the JSONs were overwritten when the sweep was re-run.

Reproduce:

```bash
python -m vitruvius.cli bench-sweep \
  --encoders minilm-l6-v2 bert-base gte-small \
  --datasets nfcorpus scifact fiqa \
  --split test --batch-size 128 --top-k 100 --device cuda \
  --output-dir experiments/phase3/
```
