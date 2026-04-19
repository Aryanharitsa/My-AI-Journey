# Session 03 Report — Vitruvius Pod Work, 2026-04-19

**Milestone delivered:** **Phase 5 at 55%**, absorbing the deferred Phase 4.
**Outcome:** Three non-transformer bi-encoders (BiLSTM, 1D-CNN, Mamba2) trained
from random init on 500K MS MARCO triplets with identical hyperparameters,
evaluated through the existing Phase 3 / 3.5 harness, plotted as the first
6-point latency-accuracy Pareto frontier.

## One-sentence outcome

All three from-scratch encoders trained to completion on the same 500K MS MARCO
triplets with identical hyperparameters; Mamba delivered the highest nDCG@10
among the three but at ~10× the batch-1 latency of LSTM, and the six-point
Pareto shows `conv-retriever`, `lstm-retriever`, `minilm-l6-v2`, `gte-small` on
the frontier.

## Commits

6 new commits on top of `origin/main` (= `f4f9630`, archivist's bundle of
Session 02):

```
63fb067  docs(vitruvius): CHANGELOG 0.5.0 + README roadmap — 40% absorbed, 55% done
6c500ff  feat(vitruvius/experiments): phase 5 training artifacts + pareto v2 (55% milestone)
a2ee28e  feat(vitruvius): phase 5 eval hooks + trainer robustness
57d79df  feat(vitruvius): phase 5 scaffolding — from-scratch encoders + trainer + mamba gate passed
83834d6  chore(vitruvius/experiments): backfill phase3 JSONs with per_query_results
ffa7813  feat(vitruvius/evaluation): preserve per-query results in bench JSONs for Phase 6 failure analysis
```

All bundled in `pod_commits.bundle` in this directory. Not pushed.

## Headline numbers

### Training (3 epochs, batch 64, 19,452 steps each)

| Encoder | Params | Best val loss | Final val loss | Wall (s) | Peak GPU (MB) | AMP | num_workers |
|---|---:|---:|---:|---:|---:|:---:|:---:|
| `lstm-retriever`    |  6.41M | 0.4719 | 0.4719 |  615.7 | 561  | ✓ | 2 |
| `conv-retriever`    |  4.92M | 0.7962 | 0.7964 |  984.6 | 189  | ✓ | 2 |
| `mamba-retriever-fs`| 23.74M | **0.1807** | 0.1807 | 2974.2 | 2239 | ✓ | 0 |

Mamba's `num_workers=0` is the only training-config delta across the three
encoders. Workers forked from a process holding Triton-JIT state from
`mamba_ssm`'s selective-scan kernels segfault mid-training (attempt 1 died
at step 5000). Full diagnosis in `notes/mamba_install_attempt_02.md`. Same
model, same hyperparameters, same data.

### nDCG@10 on BEIR test subsets

**From-scratch (Phase 5):**

| Encoder | `nfcorpus` | `scifact` | `fiqa` | Avg |
|---|---:|---:|---:|---:|
| `lstm-retriever`    | 0.1901     | 0.3606 | 0.0886 | 0.2131 |
| `conv-retriever`    | 0.1400     | 0.1978 | **0.0370** | 0.1249 |
| `mamba-retriever-fs`| **0.2083** | **0.3752** | 0.0863 | 0.2233 |

**Pre-trained reference (Phase 3, for comparison):**

| Encoder | `nfcorpus` | `scifact` | `fiqa` | Avg |
|---|---:|---:|---:|---:|
| `minilm-l6-v2` | 0.3165 | 0.6451 | 0.3687 | 0.4434 |
| `bert-base`    | 0.3169 | 0.6082 † | 0.3229 | 0.4160 |
| `gte-small`    | 0.3492 | 0.7269 | 0.3937 | 0.4899 |

(† `bert-base × scifact` flagged in Phase 3 — MSMARCO-dot-v5 OOD transfer gap.)

### Query encoding latency @ batch 1 (median ms, 200-query sample)

| Encoder | `nfcorpus` | `scifact` | `fiqa` | Avg |
|---|---:|---:|---:|---:|
| `conv-retriever`    |  0.84 |  0.88 |  0.88 |  0.87 |
| `lstm-retriever`    |  0.96 |  1.45 |  1.54 |  1.32 |
| `minilm-l6-v2`      |  4.18 |  4.30 |  4.30 |  4.26 |
| `bert-base`         |  7.26 |  7.25 |  7.23 |  7.25 |
| `gte-small`         |  7.60 |  7.60 |  7.65 |  7.62 |
| `mamba-retriever-fs`| 11.82 | 11.96 | 11.87 | 11.88 |

### Pareto v2 — 6-point frontier

[`figures/pareto_v2.png`](figures/pareto_v2.png) +
[`figures/pareto_v2_caption.md`](figures/pareto_v2_caption.md).

**Pareto-optimal subset**: `conv-retriever`, `lstm-retriever`,
`minilm-l6-v2`, `gte-small`. `bert-base` and `mamba-retriever-fs` are
dominated — slower AND lower accuracy than at least one point.

## Surprises, deviations, judgment calls

1. **Mamba attempt 1 crashed at step 5000 with a DataLoader segfault.**
   Root cause: `multiprocessing.fork` in PyTorch's DataLoader workers
   inherits inconsistent Triton-JIT state from `mamba_ssm`'s selective-scan
   kernels. Not a model bug (val loss was descending cleanly, best.pt at
   0.418 was already better than LSTM's final 0.472). I preserved the
   interrupted checkpoint at
   `models/mamba-retriever-fs/interrupted_run_best_step5000.pt` for forensic
   provenance (shows Mamba's per-step convergence is faster than LSTM's)
   and restarted with `num_workers=0`. Attempt 2 trained all 19,452 steps.

2. **CNN attempt 1 crashed on fp16 overflow in masked max-pool.** Used
   `float("-1e9")` as the sentinel in `masked_fill`; under AMP autocast to
   fp16 that value exceeds half-precision range. Switched to
   `torch.finfo(x.dtype).min`. Smoke test now would have caught this —
   added to the flow for future sessions.

3. **LSTM attempt 1 crashed on malformed JSON in MS MARCO.** A few
   web-scraped passages in the triplet data contain embedded control
   characters that `json.loads` rejects by default. Patched `TripletDataset`
   to use `strict=False` and skip unrecoverable lines with a warning.

4. **CNN × FiQA = 0.037**, below the session-03 §5.8 red-flag threshold of
   0.05. Documented as a measured finding: max receptive field of 7 tokens
   (stacked kernels 3/5/7) cannot span FiQA's longer financial documents.
   Training loss previewed this (CNN final val 0.80 vs LSTM 0.47 vs Mamba
   0.18). pytrec_eval agrees bit-exact. Kept in the grid as-is per the
   project's "don't bury findings" discipline.

5. **Mamba is the slowest encoder here**, at ~12 ms batch-1 vs transformers'
   ~4-7 ms. The paper's linear-time inference advantage only pays off at
   long sequences; at max_seq_len=128 the 12-layer stack's kernel-launch
   overhead dominates. An honest data point for the Pareto story — Mamba
   wins on accuracy among the from-scratch encoders but pays ~10× latency
   for ~0.5× accuracy advantage over LSTM.

6. **Skipped CHANGELOG 0.4.0** on the version bump (jumped 0.3.5 → 0.5.0).
   Matches the phase-number gap now that Phase 4 is absorbed into Phase 5.

7. **Mamba install succeeded on attempt 1** of this session (unpinned
   latest, 33 min from source). Session-02 attempt got halted mid-compile
   by the kill-switch decision, but nothing was wrong with the build itself.
   Full runbook + PATH gotcha in `notes/mamba_install_attempt_02.md`.

## Discipline held

- Per-query results schema (§5.7) added as the FIRST commit before any
  training; Phase 3 JSONs backfilled with zero aggregate-metric drift.
  Phase 6 is unblocked.
- One training pass per encoder. No hyperparameter sweeps. No silent re-runs.
- When Mamba attempt 1 crashed, I surfaced A/B/C to the operator and did
  not act unilaterally. Operator chose B; I executed.
- All commits on pod, not pushed. Archivist merges.
- Truthful numbers throughout. `conv × fiqa` left flagged in-grid, not
  massaged or hidden behind an asterisk in prose.
- Checkpoints NOT committed (see gitignored `models/`).

## Checkpoint sizes (stay on pod, not committed, will die with the pod)

```
models/lstm-retriever/best.pt             ~26 MB
models/lstm-retriever/final.pt            ~26 MB
models/conv-retriever/best.pt             ~20 MB
models/conv-retriever/final.pt            ~20 MB
models/mamba-retriever-fs/best.pt         ~95 MB
models/mamba-retriever-fs/final.pt        ~95 MB
models/mamba-retriever-fs/
  interrupted_run_best_step5000.pt        ~95 MB  (archived, attempt 1, val 0.418)
```

Total ~377 MB pod-local. No plan to commit any of them; Phase 6 and future
benches can re-load from HF cache of the transformer encoders + these pod
checkpoints while the pod stays warm, or re-train from `data/msmarco/` via
`scripts/download_msmarco.py` + `vitruvius train` on a fresh pod.

## Recommendations for Session 04 / Phase 6

1. **Phase 6 (per-query failure analysis) is ready to run locally on the
   Mac.** Every bench JSON already has `per_query_results` populated.
   Natural focus: why does Mamba win SciFact but CNN lose FiQA? Per-query
   overlap between the three architectures' top-10 rankings would say
   something about what each architecture actually learns to retrieve.

2. **If a future Mamba training session runs**, use `--num-workers 0`
   unconditionally for any mamba_ssm-backed body. Alternative: pre-tokenize
   the dataset to numpy arrays off-line — worker processes then never touch
   Python-side Triton state.

3. **Pre-trained Mamba bi-encoder path remains open.** The SPScanner
   cross-encoder from MambaRetriever is still on HF; a dedicated session
   could add a separate cross-encoder scoring path to Vitruvius and score
   it through the harness. Different path than Phase 5's from-scratch
   bi-encoder; orthogonal Pareto data point.

4. **Longer-sequence experiment would favor Mamba.** At `max_seq_len=128`
   the O(n²) attention cost of transformers is small. Running the Pareto
   plot again with `max_seq_len=512` or `1024` would likely flip Mamba
   onto the frontier (both faster than attention at long seqs, and its
   23.7M parameters give it expressive headroom). Out of scope for
   Session 04 (that's Phase 8 territory — position/length sensitivity).

## Artifacts in this bundle

```
experiments/session_03/
  SESSION_03_REPORT.md              (this file)
  pod_commits.bundle                (6 commits on top of f4f9630)
  session_03_full_log.txt           (condensed run log)
  phase5/                           (copy of experiments/phase5/)
    lstm-retriever__training.{json,csv}
    conv-retriever__training.{json,csv}
    mamba-retriever-fs__training.{json,csv}
    bench/<encoder>__<dataset>.json × 9
    profile/<encoder>__<dataset>.json × 9
    dataset_length_stats.json
    SUMMARY.md
    *.log
  figures/
    pareto_v2.png, pareto_v2.pdf, pareto_v2_caption.md
  notes/
    mamba_install_attempt_02.md     (reads first, next Mamba session)
```

Operator will pull to Mac and invoke the archivist.
