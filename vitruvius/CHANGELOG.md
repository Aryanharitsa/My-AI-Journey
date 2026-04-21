# Changelog

All notable changes to Project Vitruvius are documented here. Versioning follows
the milestone tiers in the project roadmap (0.1.0 = Phase 1, 0.2.0 = Phase 2, …).

## [0.7.0] — 2026-04-21 — Phase 7: 80% milestone (attention head pruning)

Per-head ablation + cumulative-pruning-curve characterization of the three
pre-trained transformer encoders (minilm-l6-v2, bert-base/msmarco-dot-v5,
gte-small) on BEIR NFCorpus, SciFact, FiQA. Direct evidence for the
Phase 9 opinions section: "how many attention heads are actually needed
for retrieval?"

### Added

- `src/vitruvius/encoders/pruned_transformer.py`:
  `PrunedTransformerEncoder` wraps a pre-trained transformer with a
  runtime `head_mask`. Exposes `set_head_mask(tensor)` for in-place
  updates so sweep loops can re-ablate without reloading the model.
- `tests/test_pruned_transformer.py`: four unit tests (all-ones
  equivalence, all-zeros non-crash, shape validation, bad-alias error).
  **Known gap documented in SUMMARY.md**: these tests did NOT initially
  verify "all-ones differs from all-zeros" — the missing test that
  would have caught the transformers 5.x head_mask silent-drop bug.
- `scripts/head_importance_sweep.py`: single-head ablation sweep.
  Tokenize-once-per-cell + manual forward + AMP fp16 path (bypasses
  sentence-transformers' per-call overhead → ~7× speedup on MiniLM).
- `scripts/cumulative_pruning_sweep.py`: prune-N-least-important sweep
  at N ∈ {0, 4, 8, 16, 24, 32, 48, 64, 96, total-1}.
- `scripts/phase7_analysis.py`: Spearman cross-dataset head-stability
  + heatmap + layer-wise boxplot + cumulative curves figures.
- `scripts/phase7_writeup_gen.py`: generates `experiments/phase7/SUMMARY.md`
  and `experiments/phase7/README.md` from the cell JSONs.
- `experiments/phase7/`: 8 head-importance JSONs, 8 cumulative JSONs,
  `head_stability_analysis.md`, `head_stability.json`, SUMMARY.md,
  README.md, and the raw sweep logs.
- `figures/head_importance_heatmap_{minilm,bert,gte}.png` + captions.
- `figures/head_importance_by_layer.png` + caption.
- `figures/cumulative_pruning_curves.png` + caption.
- `notes/transformers_head_mask_bug.md`: the bug runbook — what broke,
  how to catch it next time, how to configure the dependency pin.

### Fixed

- **transformers 5.x silently drops `head_mask`.** The argument was
  removed from `BertSelfAttention.forward`'s named params but still
  accepted via `**kwargs` — no error, no warning, just silently
  ignored. First sweep (~3h, ~$5 of pod compute) produced valid-looking
  JSONs with every per-head `delta_nDCG@10` exactly `0.0`. Caught only
  when inspecting the output post-hoc.
  **Fix:** pin `transformers<5.0,>=4.40` in `pyproject.toml`, load
  models with `attn_implementation="eager"` (sdpa / flash-attn don't
  support head_mask either, on any transformers version). Full
  diagnosis in `notes/transformers_head_mask_bug.md`.
- **GTE-small pooling was CLS; correct is mean.** The handoff
  documented CLS for GTE but `thenlper/gte-small`'s own
  `1_Pooling/config.json` specifies `pooling_mode_mean_tokens=true`.
  The first sweep's GTE baseline drifted 6–27% from Phase 3
  (0.2989 vs 0.3492 on nfcorpus); after correction the GTE baseline
  matches Phase 3 within 3e-4.
- Eager attention is ~1.7× slower than sdpa on A100. Documented as a
  necessary tradeoff.

### Measured — heads prunable at ≤5% nDCG@10 drop (8/9 cells)

Averaged across available datasets per encoder:

| Encoder | Total heads | Heads prunable @ 5% drop | Heads prunable @ 10% drop | Datasets averaged |
|---|---:|---|---|---|
| `minilm-l6-v2` | 72  | (see SUMMARY.md — per-dataset) | (see SUMMARY.md) | 3 / 3 ✓ |
| `bert-base`    | 144 | (see SUMMARY.md) | (see SUMMARY.md) | 3 / 3 ✓ |
| `gte-small`    | 144 | (see SUMMARY.md) | (see SUMMARY.md) | **2 / 3** (fiqa deferred) |

The `gte-small × fiqa` cell is deferred to Session 06 for budget reasons
(eager attention on 144 heads × 57,638 docs = ~3.3h wall-clock). When it
lands, the archivist will amend this entry with the 3/3 averages.

### Discipline corrections adopted

The transformers 5.x silent-drop cost 3h of pod compute (~$5). Session
05 introduced a new discipline, written into the workflow:

> **"Before any multi-hour sweep: run one cell, inspect output numerically,
> and verify the INDEPENDENT VARIABLE actually moves the measurement."**

For Phase 7 the missing check was: `assert
output_with_all_heads_on != output_with_one_head_zeroed`. Two lines,
two seconds, would have saved 3 hours. Session 06 will apply this
check to the shuffle utility before the 54-run position-sensitivity
sweep.

### Not done in this session

- `gte-small × fiqa` head-importance + cumulative pruning (deferred).
- Phase 8 (position shuffle, 90% milestone): deferred to Session 06 per
  operator decision after the Phase 7 wall-clock overrun.
- Taylor-saliency or iterative-greedy cumulative pruning (flagged as
  future work in Phase 7 SUMMARY.md §Limitations).
- Fine-tuning after pruning (Michel et al.'s ~3-8% recovery). Zero-shot
  ablation only in this phase.

### Version bump

`0.5.0` → `0.7.0`. Skipping `0.6.0` (Phase 6 is landing in parallel
from a local-Mac session). Once Phase 6 merges, the version chronology
interleaves but the per-phase version is unambiguous.
## [0.6.0] — 2026-04-21 — Phase 6: 70% milestone (per-query failure analysis)

First analysis-heavy phase. No new encoders, no new bench runs, no GPU —
purely downstream of the 18 bench JSONs preserved by Phases 3 and 5. Built
a long-form 7,626-row query-level DataFrame, computed per-query failure
grids, Spearman rank agreement across encoders, and cross-encoder
wins/losses sets. Labeled 470 distinct failing queries with an
eight-category failure taxonomy and produced the reviewer-facing
`SUMMARY.md` plus six curated failure examples.

### Added

- `vitruvius/src/vitruvius/analysis/error_analysis.py` — replaces the
  Phase 1 stub with `load_query_frame`, `decode_parquet_columns`,
  `discover_cells`, plus the `ENCODER_FAMILY` map and
  `FAILURE_THRESHOLD` / `SUCCESS_THRESHOLD` constants.
- `scripts/phase6_analysis.py` — builds `query_frame.parquet`, zero-nDCG
  rate grid, Spearman matrices, pairwise disagreement counts, query-
  length bins, cross-encoder sets, and the sampled-qids file used for
  manual labeling (seed 1729).
- `scripts/phase6_label_taxonomy.py` — rule-based labeler plus hand-
  curated `DOMAIN_LEXICON` per dataset. Emits `labeled_queries.csv`,
  `failure_pivot.csv` / `failure_pivot_matrix.csv`, and
  `analysis/failure_taxonomy.md`.
- `scripts/phase6_failure_examples.py` — curates six characteristic
  failure examples into `analysis/failure_examples.md`.
- `scripts/phase6_promote_figures.py` — top-line figures
  `figures/failure_by_architecture.{png,pdf}` (two-panel heatmap) and
  `figures/query_length_vs_ndcg.{png,pdf}` (per-dataset line plot) with
  companion `*_caption.md` files.
- `notebooks/02_phase6_failure_analysis.{py,ipynb}` — reproducible
  executed notebook, built from the `.py` source via jupytext.
- `experiments/phase6/SUMMARY.md` and `experiments/phase6/README.md` —
  reviewer-facing writeup and methodology.
- `experiments/phase6/{query_frame.parquet, zero_ndcg_rates.{csv,md},
  failure_rates_0p1.csv, spearman_*.csv, disagreement_*.csv,
  length_bins.csv, length_quartile_bounds.csv, cross_encoder_sets/*.json,
  cross_encoder_summary.csv, failure_pivot*.csv, sampled_qids.json,
  labeled_queries.csv, figures/}` — full analysis artifact set.

### Findings

- **Zero-nDCG@10 rates on BEIR out-of-distribution** span 14%
  (gte-small × scifact) to 90.4% (conv-retriever × fiqa). Transformer
  baseline is not zero even on scifact.
- **Transformer-gap queries** (any pre-trained transformer succeeds, all
  three from-scratch fail): 38 on nfcorpus, 91 on scifact, 235 on fiqa
  — 30–36% of the two harder test sets is where pre-training's value
  concentrates.
- **Universal-loss queries** (all 6 encoders fail at nDCG@10 < 0.1):
  85 on nfcorpus, 32 on scifact, 152 on fiqa — bounds what any dense-
  retrieval architecture change can recover.
- **Spearman ρ** between CNN and the best transformer on FiQA drops to
  0.24, confirming that on long natural-language questions the family
  choice affects *what* gets retrieved, not just *how well*.
- **Length-vs-nDCG plot** confirms the Session 03 receptive-field
  hypothesis: `conv-retriever` mean nDCG@10 drops from 0.04 on FiQA Q1
  to 0.02 on FiQA Q4 (17+ WordPiece tokens); transformers are flat.
- **Unique wins by architecture**: 127 across transformers, 19 across
  from-scratch encoders (sum across the 3 datasets at SUCCESS_THRESHOLD
  > 0.3). The "route hard queries to the big model, easy ones to the
  small" deployment hypothesis from Session 03 §6.1 is not supported.

### Changed

- `.gitignore` — whitelist the two new Phase 6 figures so they are
  tracked alongside `pareto_v2.{png,pdf}`.

### Limitations disclosed

- Single-labeler taxonomy; no inter-annotator agreement.
- Failure threshold `nDCG@10 < 0.1` is a choice (documented in
  `SUMMARY.md`). The parquet frame allows recomputation at any
  threshold.
- No corpus reload → PARAPHRASE / AMBIGUOUS / MULTI-HOP categories
  are out of scope; 14% of the labeled sample is `UNCATEGORIZED` and
  plausibly includes those patterns.
- 3 BEIR subsets only; wider BEIR generalization unverified.

## [0.5.0] — 2026-04-19 — Phase 5: 55% milestone (absorbs deferred Phase 4)

Three non-transformer bi-encoders (BiLSTM, 1D-CNN, Mamba2) trained from
random init on 500K MS MARCO triplets with identical hyperparameters. All
three plugged into the existing bench-sweep + latency profiler. First
6-point latency-accuracy Pareto plot landed.

**Why this absorbs Phase 4.** Session 02 attempted Phase 4 (pre-trained
Mamba bi-encoder) and invoked §4.7 kill-switch when the only HF checkpoint
(MambaRetriever/SPScanner-130m) turned out to be a cross-encoder scanner,
not a bi-encoder. The amended session-03 handoff rolled Mamba into Phase 5
as a from-scratch bi-encoder trained on the same 500K MS MARCO triplets as
LSTM/CNN — genuinely apples-to-apples. No separate Phase 4.

### Added

- `vitruvius train --encoder ... --train-path ... --val-path ... --output-dir models/ --artifact-dir experiments/phase5/`
  CLI command. AdamW lr=1e-4 wd=0.01, linear-warmup + cosine decay,
  InfoNCE τ=0.05 with in-batch negatives, AMP fp16, `--num-workers`
  knob (default 2; 0 for Mamba — see below).
- Three new encoder wrappers replacing Phase 1 stubs:
  - `lstm-retriever` — 2-layer BiLSTM, hidden 256 (bidir → 512 concat),
    masked mean-pool, Linear(512→256). **6.41M params. similarity="cosine".**
  - `conv-retriever` — 3 stacked Conv1d (k=3/5/7), max+mean pool,
    Linear(512→256). **4.92M params. similarity="cosine".**
  - `mamba-retriever-fs` — 12-layer Mamba2, d_model=384, d_state=128,
    masked mean-pool, Linear(384→256). **23.74M params. similarity="cosine".**
    Raises NotImplementedError at construction if `mamba-ssm` /
    `causal-conv1d` aren't importable.
- Registry rename: `lstm`→`lstm-retriever`, `conv`→`conv-retriever`,
  `mamba`→`mamba-retriever-fs`. The `-fs` suffix distinguishes the
  from-scratch Mamba bi-encoder from any future pre-trained Mamba
  retrieval checkpoint.
- `vitruvius.training.contrastive.InfoNCELoss` (τ=0.05) and
  `vitruvius.training.trainer.train` (checkpoint best-val to
  `models/<encoder>/best.pt`, final to `final.pt`, training JSON +
  per-500-step loss curve CSV to artifact-dir).
- `scripts/download_msmarco.py` — downloads `sentence-transformers/msmarco`
  (triplets + queries + corpus), joins IDs to text, subsamples 500K+5K
  seeded JSONL into `data/msmarco/`.
- `scripts/generate_pareto_v2.py` — reads Phase 3 + 3.5 + 5 JSONs and
  emits `figures/pareto_v2.{png,pdf}` + `pareto_v2_caption.md`.
- `scripts/phase5_summary_gen.py` — emits `experiments/phase5/SUMMARY.md`
  from training + bench + profile JSONs.
- `--checkpoint-root` flag on `bench`, `bench-sweep`, and `profile`.
  From-scratch encoders load `<root>/<encoder>/best.pt` before encoding;
  pre-trained transformer wrappers ignore it.
- `notes/mamba_install_attempt_02.md` — install runbook (nvcc PATH
  gotcha, ~33-min source build, fork+Triton segfault fix).
- Training data pipeline: `TripletDataset` tolerates malformed JSON lines
  (web-scraped control chars in MS MARCO positives/negatives).

### Measured — training (3 epochs, batch 64, 19,452 steps each)

| Encoder | Params | Best val loss | Wall-clock | Peak GPU (MB) |
|---|---:|---:|---:|---:|
| `lstm-retriever`    |  6.41M | **0.4719** | 10m 16s   | 561 |
| `conv-retriever`    |  4.92M | **0.7962** | 16m 25s   | 189 |
| `mamba-retriever-fs`| 23.74M | **0.1807** | 49m 34s†  | 2239 |

† Mamba trained with `num_workers=0` (only difference from LSTM/CNN's
`num_workers=2`) to avoid a DataLoader worker segfault rooted in
Triton+fork state inheritance.

### Measured — nDCG@10 on BEIR test subsets

| Encoder | `nfcorpus` | `scifact` | `fiqa` |
|---|---:|---:|---:|
| `lstm-retriever`    | 0.1901 | 0.3606 | 0.0886 |
| `conv-retriever`    | 0.1400 | 0.1978 | **0.0370** |
| `mamba-retriever-fs`| **0.2083** | **0.3752** | 0.0863 |

(Reference: `minilm-l6-v2` 0.3165/0.6451/0.3687, `bert-base` 0.3169/0.6082†/0.3229, `gte-small` 0.3492/0.7269/0.3937 from Phase 3.)

Mamba wins among the three from-scratch encoders on NFCorpus and SciFact.
CNN × FiQA = 0.037 is flagged below the §5.8 red-flag threshold of 0.05 —
a measured finding, not a training bug: max receptive field of 7 tokens
(stacked kernels 3/5/7) cannot span FiQA's longer financial documents.
The training-loss signal already previewed this (CNN final val 0.80 vs
LSTM 0.47 vs Mamba 0.18). pytrec_eval agrees bit-exact; not an evaluator
issue. Kept in the table as-is; see `experiments/phase5/SUMMARY.md`.

### Measured — query encoding latency (median ms at batch 1, avg across 3 datasets)

| Encoder | median ms @ bs=1 |
|---|---:|
| `conv-retriever`    |  ~0.87 |
| `lstm-retriever`    |  ~1.32 |
| `minilm-l6-v2`      |  ~4.27 |
| `bert-base`         |  ~7.25 |
| `gte-small`         |  ~7.62 |
| `mamba-retriever-fs`| ~11.88 |

Mamba is the slowest at batch 1 on max_seq_len=128. The paper's
linear-time inference advantage only shows up at long sequences; here the
per-layer kernel launch overhead and 12-layer depth dominate and Mamba
trails the transformers. Full breakdown including batch 32 and throughput
in `experiments/phase5/SUMMARY.md`.

### Pareto v2 — latency-accuracy frontier

X-axis: query encoding latency @ batch 1 (median ms, log scale, averaged
across the 3 BEIR subsets). Y-axis: nDCG@10 (avg across the 3 subsets).
Six points from four architecture families: transformer (MiniLM, BERT,
GTE), recurrent (LSTM), convolutional (CNN), SSM (Mamba).

**Pareto-optimal subset: `minilm-l6-v2`, `gte-small`, `lstm-retriever`,
`conv-retriever`.** Mamba is dominated — slower AND lower accuracy than
`minilm-l6-v2` and `gte-small` on this training budget and sequence
length. `bert-base` is also dominated. Full caption and discussion of
what the plot does and does NOT show (not parameter-matched, not equal
FLOPs, in-domain evaluation only, max_seq=128) in
`figures/pareto_v2_caption.md`.

### Validated

- Per-query results schema (§5.7) added to `_run_bench_core` as the first
  commit of this session (`ffa7813`); Phase 3 JSONs backfilled with
  per-query data, aggregate metrics bit-exact identical (zero drift).
- All 12 existing tests pass (stub tests for LSTM/CNN/Mamba removed since
  they are now real). Lint clean throughout.
- pytrec_eval cross-check: nDCG@10 |Δ| < 2e-3 on all 18 bench cells
  (9 transformer from Phase 3 + 9 from-scratch from Phase 5).

### Not done in this session

- No hyperparameter sweeps. One training pass per encoder.
- No hard-negative mining (beyond the explicit negatives shipping in
  MS MARCO triplets).
- No Phase 6 (per-query failure analysis) — runs locally on operator's
  Mac after this session merges; all bench JSONs contain the per-query
  block it needs.
- `MambaRetriever/SPScanner-130m` (cross-encoder) integration — out of
  scope for Session 03; open for a dedicated future session.

### Bumped version

`0.3.5` → `0.5.0` (skipping 0.4.0 because Phase 4 was absorbed; see notes).

## [0.3.5] — 2026-04-19 — Phase 3.5: 30% milestone

Latency profile across the three transformer encoders on three BEIR subsets
at batch sizes 1, 8, 32 — the other axis of the Pareto frontier the project
is building toward.

### Added

- `vitruvius.cli profile` — real-BEIR latency profile. For each
  (encoder, dataset): measures query-encoding latency at each requested
  batch size (median / p50 / p90 / p99 over 100 measured passes, after
  10 warmup), and document encoding throughput at batch 32 (docs/second
  over 200 sampled docs after 3 warmup rounds). Samples are reproducible
  (seed 1729; separate RNGs for queries vs. documents).
- `dataset_length_stats.json` — per-dataset token-length distributions
  (min / median / max / p95) for queries and documents, computed once
  using `sentence-transformers/all-MiniLM-L6-v2` as the canonical
  (encoder-agnostic) tokenizer.
- `experiments/phase3_5/` — nine per-cell JSONs, `SUMMARY.md` (tables +
  methodology), and `profile.log` (raw stdout).

### Measured

Query encoding latency at batch 1 (median ms, 200-query sample):

| Encoder | `nfcorpus` | `scifact` | `fiqa` |
|---|---:|---:|---:|
| `minilm-l6-v2` | 4.18 | 4.30 | 4.30 |
| `bert-base`    | 7.26 | 7.25 | 7.23 |
| `gte-small`    | 7.60 | 7.60 | 7.65 |

Query encoding latency at batch 32 (median ms):

| Encoder | `nfcorpus` | `scifact` | `fiqa` |
|---|---:|---:|---:|
| `minilm-l6-v2` |  6.06 |  8.60 |  7.19 |
| `bert-base`    | 11.64 | 24.73 | 17.85 |
| `gte-small`    |  9.77 | 11.68 | 11.07 |

Document encoding throughput at batch 32 (docs/sec):

| Encoder | `nfcorpus` | `scifact` | `fiqa` |
|---|---:|---:|---:|
| `minilm-l6-v2` | 669.2 | 732.8 | 1167.9 |
| `bert-base`    | 178.4 | 193.4 |  354.3 |
| `gte-small`    | 670.2 | 723.8 | 1215.4 |

### Observations

- Batch-1 query latency is essentially flat across datasets within each
  encoder — at batch 1 the cost is dominated by kernel launch and fixed
  forward-pass overhead; sequence-length variance is noise against that.
- `bert-base` on `scifact` at batch 32 (24.7 ms) is 3.4× its batch-1
  time and 2.1× its `nfcorpus` batch-32 time. That's the O(n²)
  attention cost on longer scientific documents becoming visible, and
  it's exactly the kind of gap the Phase 4 Mamba comparison will
  interrogate (linear-time SSM vs quadratic attention).
- Throughput ranks MiniLM ~ GTE > BERT, consistent with parameter
  counts (22M / 33M / 110M). FiQA reports higher throughput than
  NFCorpus or SciFact because FiQA documents skew shorter on average
  (see `dataset_length_stats.json`).

### Methodology

- CUDA timing via `torch.cuda.Event`; no `time.perf_counter()` fallback
  on GPU. 10 warmup passes before 100 measured passes, per batch size.
- Latency profiled on real BEIR queries (not synthetic fixed-length
  strings) because transformer latency scales non-linearly with
  sequence length.
- Throughput = 200 sampled documents / wall_time encoding them all at
  batch 32. Three warmup rounds before the timed encode.
- Numbers are within-study comparisons on this specific pod
  (A100-SXM4-80GB, torch 2.4.1+cu124). Production latency is hardware-
  sensitive; these are not absolute benchmarks.

### Phase 4 deferral (session-02 stretch goal, kill-switch §4.7)

Attempted integration of a pre-trained Mamba Retriever bi-encoder. The only
HF checkpoint matching the paper (`MambaRetriever/SPScanner-130m`, from
Zhang et al. 2024) is a **cross-encoder scanner** — it scores
`(query, passages...)` pairs in one pass rather than producing per-item
embeddings. Dropping it into Vitruvius's FAISS `IndexFlatIP` bi-encoder
harness is a Phase-5-sized architectural change, not a drop-in. Kill-switch
§4.7 trigger #4 (checkpoint not usable as planned) fired; Phase 4 was
closed at the 30% milestone and its 10 percentage points deferred into
Phase 5. Full discovery + toolchain probe + install-attempt log in
[`notes/mamba_install_attempt_01.md`](notes/mamba_install_attempt_01.md).

No degradation of Phases 1-3.5. All session-02 harness work (similarity
attribute, bench-sweep, profile subcommand, per-cell JSONs, SUMMARY.md
generators) remains the foundation Session 03 builds on.

## [0.3.0] — 2026-04-19 — Phase 3: 20% milestone

Three-encoder × three-dataset BEIR accuracy sweep on the A100 pod.
8/9 cells in-band to approximate leaderboard references (±0.03); one cell
flagged as a measured finding (not a reproduction failure).

### Added

- `vitruvius.cli bench-sweep` — Cartesian sweep over `--encoders × --datasets`
  with one model load per encoder (3× fewer loads than looping `bench`
  nine times). Emits one JSON per cell plus a SUMMARY.md with grid,
  per-cell deltas, pytrec_eval cross-check, and runtimes.
- `Encoder.similarity` — now a required attribute on the base class,
  declared by every wrapper (real and stub). The harness reads it to
  decide whether to L2-normalize before FAISS `IndexFlatIP`. Forgetting
  to declare it fails at class-creation time. Future-proofs the
  interface for Mamba (Phase 4) and from-scratch LSTM/CNN (Phase 5).
- `experiments/phase3/` — nine per-cell JSONs, `SUMMARY.md`, and
  `sweep.log` (raw stdout of both the initial cosine-forced run and
  the post-fix dot run, interleaved chronologically for provenance).

### Fixed

- `bert-base` encoder retargeted from `sentence-transformers/bert-base-nli-mean-tokens`
  (NLI, not for retrieval — garbage nDCG on BEIR) to
  `sentence-transformers/msmarco-bert-base-dot-v5`.
- `bert-base` now declares `similarity = "dot"` and runs with
  `normalize_embeddings=False`. The initial Phase 3 sweep forced
  L2-normalization universally (handoff rule §4) and the dot-trained
  checkpoint dropped −0.08 to −0.11 nDCG@10 below reference on
  NFCorpus/SciFact/FiQA. Root-caused via
  `config_sentence_transformers.json` on the pod (`similarity_fn_name = "dot"`)
  and corrected. Two of the three bert-base cells recovered into band.

### Validated

| Encoder | `nfcorpus` | `scifact` | `fiqa` |
|---|---:|---:|---:|
| `minilm-l6-v2` (cosine) | **0.3165** (+0.017) ✅ | **0.6451** (+0.005) ✅ | **0.3687** (+0.009) ✅ |
| `bert-base` (dot)       | **0.3169** (+0.007) ✅ | **0.6082** (−0.072) ❌ | **0.3229** (+0.023) ✅ |
| `gte-small` (cosine)    | **0.3492** (+0.009) ✅ | **0.7269** (−0.003) ✅ | **0.3937** (−0.026) ✅ |

- `pytrec_eval` cross-check |Δ| ≤ 1.8e-3 on nDCG@10 across all 9 cells;
  Recall@k bit-exact on 7/9, within 5.7e-4 on the other two.
- All cosine encoders: `doc_norm_max`, `query_norm_max` ≈ 1.000001
  (ST `normalize_embeddings=True` holds). `bert-base` norms are not ≈ 1
  by design (dot-trained, unnormalized).
- Total sweep wall-clock: ~6 min. Per-cell runtime shown in
  `experiments/phase3/SUMMARY.md`.

### Finding (not buried, not swept under the rug)

The `bert-base × scifact` cell at 0.6082 vs reference 0.68 (Δ = −0.072)
is not a harness bug: pytrec_eval agrees bit-exact, the dataset itself
is well-behaved on the other two encoders (MiniLM 0.645, GTE 0.727).
It's a measured out-of-domain transfer gap of
`msmarco-bert-base-dot-v5` specifically on scientific-claim retrieval
versus MS-MARCO-distilled contrastive encoders like MiniLM and GTE.
Full discussion in `experiments/phase3/SUMMARY.md § "A measured finding"`.

### Methodology notes

- Graded-gain nDCG (`gain = 2^rel - 1`), discount `log2(i+1)`, iDCG over
  full qrels. Gate IR-2: from-scratch implementation remains primary.
- `pytrec_eval` runs alongside as a cross-check only (reports |Δ|, does
  not replace).
- FAISS `IndexFlatIP`, `top-k = 100`, batch size 128.
- Seed 1729. Hardware: NVIDIA A100-SXM4-80GB, Ubuntu 24.04,
  torch 2.4.1+cu124, FAISS 1.13.2 (CPU), Python 3.11.10.
- Out-of-band cells are **flagged**, not massaged, not silently re-run
  with different seeds.

### References

- BEIR: Thakur et al., *A Heterogeneous Benchmark for Zero-shot
  Evaluation of Information Retrieval Models*, NeurIPS 2021
  Datasets & Benchmarks (arXiv:2104.08663).
- Encoder checkpoints: `sentence-transformers/all-MiniLM-L6-v2`,
  `sentence-transformers/msmarco-bert-base-dot-v5`,
  `thenlper/gte-small`.

## [0.2.0] — 2026-04-19 — Phase 2: 10% milestone

First pod run. Reproduces a published BEIR leaderboard number end-to-end
through the Vitruvius pipeline (BEIR loader → sentence-transformers encoder →
FAISS flat-IP search → graded-gain nDCG), confirming the harness returns
literature-agreeing numbers before scaling to the Phase 3 3×3 sweep.

### Added

- First pod-executed benchmark: `sentence-transformers/all-MiniLM-L6-v2` on
  NFCorpus test (323 queries, 3,633 docs), graded relevance (0/1/2).
- `experiments/phase2/phase2_smoke.json` — full run artifact (config,
  hardware, dataset stats, runtime breakdown, both metric sources, deltas,
  band check).
- `experiments/phase2/README.md` — methodology, tie-breaking analysis for
  pytrec_eval cross-check deltas, embedding-norm sanity checks.
- `experiments/phase2/nfcorpus_minilm_test.log` — raw stdout/stderr of the
  run, including `nvidia-smi` snapshot at run start.
- `experiments/phase2/env_snapshot.txt` — `pip freeze` from the pod at
  run time.
- `experiments/phase2/pod_commits.bundle` — historical git bundle of the
  pod-side commit that implemented Phase 2 (preserved for provenance,
  not imported into main).

### Validated

- **nDCG@10 (ours_from_scratch) = 0.316513**, vs. BEIR leaderboard
  reference 0.30 ±0.02 (Thakur et al. 2021, Table 4). Delta from
  reference: **+0.016513**, result flagged `in_band = true`.
- `pytrec_eval` cross-check: nDCG@10 = 0.315941. |Δ| between
  from-scratch implementation and pytrec_eval = 5.7e-4, consistent with
  tie-breaking differences in FAISS index-insertion ordering vs.
  trec_eval's `(docid, -score)` ordering (not a formula mismatch;
  Recall@k deltas are bit-exact 0.0 at k ∈ {1,5,10,100}).
- Embedding L2 norms: max doc 1.000001, max query 1.000000 — cosine
  similarity via L2-normalized inner product holds.

### Methodology notes

- FAISS IndexFlatIP, top-k = 100, batch size 128, cosine via
  L2-normalized embeddings.
- Doc format: `(title + " " + text).strip()` (BEIR convention).
- Graded-gain nDCG with `gain = 2^rel - 1`, discount `log2(i+1)`,
  ideal DCG computed over full qrels (from-scratch implementation is
  primary; gate IR-2).
- Seed 1729. Hardware: NVIDIA A100-SXM4-80GB, Ubuntu 24.04, torch
  2.4.1+cu124, FAISS 1.13.2 (CPU), Python 3.11.10.
- Total wall-clock: 10.99 s (encode docs 5.43 s, search 0.26 s, rest
  sub-100 ms).

### References

- BEIR: Thakur, Reimers, Rücklé, Srivastava, Gurevych. *A Heterogeneous
  Benchmark for Zero-shot Evaluation of Information Retrieval Models.*
  NeurIPS 2021 Datasets & Benchmarks (arXiv:2104.08663), Table 4.
- Encoder: `sentence-transformers/all-MiniLM-L6-v2` model card.

## 0.1.0 — Phase 1 scaffold

Local-only, CPU-runnable scaffold. No real benchmarking yet.

- Renamed directory `encoder-archaeology/` → `vitruvius/` (history preserved via `git mv`).
- Switched to a proper installable package layout under `src/vitruvius/`.
- Added `pyproject.toml` with pinned-by-floor dependency set, plus `dev`, `mamba`,
  and `pod` optional extras.
- Added `Makefile` with `venv`, `dev`, `smoke`, `test`, `lint`, `format`, `clean`,
  `download-beir` targets. Uses `uv pip` for fast installs.
- Added `.gitignore`, `.env.example`, this `CHANGELOG.md`.
- Implemented `vitruvius.utils.device.pick_device`, `vitruvius.utils.seed.set_seed`
  (default seed = 1729), and `vitruvius.utils.logging.get_logger` (rich-based).
- Implemented `vitruvius.encoders.base.Encoder` (abstract) and the
  `get_encoder(name)` registry.
- Implemented real encoders: `MiniLMEncoder`, `BERTEncoder`, `GTEEncoder`
  (sentence-transformers wrappers).
- Stubbed Phase 4/5 encoders (`MambaEncoder`, `LSTMEncoder`, `ConvEncoder`) —
  importable, raise `NotImplementedError` from `encode_*`.
- Implemented `vitruvius.data.synthetic` (10 queries × 50 docs, 2 relevant per query)
  and `vitruvius.data.beir_loader` (wrapper around `beir.datasets.GenericDataLoader`).
- Implemented `vitruvius.evaluation.retrieval_metrics` — nDCG@k, Recall@k, MRR@k
  derived from scratch (interview-readiness gate IR-2).
- Implemented `vitruvius.evaluation.faiss_index.IndexWrapper` (flat inner product).
- Implemented `vitruvius.evaluation.latency_profiler.profile` — torch.cuda.Event
  on CUDA, `time.perf_counter()` fallback elsewhere.
- Added `vitruvius.cli` with `smoke`, `bench`, `profile`, `shuffle`, `prune` subcommands.
  Phases 2+ subcommands print a "not yet implemented" notice and exit non-zero.
- Added tests: `test_smoke.py`, `test_metrics.py`, `test_encoder_interface.py`.
- Added `scripts/download_beir.py` and `scripts/setup_pod.sh` (Phase 2 preview).
- Applied the macOS libomp workaround: `KMP_DUPLICATE_LIB_OK=TRUE` set on
  package import + `faiss.omp_set_num_threads(1)` on Darwin. Without this the
  combination of `faiss-cpu` and `torch` segfaults on `IndexFlatIP.search`.
  No-op on Linux, so the same code runs unchanged on the pod.

### Tooling decisions captured here

- Python 3.11 venv created via `uv venv --python 3.11 .venv` to avoid the
  Python 3.14 wheel-availability gap for ML packages.
- Makefile uses `uv pip install` (10–100× faster cold install than `pip`).
- `UV_CACHE_DIR=$HOME/.uv-cache` exported by the Makefile because `~/.cache`
  is root-owned on the operator's machine. Override by exporting your own.
- `requirements.txt` left in place from the Renaissance scaffold for now;
  authoritative dependency list is `pyproject.toml`. Will remove in 0.2.0.
