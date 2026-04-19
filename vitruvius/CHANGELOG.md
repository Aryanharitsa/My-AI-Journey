# Changelog

All notable changes to Project Vitruvius are documented here. Versioning follows
the milestone tiers in the project roadmap (0.1.0 = Phase 1, 0.2.0 = Phase 2, …).

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
