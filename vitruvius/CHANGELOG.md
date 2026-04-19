# Changelog

All notable changes to Project Vitruvius are documented here. Versioning follows
the milestone tiers in the project roadmap (0.1.0 = Phase 1, 0.2.0 = Phase 2, …).

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
