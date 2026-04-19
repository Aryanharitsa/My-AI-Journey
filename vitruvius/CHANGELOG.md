# Changelog

All notable changes to Project Vitruvius are documented here. Versioning follows
the milestone tiers in the project roadmap (0.1.0 = Phase 1, 0.2.0 = Phase 2, …).

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
