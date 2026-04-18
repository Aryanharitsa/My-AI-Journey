# Changelog

All notable changes to Project Vitruvius are documented here. Versioning follows
the milestone tiers in the project roadmap (0.1.0 = Phase 1, 0.2.0 = Phase 2, …).

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
