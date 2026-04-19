# Session 02 Report — Vitruvius Pod Work, 2026-04-19

**Final status:** Session closed at **30% milestone**. Phases 3 and 3.5 landed clean.
Phase 4 (stretch) deferred per §4.7 kill-switch — SPScanner-130m is a cross-encoder,
not a bi-encoder. Deferral absorbed into Session 03's amended Phase 5 plan.

## Milestones reached

- **20% (Phase 3):** 3 encoders × 3 BEIR subsets accuracy sweep — done.
- **30% (Phase 3.5):** 3 encoders × 3 datasets × 3 batch sizes latency profile — done.
- **40% (Phase 4):** deferred → absorbed into 55%. See `notes/mamba_install_attempt_01.md`.

## Commits landed on the pod (in order, on top of `origin/main`@`b4de557`)

| SHA (pod) | Type | Summary |
|---|---|---|
| `dbac3a3` | refactor | Retarget `bert-base` registry key to `msmarco-bert-base-dot-v5` |
| `6ba71a5` | refactor | Require `similarity` attribute on `Encoder` base class |
| `8c6272c` | fix | `bert-base` `similarity="dot"`, skip L2-norm for dot-trained encoders |
| `69b93a0` | feat | Phase 3 3×3 bench sweep — 20% milestone (cli + artifacts) |
| `d73b8ef` | docs | CHANGELOG [0.3.0] + README roadmap 20%→done |
| `4878da9` | feat | Phase 3.5 latency profile — 30% milestone |
| `501397e` | docs | CHANGELOG [0.3.5] + README roadmap 30%→done |
| `c66ff06` | docs | Phase 4 deferred per §4.7 (SPScanner is cross-encoder) |

All commits are bundled in `pod_commits.bundle` alongside this report. **Not pushed.**

## Headline numbers

### Phase 3 — nDCG@10 (graded-gain, from-scratch; pytrec_eval cross-check |Δ| ≤ 1.8e-3)

| Encoder | `nfcorpus` | `scifact` | `fiqa` |
|---|---:|---:|---:|
| `minilm-l6-v2` (cosine) | 0.3165 (+0.017) ✅ | 0.6451 (+0.005) ✅ | 0.3687 (+0.009) ✅ |
| `bert-base` (dot)       | 0.3169 (+0.007) ✅ | **0.6082 (−0.072) ❌** | 0.3229 (+0.023) ✅ |
| `gte-small` (cosine)    | 0.3492 (+0.009) ✅ | 0.7269 (−0.003) ✅ | 0.3937 (−0.026) ✅ |

**8/9 in band (±0.03 of BEIR leaderboard approx).** One flagged: `bert-base × scifact`.
Not massaged. Full story in `experiments/phase3/SUMMARY.md` (includes the pre-fix
cosine-forced numbers and the dot-vs-cosine investigation).

### Phase 3.5 — Query encoding latency at batch 1, median ms (200-query sample, CUDA events, 10 warmup + 100 measured)

| Encoder | `nfcorpus` | `scifact` | `fiqa` |
|---|---:|---:|---:|
| `minilm-l6-v2` | 4.18 | 4.30 | 4.30 |
| `bert-base`    | 7.26 | 7.25 | 7.23 |
| `gte-small`    | 7.60 | 7.60 | 7.65 |

Batch-32 latency on `bert-base × scifact` is 24.7 ms — the visible O(n²)
attention cost on longer scientific documents, a data point for the
Phase 4/5 Mamba comparison. Document throughput and percentiles in
`experiments/phase3_5/SUMMARY.md`.

## Notable surprises + decisions

1. **Origin pulled Phase 2 into main as `b4de557` between sessions.** Pod branched
   from `b312b1f` (before the merge) and had its own `478ca89` Phase 2 commit.
   Handled by `git rebase origin/main` + `--skip` for the content-equivalent
   `478ca89`, then resolving one cli.py conflict by taking the Phase-3 version
   (which includes the Phase 2 bench-code + Phase 3 sweep).

2. **Two "out of band" events on `bert-base`, different causes:**
   - *First*: all 3 cells −0.08 to −0.11 due to forcing L2-norm on a
     dot-trained checkpoint. Fixed by adding `Encoder.similarity` as a
     required attribute and flipping bert-base to `"dot"`. Two cells
     recovered.
   - *Second*: `bert-base × scifact` remains −0.072 out of band. **pytrec_eval
     agrees bit-exact**; the dataset itself is fine for the other two
     encoders. This is a measured out-of-domain transfer gap of
     `msmarco-bert-base-dot-v5` specifically on scientific-claim retrieval,
     not a harness bug. Flagged openly in SUMMARY.md and CHANGELOG per
     operator's explicit "don't bury this" instruction.

3. **MooseFS hang during the phase3 sweep re-run.** `/workspace` FUSE went
   `request_wait_answer` on one cell, leaving a 0-byte
   `gte-small__nfcorpus.json`. Process entered `D` (uninterruptible)
   state; `kill -9` was a no-op. Re-ran the cell in a fresh process
   (other cells' JSONs were unaffected). Regenerated SUMMARY.md from
   all 9 JSONs on disk. No data loss.

4. **Phase 4 architectural mismatch.** `MambaRetriever/SPScanner-130m`
   is a cross-encoder "Single-Pass Scanner", not a bi-encoder.
   Kill-switch #4 fired by spirit (not literal); operator confirmed
   and folded Phase 4 into amended Session 03 (three from-scratch
   bi-encoders trained on MS MARCO).

5. **Toolchain gotcha for Session 03:** `nvcc` is at
   `/usr/local/cuda/bin/nvcc` but not on PATH by default. `nvcc --version`
   returned "command not found" during the probe, falsely suggesting the
   CUDA toolkit was missing. The subsequent `pip install mamba-ssm` source
   build found and used nvcc fine. Session 03 should set
   `export PATH=/usr/local/cuda/bin:$PATH` before probing.

## Recommendations for Session 03

1. **Gate check first** (§0 of amended handoff): confirm `c66ff06` is on origin.
2. **Set `PATH` for CUDA before probing.** Avoid the misleading "nvcc not found"
   diagnostic.
3. **Attempt mamba-ssm install with the fallback pins** recommended in §5.4
   of the amended handoff (`mamba-ssm==2.2.2 causal-conv1d==1.4.0`) — latest
   versions compiled from source cleanly in Session 02 (estimated 15–25 min
   total) but were never allowed to complete, so the fallback may be safer.
4. **Implement §5.7 (per-query results in bench JSONs) as the first commit
   after gate check** — it's a small schema extension to `_run_bench_core` in
   `cli.py` and the Phase 3 JSONs need a backfill pass afterward. The
   amended handoff calls this out as mandatory for Phase 6.
5. **Watch MooseFS.** If `/workspace` hangs again, fresh processes can
   usually read unaffected files — rerun affected cells in a new process
   rather than trying to recover the hung one.

## Artifacts in this bundle

```
experiments/session_02/
  SESSION_02_REPORT.md                           (this file)
  pod_commits.bundle                             (8 commits, apply on origin/main)
  phase3/
    SUMMARY.md, sweep.log, <encoder>__<dataset>.json × 9
  phase3_5/
    SUMMARY.md, dataset_length_stats.json, profile.log, <encoder>__<dataset>.json × 9
  notes/
    mamba_install_attempt_01.md                  (reads first, next session)
```

Checkpoints (none this session — Phase 5 will produce them). Pod-side
build artifacts for the aborted mamba-ssm compile are disposable and will
vanish when the pod is stopped.
