# Session 06 Report — Vitruvius Pod, 2026-04-22

**Milestones delivered:**
- **Phase 7 closed to 9/9 cells** (backfilled gte×fiqa from Session 05's deferral) → 80% solid.
- **Phase 8 at 90%** — full 54-cell position-sensitivity sweep + analysis + figures + writeup.

## One-sentence outcome

> **The 90% surprise**: LSTM-Retriever is the LEAST position-sensitive family (docs-shuffle drop 0.166), NOT the most — refuting the handoff's `~0.8+` prediction; CNN is the MOST sensitive (0.314); Transformer ≈ SSM (0.21 vs 0.24). The masked mean-pool over BiLSTM hidden states aggregates content-word information heavily enough that retrieval on BEIR is mostly bag-of-concepts-dominated for LSTM too.

## Commits (4 new commits on top of `6194872` / v0.7.0)

```
cbab4c6  docs(vitruvius): CHANGELOG 0.8.0 + README roadmap 90%->done + version bump
bfec0bb  chore(vitruvius/experiments): phase 8 position-sensitivity artifacts (90% milestone, 54/54 cells)
f1b9a56  feat(vitruvius): phase 8 scaffolding — shuffle utility, ShuffledEncoder, sweep + analysis scripts
7915585  chore(vitruvius/experiments): phase 7 gte x fiqa backfill (9/9 cells complete)
```

All bundled in `pod_commits.bundle`. **Not pushed.** Archivist merges.

## Phase 7 close — gte×fiqa backfill (9/9)

- Ran under the Session-05 pinned harness: transformers 4.57.6 + `attn_implementation="eager"` + mean-pool for GTE.
- Baseline nDCG@10 = **0.3935** vs Phase 3's 0.3937 — drift 2e-4 (fp16 AMP noise). Gate holds.
- Cumulative pruning: 5%_at = **32/144 (22%)**, 10%_at = 32/144 — matches the bert-base × fiqa pattern (fiqa is the hardest to prune across encoders).
- Re-generated heatmap + layer-boxplot + cumulative-curves figures to include the 9th cell.
- Cross-dataset Spearman for GTE is now 3 pairs (previously 1 pair); mean ρ = 0.15 (still low — head-importance is domain-specific for GTE just like the other two encoders).
- README P7 row flipped from "done (8/9 cells; gte×fiqa deferred)" to `done`.

## Phase 8 — position shuffle (90%)

### Family-level headline (averaged over encoders-in-family × 3 datasets)

| Family | docs-shuffled | queries-shuffled | both-shuffled |
|---|---:|---:|---:|
| transformer   | **0.211** | 0.104 | 0.268 |
| recurrent     | **0.166** | 0.064 | 0.224 |
| convolutional | **0.314** | 0.129 | 0.346 |
| ssm           | **0.243** | 0.098 | 0.263 |

Higher = more position-dependent. `0.0` = shuffle-invariant; `1.0` = complete collapse.

### docs-shuffled — per-cell detail

| Encoder \ Dataset | nfcorpus | scifact | fiqa |
|---|:---:|:---:|:---:|
| `minilm-l6-v2`        | 0.107 | 0.062 | 0.490 |
| `bert-base`           | 0.131 | 0.050 | 0.325 |
| `gte-small`           | 0.167 | 0.113 | 0.451 |
| `lstm-retriever`      | 0.126 | 0.130 | 0.244 |
| `conv-retriever`      | 0.255 | 0.245 | 0.441 |
| `mamba-retriever-fs`  | 0.150 | 0.277 | 0.303 |

### The surprises

1. **LSTM is the LEAST position-sensitive family**, not the most. Handoff §8.4 predicted "very high (~0.8+)" rationalizing "sequential state updates ARE position — shuffle destroys meaning." Measured docs-shuffle sensitivity is **0.166** — LSTM retains 83% of retrieval quality on shuffled documents. Likely cause: masked mean-pool over bidirectional hidden states aggregates content-word information heavily; for nDCG@10 on content-word-rich BEIR subsets the cumulative-state component dominates the strict-order component.
2. **CNN is the MOST position-sensitive** (0.314). Kernel receptive fields depend on token adjacency; global max+mean pooling doesn't fully recover.
3. **Transformer ≈ SSM** (0.21 vs 0.24). The bag-of-concepts argument for transformers still holds but is not *unique* to them.
4. **FiQA is the dataset-level amplifier** — all 6 encoders have their highest sensitivity on FiQA (range 0.24–0.49 across encoders). NFCorpus and SciFact stay ≤0.27. Hypothesis: FiQA's longer financial-QA documents carry retrieval-relevant information in discourse structure; NFCorpus/SciFact are more content-word-dominant.

### Cross-phase synthesis (offline / Phase 9)

`experiments/phase8/query_level_sensitivity.parquet` has **22,878 per-query rows** (6 encoders × ~300–650 queries × 3 modes × up-to-3 datasets) with baseline/shuffled/delta nDCG@10 and hit@10 flags. Phase 6's failure taxonomy can be joined on `query_id` offline on your Mac — no pod time needed. Natural questions to test: are `LEN-LONG` failures more position-sensitive than `LEN-SHORT`? Are `PARAPHRASE` failures position-insensitive (surface-form shuffle shouldn't change paraphrase)? These join-analyses are the Phase 9 writeup's cross-phase synthesis hooks.

## Discipline held

1. **Pre-sweep sanity check** per the rule adopted after Session 05's transformers-5.x head_mask silent-drop. Before the 54-cell sweep I ran 1 cell (minilm × nfcorpus × docs-shuffled) and confirmed the nDCG@10 differs materially from baseline (**measured 10.7% drop**). Only then kicked off the full sweep. **Zero pod time wasted on broken-data reruns this session.**
2. **Identical shuffle across encoders** — seed 1729, same permutation for a given (dataset, mode, sample) regardless of encoder. Cross-encoder comparison is apples-to-apples.
3. **Special tokens pinned, padding untouched** — verified by 6 unit tests.
4. **pytrec_eval cross-check** on every cell (one per encoder per mode) — on average |delta| ≤ 2e-3 across 54 cells. Evaluator is fine.

## Wall-clock

| Segment | Wall-clock |
|---|---:|
| Gate check + env install (vitruvius + dev deps) | ~5 min |
| Phase 7 gte×fiqa head-importance sweep (144 heads × 82s, eager attn) | ~3.3 h |
| Phase 7 gte×fiqa cumulative pruning (10 points × ~82s) | ~14 min |
| Phase 7 analysis + figures + SUMMARY regen + commit | ~8 min |
| Mamba-ssm install (parallel CPU, overlapped with GPU sweep) | ~33 min (free) |
| Phase 8 sanity: 1-cell diagnostic | ~15 s |
| Phase 8 54-cell shuffle sweep (sdpa, bs=128) | **~33 min** |
| Phase 8 analysis + figures + SUMMARY + commits | ~10 min |
| Session 06 bundle + scp | ~5 min |
| **Total session** | **~4.5 h** |

## Surprises + judgment calls

1. **Pod `/workspace` persisted** from Session 03 — LSTM/CNN/Mamba checkpoints and MS MARCO data still on disk. Saved ~2h of re-training.
2. **Mamba install succeeded on attempt 1 (unpinned, latest)** — parallelized with Phase 7 GPU sweep, zero idle time.
3. **Phase 8 sweep was much faster than estimated** (33 min vs estimated ~1-1.5h). Default sdpa attention + preloaded BEIR splits per dataset meant 36 of the 54 cells were sub-second. FiQA cells dominated (still only ~45s each).
4. **LSTM-least-sensitive surprise** — kept as a headline because it's the project's cleanest result. Refuting the handoff's prediction IS the finding.

## Recommendations for Session 07 (Phase 9 writeup)

1. **Open with the Pareto plot from Phase 5** (`figures/pareto_v2.png`), then layer Phase 7's "heads prunable" numbers onto the transformer points, then the Phase 8 position-sensitivity axis. Three phases compose naturally into one frame: accuracy-latency, parameter-efficiency (head count), position-dependence.
2. **Lead the opinions section with the LSTM-is-bag-of-concepts surprise** — it's the cleanest anti-prior finding and gives a concrete rebuttal to "sequential models need position." Retrieval-specific training has selected LSTM for position-tolerance.
3. **Cross-phase join (Phase 6 × Phase 8)** on `query_id` for the failure-taxonomy × position-sensitivity table. This is a local join, no pod needed — probably an afternoon's work on your Mac before writing.
4. **Budget `v0.9.0`** for Phase 9. Version is at `0.8.0` after today.

## Artifacts in this bundle

```
experiments/session_06/
  SESSION_06_REPORT.md                     (this file)
  pod_commits.bundle                       (4 commits on top of 6194872 / v0.7.0)
  session_06_full_log.txt                  (consolidated: gte×fiqa sweep log + shuffle sweep log)
  phase7_gte_fiqa/                         (copy of the backfilled 9th cell)
    head_importance/gte-small__fiqa.json
    cumulative_pruning/gte-small__fiqa.json
    head_sweep_gte_fiqa.log
  phase8/                                  (full copy of experiments/phase8/)
    <encoder>__<dataset>__<mode>.json × 54
    position_sensitivity.json
    query_level_sensitivity.parquet        (22,878 rows)
    SUMMARY.md
    README.md
    shuffle_sweep.log
  figures/
    head_importance_heatmap_gte-small.{png,pdf}   (updated 3-panel with fiqa)
    head_importance_by_layer.{png,pdf}             (updated to include gte fiqa data)
    cumulative_pruning_curves.{png,pdf}            (updated 9-curve version)
    position_sensitivity.{png,pdf} + caption.md
    sensitivity_by_family.{png,pdf} + caption.md
```
