# Session 05 Report — Vitruvius Pod Phase 7, 2026-04-21 → 22

**Milestone delivered:** Phase 7 at 80%, attention-head pruning characterization on the three pre-trained transformer encoders. **8 of 9 cells complete** — `gte-small × fiqa` deferred to Session 06 due to eager-attention wall-clock cost (3.3h for 144 heads × 57k docs).

## One-sentence outcome

> **44% of BERT-base's attention heads are prunable at ≤5% nDCG@10 drop** (67% at SciFact, 44% at NFCorpus, 22% at FiQA); MiniLM-L6-v2 is only 30% prunable at 5% drop (already distilled); GTE-small is 44% prunable (on 2/3 datasets). **Cross-dataset head-importance stability is low for all encoders (mean Spearman ρ 0.11–0.22)** — the "retrieval-essential head set" does NOT transfer strongly across BEIR domains, which argues against a universal pruning recipe.

## Commits (4 new commits on top of `ed3addd` / v0.5.0)

```
49f94d5  docs(vitruvius): CHANGELOG 0.7.0 + README roadmap 80%→done + transformers pin + version bump
c7475e9  chore(vitruvius/experiments): phase 7 head-pruning artifacts (80% milestone, 8/9 cells)
8b2c54e  fix(vitruvius): transformers 5.x head_mask silent-drop + GTE mean-pool bug
300f1e2  feat(vitruvius/encoders): PrunedTransformerEncoder + phase 7 sweep runner
```

All bundled in `pod_commits.bundle`. **Not pushed.** Archivist merges.

## Headline numbers

### Heads prunable at ≤5% / ≤10% nDCG@10 drop (averaged across available datasets)

| Encoder | Total heads | Baseline nDCG@10 (avg) | @5% drop | @10% drop | Coverage |
|---|---:|---:|:---|:---|---|
| `minilm-l6-v2` |  72 | 0.4434 | 21.3 / 72 (**30%**) | 26.7 / 72 (37%) | 3/3 ✓ |
| `bert-base`    | 144 | 0.4153 | 64.0 / 144 (**44%**) | 90.7 / 144 (63%) | 3/3 ✓ |
| `gte-small`    | 144 | 0.5380 | 64.0 / 144 (**44%**) | 64.0 / 144 (44%) | **2/3** (fiqa deferred) |

### Per-dataset 5%-drop prunability

| Encoder \ Dataset | nfcorpus | scifact | fiqa |
|---|:---:|:---:|:---:|
| `bert-base`    | 64 / 144 (44%) | 96 / 144 (67%) | 32 / 144 (22%) |
| `minilm-l6-v2` | 32 /  72 (44%) | 24 /  72 (33%) |  8 /  72 (11%) |
| `gte-small`    | 64 / 144 (44%) | 64 / 144 (44%) | — (deferred) |

### Cross-dataset head-importance Spearman ρ

| Encoder | ρ(nf, sci) | ρ(nf, fq) | ρ(sci, fq) | Mean ρ |
|---|---:|---:|---:|---:|
| `minilm-l6-v2` |  0.2732 |  0.3066 | -0.0375 | **0.1808** |
| `bert-base`    |  0.2194 |  0.1739 |  0.2795 | **0.2243** |
| `gte-small`    |  0.1104 |    —    |    —    | **0.1104** (1 pair) |

**All three encoders show LOW cross-dataset stability (ρ < 0.25).** This is itself a finding — important-for-retrieval heads are domain-specific, so universal head-pruning recipes would under-serve at least one dataset. Michel et al.'s 2019 result ("60% prunable on BERT") replicated at the dataset level (BERT × SciFact: 67% prunable) but the heads that matter shift between domains.

## Baselines vs Phase 3 (sanity check post-bugfix)

All 8 completed cells match Phase 3 within fp16 AMP noise:

| Cell | Phase 7 baseline | Phase 3 | \|Δ\| |
|---|---:|---:|---:|
| minilm-l6-v2 × nfcorpus | 0.3165 | 0.3165 | 0.0000 |
| minilm-l6-v2 × scifact  | 0.6451 | 0.6451 | 0.0000 |
| minilm-l6-v2 × fiqa     | 0.3687 | 0.3687 | 0.0000 |
| bert-base × nfcorpus    | 0.3174 | 0.3169 | 0.0005 |
| bert-base × scifact     | 0.6025 | 0.6082 | 0.0057 |
| bert-base × fiqa        | 0.3261 | 0.3229 | 0.0032 |
| gte-small × nfcorpus    | 0.3489 | 0.3492 | 0.0003 |
| gte-small × scifact     | 0.7270 | 0.7269 | 0.0001 |

Max drift 6e-3 on bert-base × scifact (fp16 AMP on 768-dim). Gate §7.10 holds — per-head deltas are computed inside the same fp16 pipeline so absolute drift doesn't affect head importance rankings.

## The transformers 5.x head_mask bug (full runbook in `notes/transformers_head_mask_bug.md`)

- **First sweep (attempt 1)**: ran 9 cells to completion, all per-head `delta_nDCG@10` = exactly 0.0. The sweep didn't crash; it silently produced bogus data.
- **Root cause**: transformers 5.5.4 removed `head_mask` from `BertSelfAttention.forward`'s named params. `**kwargs` accepts + discards it. `sdpa` attention (the default) also doesn't support `head_mask` on any transformers version.
- **Fix**: pinned `transformers>=4.40,<5.0` in `pyproject.toml`; loads models with `attn_implementation="eager"`.
- **Cost**: ~3 hours of pod compute (~$5). Detectable in 2 seconds with a `all-ones ≠ single-head-zero` diagnostic that wasn't in the original unit tests.

## GTE pooling fix

`thenlper/gte-small` uses **mean pooling** per its `1_Pooling/config.json`, not CLS. The handoff documented CLS. First sweep's GTE baselines drifted 6–27% (nfcorpus 0.2989 vs Phase 3's 0.3492). Post-fix, GTE baselines match Phase 3 within 3e-4.

## Discipline correction adopted going forward

> **Before any multi-hour sweep: run one cell end-to-end, inspect the output numerically, and verify the independent variable actually moves the measurement.**

For Phase 7 the missing check was `assert ablated_embeddings != baseline_embeddings`. For Phase 8 (tomorrow) the analogous check is `assert shuffled_ndcg != baseline_ndcg` on one cell before launching the 54-run sweep.

## Why `gte-small × fiqa` was deferred

GTE's `max_seq_len=512` (vs MiniLM's 256, BERT's 350) × FiQA's 57,638 docs × 144 heads × eager-attention slowdown = 3.3h per cell. After the 3h bugfix-rerun cost, operator chose to defer and fold the missing cell into Session 06 tomorrow. Financial cost is equal either way (~$5 of pod time); the cleaner methodology is single-session, so the deferral adds a small caveat ("cross-session cell, identical pinned harness, A100 + transformers 4.57.6 + eager + mean-pool, drift within fp16 noise").

Deferred-cell impact:
- GTE's "X% prunable at 5% drop" headline averaged over 2 datasets instead of 3.
- GTE's cross-dataset Spearman is **1 pair** (nfcorpus↔scifact) rather than 3. Weaker stability claim for GTE specifically.
- FiQA is the dataset most likely to reveal different head-importance patterns (longer docs, harder retrieval), so the missing cell is the one that would have most strongly validated domain-generalization for GTE.

## Wall-clock

| Segment | Actual |
|---|---:|
| Env setup, Phase 0 gate (waived Phase 6 dependency per parallel session), head-count sanity | ~20 min |
| PrunedTransformerEncoder + tests + sweep code | ~40 min |
| Head-importance sweep **attempt 1 (bogus due to transformers bug)** | **~3h wasted** |
| Bug diagnosis + transformers downgrade + GTE pool fix | ~20 min |
| Head-importance sweep rerun (8 of 9 cells, gte×fiqa killed) | ~2h 30 min |
| Cumulative pruning (8 cells) | ~25 min |
| Analysis + figures + docs + commits + bundle + scp | ~30 min |
| **Total session** | **~8 hours** |

## Surprises + judgment calls

1. **transformers 5.x silent-drop of `head_mask`.** Wasted 3h of pod time. Fix + discipline rule documented. The unit test gap that allowed it (no `all-ones ≠ single-head-zero` check) is noted prominently in `notes/transformers_head_mask_bug.md` so Session 06 applies the analogous check for the shuffle utility.
2. **GTE uses mean pool, not CLS.** Handoff was wrong; model's own config is the source of truth.
3. **Eager attention is ~1.7× slower than sdpa** — the price of needing head_mask.
4. **bert-base SciFact is 67% prunable** — matches Michel et al. 2019 quite closely. BERT on NFCorpus (44%) and FiQA (22%) less so. **FiQA is much harder to prune** — FiQA's mixed-length financial-domain docs evidently need more of BERT's attention to parse.
5. **MiniLM is less prunable than BERT per-cell (30% vs 44% avg at 5% drop).** Intuition: the distillation that produced MiniLM already eliminated redundant heads. The 72 heads left are more load-bearing individually.
6. **Cross-dataset stability is surprisingly low** (ρ < 0.25 for all encoders). Evidence against a single domain-agnostic "head-pruning recipe" per encoder. Section for the Phase 9 writeup.
7. **gte-small × fiqa deferred** — operator-owned scope decision.

## Phase 6 interaction note (parallel session)

Phase 6 is landing from a separate local-Mac session during Session 05's pod time. Archivist will interleave Phase 6's `v0.6.0` tag with Phase 7's `v0.7.0` on origin. No file conflicts expected (Phase 6 touches `experiments/phase6/`, `analysis/`; Phase 7 touches `experiments/phase7/`, `src/.../pruned_transformer.py`). CHANGELOG + README + version files will merge-conflict and need archivist resolution (normal per established workflow).

## Recommendations for Session 06 (Phase 8 + gte×fiqa backfill)

1. **Lead with `gte-small × fiqa` head-importance + cumulative pruning** (~3.3h) using the pinned harness (transformers 4.57.6 + `attn_implementation="eager"` + mean pool). Then regenerate Phase 7 figures + SUMMARY.md to replace the 2/3 GTE numbers with 3/3.
2. **Before any Phase 8 shuffle sweep**: apply the discipline rule. Run one cell (e.g., minilm × nfcorpus, baseline vs docs-shuffled) and assert the nDCG differs. If it's 0, the shuffle utility is broken — fix before launching 54 runs.
3. **Phase 8 does NOT need eager attention** — token-position shuffle doesn't use head_mask. Use default sdpa for speed. Only the single gte×fiqa Phase 7 backfill needs eager.
4. **From-scratch checkpoints will need re-training** — LSTM (~12 min), CNN (~17 min), Mamba (~50 min with `num_workers=0`). Plus Mamba install (~33 min). Plus MS MARCO re-download (~3 min). ~2h setup before the 54-run shuffle sweep.
5. Phase 8 shuffle utility + `ShuffledEncoder` wrapper + unit tests are pre-written in `experiments/session_05/phase8_prep/`. Ready to ship + commit in Session 06.

## Artifacts in this bundle

```
experiments/session_05/
  SESSION_05_REPORT.md                         (this file)
  pod_commits.bundle                           (4 commits on top of ed3addd)
  session_05_full_log.txt                      (consolidated sweep + cumulative logs)
  phase7/
    head_importance/<encoder>__<dataset>.json × 8
    cumulative_pruning/<encoder>__<dataset>.json × 8
    head_stability.json
    head_stability_analysis.md
    SUMMARY.md
    README.md
    head_sweep.log
    cumulative_sweep.log
  figures/
    head_importance_heatmap_{minilm-l6-v2,bert-base,gte-small}.{png,pdf}
    head_importance_heatmap_{minilm-l6-v2,bert-base,gte-small}_caption.md
    head_importance_by_layer.{png,pdf}
    head_importance_by_layer_caption.md
    cumulative_pruning_curves.{png,pdf}
    cumulative_pruning_curves_caption.md
  notes/
    transformers_head_mask_bug.md              (runbook for Session 06)
  phase8_prep/                                 (pre-written for Session 06)
    shuffle.py                                  token-position shuffle utility
    shuffled.py                                 ShuffledEncoder wrapper
    test_shuffle.py                             unit tests (content multiset, special-token-pinning, determinism)
```
