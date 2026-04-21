# Phase 7 — Attention head pruning for retrieval (80% milestone)

What this phase measures: for each of the three pre-trained transformer
retrieval encoders, zero one attention head at a time and measure the
drop in nDCG@10 on three BEIR test subsets. Rank heads by importance,
then cumulatively prune the least-important N heads to characterize the
'how many heads can we remove before retrieval breaks?' curve.

This is the direct, quantified evidence for the Phase 9 opinions
section's `transformers are over-provisioned in heads for retrieval`
claim.

## Headline numbers — heads prunable at ≤N% nDCG@10 drop

(Averaged across NFCorpus / SciFact / FiQA.)

| Encoder | Total heads | Baseline nDCG@10 | Prunable @ ≤5% drop | Prunable @ ≤10% drop |
|---|---:|---:|---:|---:|
| `bert-base` | 144 | 0.4153 | 64.0 / 144 (44.4%) | 90.7 / 144 (63.0%) |
| `minilm-l6-v2` | 72 | 0.4434 | 21.3 / 72 (29.6%) | 26.7 / 72 (37.0%) |
| `gte-small` | 144 | 0.5380 | 64.0 / 144 (44.4%) | 64.0 / 144 (44.4%) |

## Per-dataset headline (5% drop threshold)

| Encoder \ Dataset | nfcorpus | scifact | fiqa |
|---|:---:|:---:|:---:|
| `bert-base` | 64 / 144 (44%) | 96 / 144 (67%) | 32 / 144 (22%) |
| `minilm-l6-v2` | 32 / 72 (44%) | 24 / 72 (33%) | 8 / 72 (11%) |
| `gte-small` | 64 / 144 (44%) | 64 / 144 (44%) | — |

## Cross-dataset head stability (Spearman ρ)

| Encoder | ρ(nfcorpus,scifact) | ρ(nfcorpus,fiqa) | ρ(scifact,fiqa) | Mean ρ |
|---|---:|---:|---:|---:|
| `bert-base` | 0.2194 | 0.1739 | 0.2795 | **0.2243** |
| `minilm-l6-v2` | 0.2732 | 0.3066 | -0.0375 | **0.1808** |
| `gte-small` | 0.1104 | — | — | **0.1104** |

## Layer-wise observations

See [`figures/head_importance_by_layer.png`](../../figures/head_importance_by_layer.png) for per-(encoder, layer) distributions of head importance. Retrieval literature's intuition is that early layers (surface features) should be more prunable than late layers (task-specific). Whether that holds here is encoder-dependent and dataset-dependent; the boxplot is the authoritative view.

## Methodology

- Head ablation via HuggingFace's native `head_mask` argument on `AutoModel.forward` (same API Michel et al. 2019 used). No state-dict hacking, no custom hooks.
- Pre-trained transformer encoders only: `bert-base` (msmarco-bert-base-dot-v5, 144 heads), `minilm-l6-v2` (72 heads), `gte-small` (144 heads). LSTM / CNN / Mamba encoders have no multi-head attention in the transformer sense and are scoped out of this phase.
- Pooling per encoder config: mean for MiniLM and BERT-dot-v5, CLS for GTE.
- `bert-base` keeps dot-product (no L2-norm); MiniLM and GTE use cosine (L2-normalized). Matches each checkpoint's training objective per `Encoder.similarity`.
- BEIR subsets: nfcorpus, scifact, fiqa test splits. FAISS `IndexFlatIP`, top-k=100.
- AMP fp16 autocast for inference speed. Baseline nDCG@10 reproduces Phase 3's numbers within fp16 rounding (|Δ| ≤ 2e-5).
- Cumulative pruning ranks heads by single-head importance (ascending), zeros the N least-important, measures. See limitations below.

## Interpretation

_(Filled in by operator/archivist during Phase 9 writeup synthesis.)_

## Limitations

1. **Single-head-importance ordering used for cumulative pruning.** Heads can compensate for each other, so the true-optimal set of N heads to prune is NOT necessarily the N lowest-scoring-individually. A Taylor-saliency or iterative-greedy baseline would be a stronger claim — future work.
2. **Zero-shot ablation only.** No fine-tuning after pruning. Michel et al. report 3-8% recovery from post-pruning fine-tuning — untested here.
3. **Out-of-distribution evaluation.** BEIR is zero-shot; head importance measured on in-distribution MS MARCO might differ. Scope decision.
4. **AMP fp16 inference.** Inference-side numerics differ from fp32 by ~2e-5 on baseline; order of magnitude below the per-head deltas of interest.

