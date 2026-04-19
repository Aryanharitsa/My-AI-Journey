# Pareto v2 — caption

**X-axis:** query encoding latency at batch size 1, median ms measured via `torch.cuda.Event` (10 warmup + 100 measured passes per dataset), averaged across NFCorpus / SciFact / FiQA test splits (200 sampled queries each). Log scale (dynamic range > 10×).

**Y-axis:** nDCG@10 using the from-scratch `retrieval_metrics.evaluate` (graded-gain, `2^rel - 1` / `log2(i+1)`, iDCG over full qrels), averaged across the same three test splits. pytrec_eval cross-check was run but not plotted.

**Points** (6):
- `minilm-l6-v2` (transformer): latency 4.26 ms, nDCG@10 0.4434 ⋆ Pareto-optimal. Per-dataset nDCG [0.316513, 0.645082, 0.368671]. Per-dataset latency [4.1768479347229, 4.302799940109253, 4.304015874862671].
- `bert-base` (transformer): latency 7.25 ms, nDCG@10 0.4160. Per-dataset nDCG [0.316871, 0.608237, 0.322878]. Per-dataset latency [7.2567360401153564, 7.249248027801514, 7.229439973831177].
- `gte-small` (transformer): latency 7.61 ms, nDCG@10 0.4899 ⋆ Pareto-optimal. Per-dataset nDCG [0.349164, 0.72693, 0.393748]. Per-dataset latency [7.60099196434021, 7.5963521003723145, 7.646015882492065].
- `lstm-retriever` (recurrent): latency 1.32 ms, nDCG@10 0.2131 ⋆ Pareto-optimal. Per-dataset nDCG [0.190144, 0.360594, 0.08855]. Per-dataset latency [0.9638240039348602, 1.4516000151634216, 1.5429120063781738].
- `conv-retriever` (convolutional): latency 0.87 ms, nDCG@10 0.1249 ⋆ Pareto-optimal. Per-dataset nDCG [0.139984, 0.197786, 0.036954]. Per-dataset latency [0.8399040102958679, 0.8792479932308197, 0.8780640065670013].
- `mamba-retriever-fs` (ssm): latency 11.88 ms, nDCG@10 0.2232. Per-dataset nDCG [0.208259, 0.375194, 0.086297]. Per-dataset latency [11.81876802444458, 11.95641565322876, 11.873583793640137].

**Pareto-optimal subset**: `conv-retriever`, `gte-small`, `lstm-retriever`, `minilm-l6-v2`.

**What this plot shows:** The latency-accuracy frontier across transformer (blue family), recurrent BiLSTM (orange), 1D-CNN (green), and SSM/Mamba2 (red) architectures, all evaluated via the same FAISS `IndexFlatIP` retrieval harness. Transformer encoders are pre-trained checkpoints (MiniLM, MSMARCO-BERT-dot, GTE-small); the non-transformer encoders are trained from scratch on the same 500K MS MARCO triplets with identical hyperparameters (InfoNCE τ=0.05, AdamW lr=1e-4, 3 epochs, batch 64).

**What this plot does NOT show:** (1) throughput at batch 32 — see `experiments/phase5/SUMMARY.md` and `experiments/phase3_5/SUMMARY.md`. (2) per-query failure modes — that's Phase 6. (3) comparison at equal FLOPs or equal parameter counts — the three from-scratch encoders have ~5-25M parameters vs. ~22-110M for the pre-trained transformers, so they're not parameter-matched and any frontier claim is about *this specific training budget*, not about architecture classes in general. (4) out-of-distribution robustness — all cells use BEIR test splits that overlap MS MARCO's domain.

Source JSONs: `experiments/phase3/`, `experiments/phase3_5/`, `experiments/phase5/`.
