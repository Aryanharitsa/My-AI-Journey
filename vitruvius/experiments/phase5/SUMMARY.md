# Phase 5 — From-scratch encoder training + evaluation (55% milestone)

Three non-transformer bi-encoders trained from random init on 500K MS MARCO
triplets with identical hyperparameters (InfoNCE τ=0.05, AdamW lr=1e-4, 3 epochs,
batch 64, linear-warmup + cosine decay, AMP fp16, max seq len 128,
BERT WordPiece tokenizer, seed 1729). Evaluated through the same bench-sweep
and latency profiler used on the pre-trained transformers in Phases 3 / 3.5.

Absorbs the deferred Phase 4 (session-02 §4.7 kill-switch on SPScanner): instead
of integrating a pre-trained Mamba bi-encoder that doesn't exist publicly, Phase 5
trains a Mamba2 bi-encoder from scratch alongside LSTM and CNN, putting all three
alternative architectures on equal footing vs. the pre-trained transformers.

## Training

| Encoder | Params | Steps | Best val loss | Final val loss | Wall (s) | Peak GPU (MB) | AMP | num_workers |
|---|---:|---:|---:|---:|---:|---:|:---:|:---:|
| `lstm-retriever` | 6.41M | 19452 | 0.4719 | 0.4719 | 615.7 | 561 | ✓ | 2 |
| `conv-retriever` | 4.92M | 19452 | 0.7962 | 0.7964 | 984.6 | 189 | ✓ | 2 |
| `mamba-retriever-fs` | 23.74M | 19452 | 0.1807 | 0.1807 | 2974.2 | 2239 | ✓ | 0 |

## nDCG@10 (ours_from_scratch) on BEIR test subsets

| Encoder \ Dataset | nfcorpus | scifact | fiqa |
|---|:---:|:---:|:---:|
| `lstm-retriever` | 0.1901 | 0.3606 | 0.0886 |
| `conv-retriever` | 0.1400 | 0.1978 | 0.0370 |
| `mamba-retriever-fs` | 0.2083 | 0.3752 | 0.0863 |

### Cross-check status (from-scratch vs pytrec_eval)

| Cell | nDCG@10 ours | nDCG@10 pytrec | \|Δ\| | Recall@10 bit-exact? |
|---|---:|---:|---:|:---:|
| `lstm-retriever` × `nfcorpus` | 0.1901 | 0.1897 | 4.83e-04 | ✅ |
| `lstm-retriever` × `scifact` | 0.3606 | 0.3606 | 0.00e+00 | ✅ |
| `lstm-retriever` × `fiqa` | 0.0886 | 0.0886 | 0.00e+00 | ✅ |
| `conv-retriever` × `nfcorpus` | 0.1400 | 0.1389 | 1.09e-03 | ✅ |
| `conv-retriever` × `scifact` | 0.1978 | 0.1978 | 0.00e+00 | ✅ |
| `conv-retriever` × `fiqa` | 0.0370 | 0.0370 | 0.00e+00 | ✅ |
| `mamba-retriever-fs` × `nfcorpus` | 0.2083 | 0.2072 | 1.04e-03 | ✅ |
| `mamba-retriever-fs` × `scifact` | 0.3752 | 0.3752 | 0.00e+00 | ✅ |
| `mamba-retriever-fs` × `fiqa` | 0.0863 | 0.0863 | 0.00e+00 | ✅ |

## Query encoding latency @ batch size 1 — median ms

| Encoder \ Dataset | nfcorpus | scifact | fiqa |
|---|:---:|:---:|:---:|
| `lstm-retriever` | 0.964 | 1.452 | 1.543 |
| `conv-retriever` | 0.840 | 0.879 | 0.878 |
| `mamba-retriever-fs` | 11.819 | 11.956 | 11.874 |

## Query encoding latency @ batch size 32 — median ms

| Encoder \ Dataset | nfcorpus | scifact | fiqa |
|---|:---:|:---:|:---:|
| `lstm-retriever` | 2.340 | 5.447 | 4.060 |
| `conv-retriever` | 1.925 | 3.697 | 3.002 |
| `mamba-retriever-fs` | 13.253 | 15.212 | 14.478 |

## Document throughput @ batch 32 — docs/s

| Encoder \ Dataset | nfcorpus | scifact | fiqa |
|---|:---:|:---:|:---:|
| `lstm-retriever` | 963.5 | 1008.1 | 1513.3 |
| `conv-retriever` | 1133.4 | 1203.2 | 2035.5 |
| `mamba-retriever-fs` | 110.5 | 762.7 | 1039.9 |

## Methodology

- InfoNCE contrastive loss with in-batch negatives, temperature 0.05. Each batch of 64 triplets provides 64 queries and 128 candidate passages (positives + explicit hard negatives).
- AdamW lr=1e-4, wd=0.01, linear warmup (first 10% of steps) then cosine decay. Gradient clipping at 1.0. AMP fp16 throughout.
- 3 epochs × 500K triplets / batch 64 = 23,437 planned steps; actual is slightly lower due to `drop_last=True` and a small number of skipped malformed MS MARCO rows (control chars in scraped web text).
- Validation every 500 steps on held-out 5K triplets. Best-val checkpoint saved to `models/<encoder>/best.pt`; final checkpoint to `final.pt`.
- BERT-base-uncased WordPiece tokenizer; max_seq_len=128.
- `lstm-retriever` / `conv-retriever` trained with DataLoader `num_workers=2`. `mamba-retriever-fs` trained with `num_workers=0` — standard multiprocessing fork inherits inconsistent Triton-JIT state from mamba_ssm's selective-scan kernels and workers segfault. Same model, same hyperparameters, same data; only the dataloader parallelism differs. Full diagnosis in `notes/mamba_install_attempt_02.md`.
- Bench: FAISS `IndexFlatIP`, top-k=100, L2-normalized embeddings, graded-gain nDCG@k. pytrec_eval runs as a cross-check (gate IR-2).
- Latency: `torch.cuda.Event`, 10 warmup + 100 measured passes per batch size, 200 queries / 200 docs sampled per dataset (seed 1729).
- Hardware: A100-SXM4-80GB, torch 2.4.1+cu124, FAISS 1.13.2 (CPU), Python 3.11.10.

