# Encoder Archaeology

**Research Question:** How does encoder architecture choice affect the latency-accuracy Pareto frontier in dense retrieval? Where does the transformer's full bidirectional attention provide genuine value, and where is it wasted compute?

## Motivation

Dense retrieval systems rely on deep encoder models to produce embeddings for similarity search. In production, these encoders operate under hard latency constraints — and under that pressure, the transformer quietly becomes a bottleneck. The standard response is distillation: smaller transformers that trade accuracy for speed. But this accepts a framing we want to challenge:

**What if the transformer is the wrong inductive bias for retrieval in the first place?**

Retrieval differs from language modeling in ways that matter architecturally:
- **Asymmetric compute**: Document encoding is offline; query encoding is latency-critical
- **Output modality**: The encoder produces a single fixed vector, not a token sequence
- **Position sensitivity**: Retrieval may be more bag-of-concepts than sequence-order-dependent
- **Task structure**: Compressing meaning into a dot-product-comparable vector is fundamentally different from next-token prediction

This project empirically compares transformer, SSM (Mamba), recurrent (LSTM), and convolutional (CNN) encoders on standard retrieval benchmarks, measuring both accuracy AND latency, and asks: where on the Pareto frontier do non-transformer architectures actually sit?

## Approach

1. Evaluate 4-6 encoder architectures on BEIR subsets (NFCorpus, SciFact, FiQA)
2. Profile inference latency at multiple batch sizes alongside retrieval accuracy
3. Analyze *where* architectures differ: per-query error analysis, attention head pruning, position sensitivity
4. Produce an opinionated analysis of what retrieval actually needs from its encoder

## Connection to MSR India Research

This project was motivated by the [MSR India research agenda on alternative encoder architectures for dense retrieval](https://www.microsoft.com/en-us/research/lab/microsoft-research-india/). The question — "what assumptions are we making when we reach for a transformer?" — is the starting point for this work.

## Status

🔨 **Active development** — Literature survey complete, benchmarking infrastructure in progress.

## References

See [LITERATURE.md](./LITERATURE.md) for the full reading list.
