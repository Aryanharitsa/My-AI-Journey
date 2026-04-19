# My AI Journey — Research Portfolio

Active research projects exploring how neural network architectures shape computation — from pushing inference limits on fixed architectures, to analyzing what trained models actually compute internally, to comparing encoder architectures for retrieval.

## Projects

### [Project Ramanujan](./PROJECT_RAMANUJAN_AIMO_3/)
Solo submission to AIMO Prize 3 (Kaggle). Score: 36/50 in 3 weeks. GPT-OSS-120B + vLLM + Harmony inference stack with N=8 majority voting and TIR reasoning.

### [Mesa-Optimization Probe](./mesa-probe/)
Empirical investigation into whether small transformers trained on in-context learning tasks develop internal optimization-like circuits. Uses TransformerLens and activation patching on synthetic ICL tasks.

### [Project Vitruvius](./vitruvius/)
Comparative study of transformer vs non-transformer encoder architectures for dense retrieval, focused on the latency–accuracy Pareto frontier under production constraints. **Status:** 30% milestone reached (Phase 3.5 latency profile) on an A100 pod; `minilm-l6-v2 × nfcorpus` reproduces BEIR reference nDCG@10 = 0.3165 bit-exact across sessions. Phase 4 (Mamba) deferred and re-scoped into Phase 5's from-scratch bi-encoder training per the §4.7 kill-switch — see [`vitruvius/notes/mamba_install_attempt_01.md`](./vitruvius/notes/mamba_install_attempt_01.md).

---
Aryan D Haritsa | PES University Bengaluru | [GitHub](https://github.com/Aryanharitsa) | [LinkedIn](https://www.linkedin.com/in/aryan-haritsa-60292925b)
