# Project Ramanujan — AIMO Prize 3

Solo submission to the AI Mathematical Olympiad Progress Prize 3 on Kaggle. Built a multi-stage inference pipeline around GPT-OSS-120B in 3 weeks, scoring 36/50 on problems ranging from national olympiad to IMO difficulty.

## Result

- **Public Leaderboard Score:** 36/50 ({RANK_TBD} place)
- **Out-of-box GPT-OSS-120B baseline:** ~20/50
- **Previous year's winning score (AIMO2):** 34/50
- **Time to build:** 3 weeks (entered late March 2026; competition ran since Nov 2025)
- **Team size:** Solo

## What's Technically Interesting

- **Inference stack debugging under pressure.** Got GPT-OSS-120B (116.8B params, 5.1B active MoE) running on vLLM with Harmony transport, debugging FlashInfer fp8 dtype assertions, VLLM_ATTENTION_BACKEND enum changes between 0.10.x and 0.11.x, KV cache manager misconfigs, and a startup timing race condition — all within the first week.

- **Systematic failure mode characterization.** Identified 5 distinct failure classes in LLM mathematical reasoning through controlled experiments on 30+ curated problems: F1 attractor traps (model converges 5/8 on a wrong answer that's off by a clean factor), F3 long-horizon case analysis (context window exhaustion on 20+ case problems), F4 existential search (model settles on suboptimal bounds), F5 generation-miss walls (correct technique appears in <1/8 samples).

- **Entropy-weighted majority voting.** Replaced naive majority vote with `weight = 1/max(entropy, 1e-9)` per sample, where entropy is computed from top-5 token logprobs during streaming. Adopted from the 43/50 public notebook technique. Combined with early stopping at 4/8 consensus to save compute budget for harder problems.

- **Controlled ablation of 4 post-processing techniques** (PACER revision, DeepConf Online confidence filtering, GenSelect comparative selection, LeaP peer exchange). Result: none improved over baseline voting — the model's built-in stochastic diversity at T=1.0 already decorrelates errors sufficiently. LeaP actively damaged performance by amplifying wrong-majority consensus. This negative result is itself informative.

- **44/50 solver adaptation.** Integrated the top public notebook's solver architecture (streaming Harmony completions, Jupyter sandbox pool, inline `\boxed{}` detection) into the submission pipeline, achieving 8/10 on curated hard-problem benchmarks before competition-day variance.

## Stack

GPT-OSS-120B (MXFP4) | vLLM 0.19.0 | OpenAI Harmony Protocol | N=8 Parallel Sampling | Entropy-Weighted Voting | Tool-Integrated Reasoning (Jupyter) | Early Stop at 4/8 Consensus | Contrastive Prompting (selective) | RunPod A100/H100

## Repo Map

| Document | What it covers |
|---|---|
| **[STORY.md](./STORY.md)** | The 3-week sprint, week by week — what worked, what broke, what was learned |
| **[TECHNICAL_REPORT.md](./TECHNICAL_REPORT.md)** | Full methods and results in short-paper format |
| **[ARCHITECTURE.md](./ARCHITECTURE.md)** | System design, data flow, and component interactions |
| **[LESSONS.md](./LESSONS.md)** | Research insights as short essays — failure modes, infrastructure debugging, what RL-trained models resist |
| **[research/](./research/)** | Dataset analysis, pre-seed prompting research, contrastive prompting findings, selector suite evolution |
| **[experiments/](./experiments/)** | Ablation results, score progression, failure analysis |
| **[src/](./src/)** | Core pipeline code — inference, sampling, verification, scoring, prompts |

## How to Reproduce

**Hardware:** 1x NVIDIA A100-SXM4-80GB (or H100) with 200GB+ disk for model weights.

```bash
# 1. Clone and install
git clone https://github.com/Aryanharitsa/My-AI-Journey.git
cd My-AI-Journey/PROJECT_RAMANUJAN_AIMO_3
pip install -r requirements.txt

# 2. Download model and start vLLM
python -c "from huggingface_hub import snapshot_download; snapshot_download('openai/gpt-oss-120b', cache_dir='./hf_cache')"
python -m vllm.entrypoints.openai.api_server \
  --model ./hf_cache/models--openai--gpt-oss-120b/snapshots/<hash> \
  --served-model-name openai/gpt-oss-120b \
  --max-model-len 65536 --max-num-seqs 8 --enforce-eager --trust-remote-code

# 3. Run inference
python src/solver_44_standalone.py  # Connects to vLLM on :8000
```

**Kaggle submission:** The competition notebook is in `notebooks/`. It handles vLLM startup, model loading, and the 5-hour inference loop automatically on Kaggle's H100 runtime.

## Acknowledgments

AIMO Prize 3 competition on Kaggle. OpenAI GPT-OSS-120B model. The [43/50 public notebook by kurianbenoy](https://www.kaggle.com/code/kurianbenoy/43-50-aimo-3-gpt-oss-120b-weighted-entropy) for the entropy-weighted voting technique. The [44/50 notebook by nihilisticneuralnet](https://www.kaggle.com/code/nihilisticneuralnet/skills-optional-luck-required) for the streaming solver architecture. [vLLM](https://github.com/vllm-project/vllm) for the inference server. RunPod for GPU compute during development.

---

*Aryan D Haritsa | PES University Bengaluru | April 2026*
