# Encoder Archaeology — Literature Survey

## Dense Retrieval Foundations

1. **Dense Passage Retrieval for Open-Domain Question Answering (DPR)**
   - Karpukhin, Oguz, Min, Lewis, Wu, Edunov, Chen, Yih
   - EMNLP 2020
   - arXiv: https://arxiv.org/abs/2004.04906
   - *Established the bi-encoder paradigm for dense retrieval using BERT. Baseline architecture we're interrogating.*

2. **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks**
   - Reimers, Gurevych
   - EMNLP 2019
   - arXiv: https://arxiv.org/abs/1908.10084
   - *Made BERT practical for similarity search via siamese/bi-encoder training. Still the most common production pattern.*

3. **ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT**
   - Khattab, Zaharia
   - SIGIR 2020
   - arXiv: https://arxiv.org/abs/2004.12832
   - *Late interaction architecture — preserves per-token representations instead of compressing to single vector. Important architectural alternative within the transformer family.*

4. **ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction**
   - Santhanam, Khattab, Saad-Falcon, Potts, Zaharia
   - NAACL 2022
   - arXiv: https://arxiv.org/abs/2112.01488

5. **SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking**
   - Formal, Piwowarski, Clinchant
   - SIGIR 2021
   - arXiv: https://arxiv.org/abs/2107.05720
   - *Learned sparse retrieval — shows that sparse representations can compete with dense. Relevant to the "do we need dense at all?" question.*

## Efficient Encoders and Distillation

6. **MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers**
   - Wang, Wei, Dong, Bao, Yang, Zhou
   - NeurIPS 2020
   - arXiv: https://arxiv.org/abs/2002.10957
   - *Standard distilled encoder used in production retrieval. Key comparison point on the latency-accuracy frontier.*

7. **Matryoshka Representation Learning**
   - Kusupati, Bhatt, Rege, Wallingford, Sinha, Ramanujan, Howard-Snyder, Chen, Kakade, Jain, Farhadi
   - NeurIPS 2022
   - arXiv: https://arxiv.org/abs/2205.13147
   - Code: https://github.com/RAIVNLab/MRL
   - *Nested representations that allow adaptive compute at inference. Orthogonal to architecture choice — could be applied to any encoder. Important efficiency technique.*

## Non-Transformer Architectures

8. **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**
   - Gu, Dao
   - arXiv: https://arxiv.org/abs/2312.00752
   - *The leading non-transformer sequence model. Linear time complexity in sequence length. Key question: does this advantage translate to retrieval encoders?*

9. **Mamba Retriever: Utilizing Mamba for Effective and Efficient Dense Retrieval**
   - Zhang, Chen, Mei, Liu, Mao
   - arXiv: https://arxiv.org/abs/2408.08066
   - *Directly tests Mamba as a retrieval encoder. Shows comparable effectiveness to transformer on classic retrieval and advantage on long-text retrieval. Core related work.*

10. **RetNet: Retentive Network: A Successor to Transformer for Large Language Models**
    - Sun, Dong, Huang, Ma, Xia, Xue, Wang, Wei
    - arXiv: https://arxiv.org/abs/2307.08621
    - *Another non-transformer alternative with O(n) inference. Retention mechanism as alternative to attention.*

11. **RWKV: Reinventing RNNs for the Transformer Era**
    - Peng, Alcaide, Anthony, Albalak, Arcadinho, Cao, Cheng, Chung, Grella, GV, He, Hesse, et al.
    - EMNLP Findings 2023
    - arXiv: https://arxiv.org/abs/2305.13048
    - *Linear-complexity RNN with transformer-level performance. Not yet tested as retrieval encoder — potential novel contribution.*

12. **M2-BERT: Revisiting BERT for Retrieval**
    - Fu, Dao, Saab, Thomas, Rudra, Re
    - arXiv: https://arxiv.org/abs/2310.07831
    - *Uses state space models (specifically Monarch Mixer) as BERT-replacement for retrieval. Directly relevant prior work.*

## Benchmarks and Evaluation

13. **BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models**
    - Thakur, Reimers, Ruckle, Srivastava, Gurevych
    - NeurIPS 2021 Datasets and Benchmarks
    - arXiv: https://arxiv.org/abs/2104.08663
    - GitHub: https://github.com/beir-cellar/beir
    - *Standard retrieval benchmark. We use NFCorpus, SciFact, and FiQA subsets for tractable experimentation.*

14. **MS MARCO: A Human Generated MAchine Reading COmprehension Dataset**
    - Nguyen, Rosenberg, Song, Gao, Tiwary, Majumder, Deng
    - NeurIPS 2016 Workshop
    - arXiv: https://arxiv.org/abs/1611.09268
    - *Largest retrieval training dataset. Most pre-trained retrieval encoders are fine-tuned on this.*

## Architectural Analysis and Inductive Biases

15. **Attention Is All You Need**
    - Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin
    - NeurIPS 2017
    - arXiv: https://arxiv.org/abs/1706.03762
    - *The architecture we're questioning. Key: the original design was for sequence-to-sequence translation, not embedding compression.*

16. **Are Sixteen Heads Really Better than One?**
    - Michel, Levy, Neubig
    - NeurIPS 2019
    - arXiv: https://arxiv.org/abs/1905.10650
    - *Shows many attention heads can be pruned without performance loss. Directly motivates our attention pruning analysis on retrieval encoders.*

17. **Efficiently Modeling Long Sequences with Structured State Spaces (S4)**
    - Gu, Goel, Re
    - ICLR 2022
    - arXiv: https://arxiv.org/abs/2111.00396
    - *Foundation for Mamba and modern SSM architectures. Establishes the state space model paradigm.*
