# Project Ramanujan — Literature & References

Papers, resources, and prior work that informed the AIMO3 submission.

## Mathematical Reasoning in LLMs

1. **Tool-Integrated Reasoning (TIR)**
   - Gou et al., "TORA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving," ICLR 2024
   - arXiv: https://arxiv.org/abs/2309.17452
   - *Relevance: Core reasoning strategy used in Ramanujan's sampling pipeline*

2. **Chain-of-Thought Prompting**
   - Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models," NeurIPS 2022
   - arXiv: https://arxiv.org/abs/2201.11903
   - *Relevance: Foundation for structured reasoning approach*

3. **Self-Consistency / Majority Voting**
   - Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in Language Models," ICLR 2023
   - arXiv: https://arxiv.org/abs/2203.11171
   - *Relevance: Basis for N=8 majority voting strategy*

4. **Mathematical Reasoning Benchmarks**
   - Hendrycks et al., "Measuring Mathematical Problem Solving With the MATH Dataset," NeurIPS 2021
   - arXiv: https://arxiv.org/abs/2103.03874

## Inference Infrastructure

5. **vLLM: PagedAttention**
   - Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023
   - arXiv: https://arxiv.org/abs/2309.06180
   - *Relevance: Core inference server used in Ramanujan*

6. **FlashAttention**
   - Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," NeurIPS 2022
   - arXiv: https://arxiv.org/abs/2205.14135
   - *Relevance: Attention backend debugging — FlashInfer dtype issues encountered and resolved*

7. **FlashAttention-2**
   - Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning," 2023
   - arXiv: https://arxiv.org/abs/2307.08691

## Sampling and Verification

8. **Contrastive Decoding**
   - Li et al., "Contrastive Decoding: Open-ended Text Generation as Optimization," ACL 2023
   - arXiv: https://arxiv.org/abs/2210.15097
   - *Relevance: Inspired contrastive challenger verification approach*

9. **Scaling LLM Test-Time Compute**
   - Snell et al., "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters," 2024
   - arXiv: https://arxiv.org/abs/2408.03314
   - *Relevance: Theoretical backing for adaptive compute budget strategy*

## Model Architecture

10. **Attention Is All You Need**
    - Vaswani et al., "Attention Is All You Need," NeurIPS 2017
    - arXiv: https://arxiv.org/abs/1706.03762

11. **Mixture of Experts**
    - Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer," ICLR 2017
    - arXiv: https://arxiv.org/abs/1701.06538
    - *Relevance: GPT-OSS-120B architecture context*
