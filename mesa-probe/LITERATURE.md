# Mesa-Optimization Probe — Literature Survey

## Core Papers (Must-Read, Directly Replicated/Extended)

1. **Transformers Learn In-Context by Gradient Descent**
   - von Oswald, Niklasson, Randazzo, Sacramento, Mordvintsev, Zhmoginov, Vladymyrov
   - ICML 2023
   - arXiv: https://arxiv.org/abs/2212.07677
   - Code: https://github.com/google-research/self-organising-systems/tree/master/transformers_learn_icl_by_gd
   - *Key finding: Linear self-attention implements GD. Trained transformers converge to this construction. Extends to multi-layer (preconditioned GD / curvature correction).*
   - *Our extension: We verify this with TransformerLens-based mechanistic analysis rather than weight inspection alone, and test on non-linear function classes.*

2. **What Can Transformers Learn In-Context? A Case Study of Simple Function Classes**
   - Garg, Tsipras, Liang, Valiant
   - NeurIPS 2022
   - arXiv: https://arxiv.org/abs/2208.01066
   - Code: https://github.com/dtsip/in-context-learning
   - *Key finding: Transformers can be trained from scratch to ICL linear functions, sparse linear functions, 2-layer NNs, and decision trees. Performance matches or exceeds task-specific algorithms.*
   - *Our use: We adopt their training setup as the baseline and extend with interpretability analysis.*

3. **Do Pretrained Transformers Learn In-Context by Gradient Descent?**
   - Shen, Mishra, Khashabi
   - arXiv: https://arxiv.org/abs/2310.08540
   - *Key finding: Challenges the GD equivalence claim — shows the theoretical constructions require strong assumptions that may not hold in practice. Important counterpoint.*
   - *Our use: Motivates careful empirical verification rather than assuming GD equivalence.*

## Mesa-Optimization Theory

4. **Risks from Learned Optimization in Advanced Machine Learning Systems**
   - Hubinger, van Merwijk, Mikulik, Skalse, Garrabrant
   - arXiv: https://arxiv.org/abs/1906.01820
   - *The foundational mesa-optimization paper. Defines mesa-optimizers, inner alignment, and deceptive alignment. Provides the theoretical frame for why ICL-as-optimization matters for safety.*

5. **Progress Measures for Grokking via Mechanistic Interpretability**
   - Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, Jacob Steinhardt
   - ICLR 2023
   - arXiv: https://arxiv.org/abs/2301.05217
   - *Demonstrates mechanistic interpretability on a specific learned algorithm (modular addition). Methodological template for our analysis approach.*

## Mechanistic Interpretability Methods

6. **A Mathematical Framework for Transformer Circuits**
   - Elhage, Nanda, Olsson, Henighan, Joseph, Mann, Askell, Bai, Chen, Conerly, DasSarma, Drain, Ganguli, Hatfield-Dodds, Hernandez, Jones, Kernion, Lovitt, Ndousse, Amodei, Brown, Clark, Kaplan, McCandlish, Olah
   - Transformer Circuits Thread, 2021
   - URL: https://transformer-circuits.pub/2021/framework/index.html
   - *Foundational framework for understanding transformer computations as circuits. Provides vocabulary and methods we use throughout.*

7. **In-context Learning and Induction Heads**
   - Olsson, Elhage, Nanda, Joseph, DasSarma, Henighan, Mann, Askell, Bai, Chen, Conerly, Drain, Ganguli, Hatfield-Dodds, Hernandez, Johnston, Jones, Kernion, Lovitt, Ndousse, Amodei, Brown, Clark, Kaplan, McCandlish, Olah
   - Transformer Circuits Thread, 2022
   - URL: https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
   - *Identifies induction heads as key mechanism for ICL. Von Oswald et al. show induction heads can be understood as a special case of ICL-by-GD.*

8. **Toy Models of Superposition**
   - Elhage, Hume, Olsson, Schiefer, Henighan, Kravec, Hatfield-Dodds, Lasenby, Drain, Chen, Johnston, Grosse, McCandlish, Kaplan, Amodei, Wattenberg, Olah
   - Transformer Circuits Thread, 2022
   - URL: https://transformer-circuits.pub/2022/toy_model/index.html
   - *Characterizes superposition in neural networks. Relevant methodology: training tiny models, analyzing geometric structure of learned representations.*

## Tools

9. **TransformerLens**
   - Neel Nanda et al.
   - GitHub: https://github.com/TransformerLensOrg/TransformerLens
   - Docs: https://transformerlensorg.github.io/TransformerLens/
   - *Primary interpretability toolkit. Provides hooks for activation caching, attention pattern extraction, and activation patching on transformer models.*

10. **200 Concrete Open Problems in Mechanistic Interpretability**
    - Neel Nanda, 2022
    - URL: https://www.alignmentforum.org/posts/LbrPTJ4fmABEdEnLf/200-concrete-open-problems-in-mechanistic-interpretability
    - *Reference list of open problems. Problems #15-25 on in-context learning algorithms are directly relevant.*

## Additional Context

11. **An Explanation of In-context Learning as Implicit Bayesian Inference**
    - Xie, Raghunathan, Liang, Ma
    - ICLR 2022
    - arXiv: https://arxiv.org/abs/2111.02080
    - *Alternative theory: ICL as Bayesian inference rather than gradient descent. Important competing hypothesis to test against.*

12. **Why Can GPT Learn In-Context? Language Models Secretly Perform Gradient Descent as Meta-Optimizers**
    - Dai, Sun, Dong, Hao, Ma, Sui, Wei
    - ACL Findings 2023
    - arXiv: https://arxiv.org/abs/2212.10559
    - *Shows dual form between attention and GD in pretrained models (not just toy ones). Extends the von Oswald result to practical scale.*
