# Mesa-Optimization Probe — Methodology

## Phase 1: Reproduce Baseline (Week 1, ~15 hours)

### 1.1 Synthetic Data Generation
- Generate in-context linear regression sequences following Garg et al.:
  - Input dimension d = 5, 10, 20
  - Sequence length (number of in-context examples) = 5, 10, 20, 40
  - Random weight vectors w ~ N(0, I_d), inputs x ~ N(0, I_d), outputs y = w^T x + noise
- Each training sequence: [(x_1, y_1), (x_2, y_2), ..., (x_k, y_k), x_query] → y_query

### 1.2 Model Training
- Architecture: Decoder-only transformer, 1-4 layers, 2-8 heads, hidden dim 64-256
- Training: Standard autoregressive loss on y values only
- Optimizer: AdamW, cosine schedule, ~50K-100K steps
- Validation: Hold-out function classes (unseen w vectors)
- **Checkpoint at multiple training stages** for grokking analysis

### 1.3 Baseline Verification
- Confirm model achieves ICL performance matching or exceeding OLS/ridge regression
- Plot: ICL loss vs number of in-context examples (should decrease, matching OLS curve)
- This step ensures the setup is correct before doing interpretability

## Phase 2: Mechanistic Analysis (Week 2, ~20 hours)

### 2.1 Attention Pattern Analysis
- Using TransformerLens hooks, extract attention patterns for each layer/head
- Visualize: Do attention heads attend to (x, y) pairs in structured ways?
- Compare to theoretical prediction: In the GD construction, attention weights should encode X^T X-like computations
- Key question: Does the attention pattern change with number of in-context examples?

### 2.2 Activation Patching
- For each layer and component (attention, MLP), run activation patching:
  - Replace activations from a "clean" run with activations from a "corrupted" run (different function)
  - Measure effect on output — which components are causally necessary?
- Produce causal graph of which heads/layers matter for ICL

### 2.3 Weight Comparison to GD Construction
- Following von Oswald et al., extract W_QK and W_OV matrices
- Compare to the theoretical construction: W_P = eta * (W_OV @ W_QK) should approximate gradient step
- Measure: Frobenius norm distance between learned weights and GD construction
- Track this distance across training (does it converge toward GD construction?)

### 2.4 Probing for Algorithm Identity
- Train linear probes on intermediate activations to predict:
  - The OLS solution (w_OLS = (X^T X)^{-1} X^T y)
  - The ridge regression solution
  - The gradient descent iterate at step t
  - The nearest-neighbor prediction
- Which algorithm's output is best predicted by intermediate representations?
- This is the key experiment that produces the main finding

## Phase 3: Extension and Writeup (Week 3, ~15 hours)

### 3.1 Non-Linear Function Classes
- Repeat Phase 2 analysis on models trained on:
  - Sparse linear regression (L1-constrained)
  - 2-layer neural network functions
- Does the internal algorithm change when the task changes?

### 3.2 Writeup
- 4-6 page technical report
- Structure: Motivation → Setup → Results → Discussion → Limitations
- Key visualizations: attention patterns, activation patching results, probe accuracy comparison
- Honest about null results or unexpected findings

## Success Criteria

- **Minimum viable output:** Reproduce von Oswald baseline + one interpretability analysis (attention patterns or probing) with clear visualization
- **Target output:** Full probing experiment with algorithm identification claim + attention analysis + clean writeup
- **Stretch output:** Non-linear extension + comparison to Bayesian inference hypothesis
