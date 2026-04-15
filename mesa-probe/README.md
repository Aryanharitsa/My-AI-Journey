# Mesa-Optimization Probe

**Research Question:** Do small transformers trained on in-context learning tasks implement gradient-descent-like optimization in their forward pass? Can we characterize the internal algorithm using mechanistic interpretability tools?

## Motivation

Von Oswald et al. (2023) showed theoretically that a single linear self-attention layer can implement one step of gradient descent, and provided empirical evidence that trained transformers converge to this construction. Garg et al. (2022) demonstrated that transformers can be trained to in-context learn various function classes. But a key question remains open: **what algorithm do trained transformers actually implement?** Is it gradient descent, ridge regression, nearest-neighbor lookup, or something else entirely?

This project uses mechanistic interpretability tools — attention pattern analysis, activation patching, and weight comparison — to empirically characterize the internal algorithm in small (1-4 layer) transformers trained on synthetic ICL tasks.

## Approach

1. Train small transformers on synthetic in-context linear regression (following Garg et al.)
2. Use TransformerLens to extract and analyze internal representations
3. Compare internal computations to known algorithms (GD, ridge regression, Bayesian inference)
4. Test whether the internal algorithm changes across task complexity (linear → non-linear)

## Connection to Broader Research

This work connects to:
- **Mesa-optimization** (Hubinger et al., 2019): Are trained models learning to optimize internally?
- **Mechanistic interpretability**: Can we reverse-engineer learned algorithms from weights and activations?
- **Architectural inductive biases**: How does the transformer architecture constrain what algorithms can be learned?

## Status

🔨 **Active development** — Literature survey complete, implementation in progress.

## References

See [LITERATURE.md](./LITERATURE.md) for the full reading list.
