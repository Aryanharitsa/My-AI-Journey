# Selector Suite — Evolution

## Techniques Tested

### PACER (Post-hoc Revision)
- **Mechanism:** When vote margin < 4, show each sample the consensus and ask it to revise or confirm.
- **Result:** 0 rescues, 0 damages across all experiments. PACER never disagreed with the solver's consensus.
- **Critical bug found:** On the Hamiltonian paths problem, PACER didn't fire because early stopping produced a 4/4 consensus on the wrong answer. The 3 correct samples were invisible to PACER.

### GenSelect (Comparative Selection)
- **Mechanism:** Show all 8 candidates to the model, ask "which is correct?" 16 times with permuted ordering.
- **Result:** Rubber-stamped majority vote on every problem, including wrong ones. All 16 trials picked 276 (wrong) over 552 (correct) on P10.
- **Insight:** The model-as-judge cannot distinguish between plausible wrong answers and correct ones when both are computationally valid.

### LeaP (Peer Exchange)
- **Mechanism:** 3 rounds of generation with peer summary exchange between parallel attempts.
- **Result:** Actively damaged P9 — the 7 wrong peers overwhelmed the 1 correct sample. Net -1 vs baseline.
- **Insight:** Peer exchange amplifies majority opinion, which hurts when the majority is wrong.

### DeepConf Online (Confidence Filtering)
- **Mechanism:** Mid-generation termination of low-confidence traces based on sliding-window logprob analysis.
- **Implementation:** Per paper (Fu et al., Meta AI, 2025). 16 warmup traces, 90th percentile threshold, 2048-token window.
- **Result:** On easy problems (95%+ consensus), termination never triggered. Needs problems in 50-80% solve range to show signal.

## Conclusion
Model-level generation quality dominates all post-processing techniques. The only lever that demonstrably helps on hard problems is more independent samples — and that's a compute question, not an algorithm question.
