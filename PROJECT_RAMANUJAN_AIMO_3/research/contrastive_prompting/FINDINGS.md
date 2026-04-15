# Contrastive Prompting — Findings

## Technique
Developer prompt asks the model to "consider both a correct approach and one common wrong approach, explaining why the wrong approach fails" before solving.

## Results
- **Positive signal:** On `discordant_perms_14_mod` (borderline problem, baseline 4/8 correct), contrastive improved to 7/8 correct samples. The "explain what could go wrong" instruction activated verification reasoning.
- **Zero effect on knowledge walls:** Problems where the model lacks the technique (tournament kings, Petersen chromatic polynomial) showed 0/8 across all configs.
- **Negative on some problems:** On `dd7f5e` (shifty functions), contrastive got 0/8 while vanilla normal got 1/4. The overhead of error analysis consumed tokens that could have gone to exploration.
- **Token cost:** ~15% more reasoning tokens per sample compared to baseline.

## Mechanism
The "Large Language Models are Contrastive Reasoners" paper (arXiv 2403.08211) showed that "Let's give a correct and a wrong answer" improved GSM8K from 35.9% to 88.8%. On RL-trained reasoning models, the effect is smaller because they already have built-in verification loops.

## Recommendation
Use selectively on problems where the model is likely to hit attractor traps (factor-of-2 errors, off-by-one). Do not apply universally — it hurts exploration on novel problems.
