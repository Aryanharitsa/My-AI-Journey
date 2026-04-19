"""InfoNCE contrastive loss with in-batch negatives.

For retrieval: given a batch of (query, positive) pairs, every positive
passage serves as a negative for the OTHER queries in the batch. The
explicit "negative" column in MS MARCO triplets is treated as an
additional negative (concatenated into the passage tower, same batch).

    loss = -log( exp(sim(q_i, p_i) / tau) /
                 sum_j exp(sim(q_i, p_j) / tau) )

With in-batch negatives for a batch of size B and explicit negs, the
passage tower sees 2B passages so each q has 2B candidates.

Reference: Karpukhin et al. 2020 (DPR), Ni et al. 2022 (GTR).
"""
from __future__ import annotations

import torch
from torch import nn


class InfoNCELoss(nn.Module):
    """Symmetric InfoNCE with in-batch negatives."""

    def __init__(self, temperature: float = 0.05) -> None:
        super().__init__()
        self.tau = temperature

    def forward(
        self,
        q_emb: torch.Tensor,
        p_emb: torch.Tensor,
    ) -> torch.Tensor:
        """q_emb: (B, D) L2-normalized. p_emb: (M, D) L2-normalized, M >= B.

        Assumes p_emb[0:B] are the positives for q_emb[0:B] in order; any
        remaining rows in p_emb are extra negatives (typical: explicit
        hard negatives concatenated after positives).
        """
        B = q_emb.shape[0]
        assert p_emb.shape[0] >= B, f"p_emb must have >= B rows (got {p_emb.shape[0]} vs {B})"
        # Cosine similarity via dot product on unit vectors.
        logits = q_emb @ p_emb.T / self.tau  # (B, M)
        targets = torch.arange(B, device=logits.device)  # positives at diagonal
        return nn.functional.cross_entropy(logits, targets)
