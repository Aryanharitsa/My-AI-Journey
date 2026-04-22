"""Token-position shuffle utility for Phase 8 position-sensitivity experiment.

Shuffles content-token positions within each sequence AFTER tokenization
but BEFORE encoding. Special tokens ([CLS], [SEP], [PAD], etc.) are kept at
their original positions; padding positions are not touched. Same seed
across all encoders for a given (dataset, mode) ensures cross-encoder fair
comparison.

See handoff §8.1 for the rationale — shuffling words in raw text is dirty
because multi-subword tokens split unpredictably; shuffling special tokens
confuses models that rely on [CLS]-position pooling.
"""
from __future__ import annotations

import torch


def shuffle_input_ids(
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    special_token_ids: set[int],
    seed: int,
) -> torch.LongTensor:
    """Shuffle non-special, non-padding tokens within each sequence.

    Args:
        input_ids: (B, L) long tensor.
        attention_mask: (B, L) long tensor; 0 = padding, 1 = valid token.
        special_token_ids: ids of tokenizer-special tokens ([CLS], [SEP], etc.).
        seed: deterministic permutation seed. Same (seed, sample position)
            produces the same shuffle.

    Returns:
        (B, L) long tensor with content tokens permuted in-place, special
        tokens + padding at original positions.
    """
    out = input_ids.clone()
    B, L = input_ids.shape
    g = torch.Generator(device="cpu")
    for b in range(B):
        g.manual_seed(seed + b)
        ids_row = input_ids[b].tolist()
        attn_row = attention_mask[b].tolist()
        content_positions = [
            i for i in range(L)
            if attn_row[i] == 1 and ids_row[i] not in special_token_ids
        ]
        if len(content_positions) <= 1:
            continue  # nothing to shuffle
        perm = torch.randperm(len(content_positions), generator=g).tolist()
        shuffled_ids = [ids_row[content_positions[p]] for p in perm]
        for dst, new_id in zip(content_positions, shuffled_ids):
            out[b, dst] = new_id
    return out
