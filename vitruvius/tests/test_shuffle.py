"""Unit tests for the token-shuffle utility (Phase 8 foundation).

Four invariants per handoff §8.1 unit-test spec:
(a) set(shuffled_content) == set(original_content) for content positions
(b) special tokens at identical positions
(c) padding positions unchanged
(d) same seed -> same shuffle
"""
from __future__ import annotations

import torch

from vitruvius.utils.shuffle import shuffle_input_ids

SPECIAL = {101, 102, 0}  # BERT: [CLS]=101, [SEP]=102, [PAD]=0


def _sample_batch(seed: int = 7):
    torch.manual_seed(seed)
    B, L = 4, 16
    ids = torch.randint(low=200, high=5000, size=(B, L))
    # Put [CLS] at 0, [SEP] at some position, pad the rest
    valid_lens = torch.tensor([L, 12, 8, 5])
    attn = torch.zeros(B, L, dtype=torch.long)
    for b in range(B):
        n = int(valid_lens[b].item())
        ids[b, 0] = 101
        ids[b, n - 1] = 102
        ids[b, n:] = 0
        attn[b, :n] = 1
    return ids, attn


def test_content_multiset_preserved():
    ids, attn = _sample_batch()
    shuffled = shuffle_input_ids(ids, attn, SPECIAL, seed=1729)
    for b in range(ids.shape[0]):
        ids_row = ids[b].tolist()
        sh_row = shuffled[b].tolist()
        content_orig = sorted(i for i, a in zip(ids_row, attn[b].tolist(), strict=True)
                              if a == 1 and i not in SPECIAL)
        content_sh = sorted(i for i, a in zip(sh_row, attn[b].tolist(), strict=True)
                            if a == 1 and i not in SPECIAL)
        assert content_orig == content_sh, f"batch {b}: content multiset changed"


def test_special_positions_unchanged():
    ids, attn = _sample_batch()
    shuffled = shuffle_input_ids(ids, attn, SPECIAL, seed=1729)
    for b in range(ids.shape[0]):
        for i, (orig, new) in enumerate(zip(ids[b].tolist(), shuffled[b].tolist(), strict=True)):
            if orig in SPECIAL:
                assert new == orig, f"batch {b} pos {i}: special token moved {orig}->{new}"


def test_padding_unchanged():
    ids, attn = _sample_batch()
    shuffled = shuffle_input_ids(ids, attn, SPECIAL, seed=1729)
    pad_mask = (attn == 0)
    assert torch.equal(shuffled[pad_mask], ids[pad_mask])


def test_determinism():
    ids, attn = _sample_batch()
    s1 = shuffle_input_ids(ids, attn, SPECIAL, seed=1729)
    s2 = shuffle_input_ids(ids, attn, SPECIAL, seed=1729)
    assert torch.equal(s1, s2)


def test_different_seeds_differ():
    ids, attn = _sample_batch()
    s1 = shuffle_input_ids(ids, attn, SPECIAL, seed=1729)
    s2 = shuffle_input_ids(ids, attn, SPECIAL, seed=777)
    # Not strictly required, but we expect at least one position to differ
    assert not torch.equal(s1, s2)


def test_degenerate_short_sequences():
    # 1 content token can't be shuffled — should pass through unchanged
    ids = torch.tensor([[101, 999, 102, 0, 0, 0]])
    attn = torch.tensor([[1, 1, 1, 0, 0, 0]])
    s = shuffle_input_ids(ids, attn, SPECIAL, seed=1729)
    assert torch.equal(s, ids)
