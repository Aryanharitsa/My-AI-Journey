"""Unit tests for the head-mask wrapper.

Two key invariants (handoff §7.2):
1. `head_mask = all-ones` produces output that matches the un-patched base
   encoder (torch.allclose at 1e-5).
2. `head_mask = all-zeros` does not crash; output is deterministic (may be
   nonsensical for retrieval but must be a well-formed embedding).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu

from vitruvius.encoders.pruned_transformer import (  # noqa: E402
    PrunedTransformerEncoder,
    get_base_encoder_info,
)


@pytest.fixture(scope="module")
def tiny_texts():
    return ["what is dense retrieval?", "how does attention work"]


@pytest.mark.parametrize("alias", ["minilm-l6-v2"])  # small, fast in CI
def test_all_ones_matches_base_encoder(alias: str, tiny_texts):
    """head_mask=ones should match the un-patched base encoder's output."""
    from sentence_transformers import SentenceTransformer

    info = get_base_encoder_info(alias)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Raw base
    base = SentenceTransformer(info["hf_id"], device=device)
    base.eval()
    base_emb = base.encode(
        tiny_texts,
        convert_to_numpy=True,
        normalize_embeddings=(info["similarity"] == "cosine"),
        show_progress_bar=False,
    ).astype(np.float32)

    # Wrapped with head_mask=ones
    head_mask = torch.ones(info["num_layers"], info["num_heads"])
    pruned = PrunedTransformerEncoder(alias, head_mask, device=device)
    pruned_emb = pruned.encode_queries(tiny_texts)

    assert pruned_emb.shape == base_emb.shape
    np.testing.assert_allclose(pruned_emb, base_emb, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("alias", ["minilm-l6-v2"])
def test_all_zeros_does_not_crash(alias: str, tiny_texts):
    """head_mask=zeros must not crash; output must be finite + deterministic."""
    info = get_base_encoder_info(alias)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head_mask = torch.zeros(info["num_layers"], info["num_heads"])
    pruned = PrunedTransformerEncoder(alias, head_mask, device=device)

    emb_a = pruned.encode_queries(tiny_texts)
    emb_b = pruned.encode_queries(tiny_texts)

    assert emb_a.shape == (len(tiny_texts), pruned.embedding_dim)
    assert np.all(np.isfinite(emb_a))
    np.testing.assert_allclose(emb_a, emb_b)  # determinism


def test_shape_validation():
    """Mismatched head_mask shape raises."""
    info = get_base_encoder_info("minilm-l6-v2")
    bad_mask = torch.ones(info["num_layers"] + 1, info["num_heads"])
    with pytest.raises(ValueError, match="head_mask shape"):
        PrunedTransformerEncoder("minilm-l6-v2", bad_mask, device="cpu")


def test_unknown_encoder_raises():
    with pytest.raises(ValueError, match="Unknown base encoder"):
        get_base_encoder_info("nonexistent-encoder")
