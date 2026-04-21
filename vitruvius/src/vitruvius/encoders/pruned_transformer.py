"""Runtime attention-head ablation wrapper.

Wraps any of the three pre-trained transformer encoders (minilm-l6-v2,
bert-base, gte-small) with a head_mask that zeros selected attention heads
during encoding. Uses HuggingFace's native `head_mask` argument (the same
API Michel et al. 2019 used), forwarded through sentence-transformers by
monkey-patching the underlying AutoModel's forward.

Phase 7 head-importance ranking and cumulative pruning sweep both use this
wrapper — it is a runtime wrapper, no fine-tuning, no checkpoint.

Usage:

    head_mask = torch.ones(num_layers, num_heads)   # all-ones = baseline
    head_mask[4, 7] = 0                             # zero layer 4 head 7
    enc = PrunedTransformerEncoder("bert-base", head_mask, device="cuda")
    emb = enc.encode_queries(["..."])
"""
from __future__ import annotations

import numpy as np
import torch

from vitruvius.encoders.base import Encoder
from vitruvius.utils.device import pick_device
from vitruvius.utils.logging import get_logger

_log = get_logger(__name__)


# Registry: alias -> (HF id, similarity, pooling, total layers, heads/layer).
# Values confirmed by inspecting each model's config.json on the pod.
_BASE_ENCODER_TABLE = {
    "minilm-l6-v2": {
        "hf_id": "sentence-transformers/all-MiniLM-L6-v2",
        "similarity": "cosine",
        "num_layers": 6,
        "num_heads": 12,
    },
    "bert-base": {
        "hf_id": "sentence-transformers/msmarco-bert-base-dot-v5",
        "similarity": "dot",
        "num_layers": 12,
        "num_heads": 12,
    },
    "gte-small": {
        "hf_id": "thenlper/gte-small",
        "similarity": "cosine",
        "num_layers": 12,
        "num_heads": 12,
    },
}


def get_base_encoder_info(alias: str) -> dict:
    """Return {hf_id, similarity, num_layers, num_heads} for a supported alias."""
    if alias not in _BASE_ENCODER_TABLE:
        raise ValueError(
            f"Unknown base encoder {alias!r}. "
            f"Supported: {sorted(_BASE_ENCODER_TABLE)}"
        )
    return dict(_BASE_ENCODER_TABLE[alias])


def list_prunable_encoders() -> list[str]:
    return sorted(_BASE_ENCODER_TABLE)


class PrunedTransformerEncoder(Encoder):
    """A transformer encoder with selected attention heads zeroed at inference.

    The similarity attribute is inherited from the base encoder — bert-base
    stays "dot" (no L2-norm), minilm/gte stay "cosine" (L2-norm applied).
    """

    def __init__(
        self,
        base_encoder_name: str,
        head_mask: torch.Tensor,
        device: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        info = get_base_encoder_info(base_encoder_name)
        if head_mask.shape != (info["num_layers"], info["num_heads"]):
            raise ValueError(
                f"head_mask shape {tuple(head_mask.shape)} does not match "
                f"{base_encoder_name} ({info['num_layers']}, {info['num_heads']})"
            )

        self._base_name = base_encoder_name
        self._name = f"pruned-{base_encoder_name}"
        self._similarity = info["similarity"]
        self._device = pick_device(device)
        self._num_layers = info["num_layers"]
        self._num_heads = info["num_heads"]

        self._st = SentenceTransformer(info["hf_id"], device=str(self._device))
        # Identify the Transformer module (always first child in the ST pipeline).
        transformer_module = self._st[0]
        self._auto = transformer_module.auto_model
        self._embedding_dim = self._auto.config.hidden_size

        # Monkey-patch forward to inject the head_mask on every call. Bound
        # as a lambda-style closure referencing `self._head_mask` so updating
        # the tensor in-place (or reassigning) takes effect without re-patching.
        self._head_mask = head_mask.to(self._device)
        original_forward = self._auto.forward

        def patched_forward(*args, **kwargs):
            if "head_mask" not in kwargs or kwargs["head_mask"] is None:
                kwargs["head_mask"] = self._head_mask
            return original_forward(*args, **kwargs)

        self._original_forward = original_forward
        self._auto.forward = patched_forward

        self._st.eval()

    @property
    def similarity(self) -> str:  # type: ignore[override]
        return self._similarity

    @property
    def num_total_heads(self) -> int:
        return self._num_layers * self._num_heads

    def set_head_mask(self, head_mask: torch.Tensor) -> None:
        """Update head_mask in place for the next encode call. No model reload."""
        if head_mask.shape != (self._num_layers, self._num_heads):
            raise ValueError(f"head_mask shape mismatch {tuple(head_mask.shape)}")
        self._head_mask = head_mask.to(self._device)

    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        emb = self._st.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=(self._similarity == "cosine"),
            show_progress_bar=False,
        )
        return emb.astype(np.float32, copy=False)

    def encode_queries(self, queries: list[str], batch_size: int = 32) -> np.ndarray:
        return self._encode(queries, batch_size)

    def encode_documents(self, documents: list[str], batch_size: int = 32) -> np.ndarray:
        return self._encode(documents, batch_size)

    def to(self, device) -> PrunedTransformerEncoder:  # type: ignore[override]
        self._device = pick_device(device) if isinstance(device, str) else torch.device(device)
        self._st = self._st.to(str(self._device))
        self._head_mask = self._head_mask.to(self._device)
        return self
