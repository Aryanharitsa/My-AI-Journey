"""Every registered encoder must satisfy the base interface, even stubs."""
from __future__ import annotations

import pytest

from vitruvius.encoders import Encoder, list_encoders

STUBS: list = []  # all encoders are now real; see test_encoder_real_smoke for their checks


def test_registry_lists_all_known_encoders():
    names = set(list_encoders())
    assert {"minilm-l6-v2", "bert-base", "gte-small", "mamba-retriever-fs", "lstm-retriever", "conv-retriever"} <= names


@pytest.mark.parametrize("cls", STUBS)
def test_stub_encoders_satisfy_interface(cls):
    enc = cls()
    assert isinstance(enc, Encoder)
    assert isinstance(enc.name, str) and enc.name
    assert enc.embedding_dim == 0  # stubs declare 0 to discourage use
    assert enc.similarity in {"cosine", "dot"}
    with pytest.raises(NotImplementedError):
        enc.encode_queries(["q"])
    with pytest.raises(NotImplementedError):
        enc.encode_documents(["d"])


def test_get_encoder_raises_helpful_error():
    from vitruvius.encoders import get_encoder

    with pytest.raises(ValueError) as exc:
        get_encoder("nope")
    assert "Supported" in str(exc.value)
