"""Every registered encoder must satisfy the base interface, even stubs."""
from __future__ import annotations

import pytest

from vitruvius.encoders import Encoder, list_encoders
from vitruvius.encoders.conv_encoder import ConvEncoder
from vitruvius.encoders.lstm_encoder import LSTMEncoder
from vitruvius.encoders.mamba_encoder import MambaEncoder

STUBS = [MambaEncoder, LSTMEncoder, ConvEncoder]


def test_registry_lists_all_known_encoders():
    names = set(list_encoders())
    assert {"minilm-l6-v2", "bert-base-nli", "gte-small", "mamba", "lstm", "conv"} <= names


@pytest.mark.parametrize("cls", STUBS)
def test_stub_encoders_satisfy_interface(cls):
    enc = cls()
    assert isinstance(enc, Encoder)
    assert isinstance(enc.name, str) and enc.name
    assert enc.embedding_dim == 0  # stubs declare 0 to discourage use
    with pytest.raises(NotImplementedError):
        enc.encode_queries(["q"])
    with pytest.raises(NotImplementedError):
        enc.encode_documents(["d"])


def test_get_encoder_raises_helpful_error():
    from vitruvius.encoders import get_encoder

    with pytest.raises(ValueError) as exc:
        get_encoder("nope")
    assert "Supported" in str(exc.value)
