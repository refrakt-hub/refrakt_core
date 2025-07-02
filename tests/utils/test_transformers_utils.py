import torch

from refrakt_core.utils.classes.transformers import (InputEmbeddings,
                                                     PositionalEncoding)


def test_InputEmbeddings():
    embed = InputEmbeddings(d_model=64, vocab_size=100)
    x = torch.randint(0, 100, (2, 10))
    out = embed(x)
    assert out.shape == (2, 10, 64)


def test_PositionalEncoding():
    pe = PositionalEncoding(d_model=64, seq_len=20, dropout=0.1)
    x = torch.randn(2, 10, 64)
    out = pe(x)
    assert out.shape == x.shape
