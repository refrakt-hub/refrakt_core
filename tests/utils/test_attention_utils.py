import pytest
import torch

from refrakt_core.utils.classes.attention import MHA, MSA, ShiftedWindowMSA


def test_MSA():
    msa = MSA(d=64, n_heads=4)
    x = torch.randn(2, 5, 64)
    out = msa(x)
    assert out.shape == x.shape


def test_MHA():
    mha = MHA(d_model=64, n_heads=4, dropout=0.1)
    q = torch.randn(2, 10, 64)
    k = v = torch.randn(2, 10, 64)
    out = mha(q, k, v, mask=None)
    assert out.shape == q.shape
    assert mha.att_scores is not None


def test_ShiftedWindowMSA():
    sw_msa = ShiftedWindowMSA(embed_dim=96, n_heads=4, window_size=7, mask=True)
    x = torch.randn(2, 49, 96)  # (7x7 = 49 patches)
    out = sw_msa(x)
    assert out.shape == x.shape
