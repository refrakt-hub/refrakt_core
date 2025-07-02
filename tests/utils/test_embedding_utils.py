import torch

from refrakt_core.utils.classes.embedding import Embedding, RelativeEmbedding


def test_Embedding():
    embed = Embedding(patch_size=4, C=96)
    x = torch.randn(2, 3, 32, 32)
    out = embed(x)
    assert out.shape == (2, 64, 96)  # (32/4)**2 = 64 patches


def test_RelativeEmbedding():
    rel_embed = RelativeEmbedding(window_size=7)
    attn = torch.randn(1, 4, 49, 49)  # (window_size**2, window_size**2)
    out = rel_embed(attn)
    assert out.shape == attn.shape
