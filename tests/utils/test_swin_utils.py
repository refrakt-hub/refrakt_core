import torch

from refrakt_core.utils.classes.swin import AlternateSwin, SwinBlock


def test_SwinBlock():
    B, H, W, C = 2, 7, 7, 96  # window_size = 7 => 49 tokens
    x = torch.randn(B, H * W, C)  # (B, 49, 96)

    block = SwinBlock(embed_dim=C, num_heads=4, window_size=7, mask=False)
    out = block(x)

    assert out.shape == x.shape

def test_AlternateSwin():
    B, H, W, C = 2, 7, 7, 96
    x = torch.randn(B, H * W, C)  # (B, 49, 96)

    model = AlternateSwin(embed_dim=C, num_heads=4, window_size=7)
    out = model(x)

    assert out.shape == x.shape
