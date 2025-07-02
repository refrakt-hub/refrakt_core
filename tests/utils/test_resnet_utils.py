import torch
from torch import nn

from refrakt_core.utils.classes.resnet import (BottleneckBlock, ResidualBlock,
                                               ViTResidual)


def test_ResidualBlock():
    block = ResidualBlock(64, 64)
    x = torch.randn(2, 64, 32, 32)
    out = block(x)
    assert out.shape == x.shape


def test_BottleneckBlock():
    downsample = nn.Sequential(
        nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(256)
    )
    block = BottleneckBlock(64, 64, downsample=downsample)  # 64 in, 256 out
    x = torch.randn(2, 64, 32, 32)
    out = block(x)
    assert out.shape == (2, 256, 32, 32)


def test_ViTResidual():
    vit_res = ViTResidual(hidden_d=64, n_heads=4)
    x = torch.randn(2, 10, 64)
    out = vit_res(x)
    assert out.shape == x.shape
