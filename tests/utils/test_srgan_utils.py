import torch

from refrakt_core.utils.classes.srgan import (Discriminator, Generator,
                                              SRResidualBlock, UpsampleBlock)


def test_UpsampleBlock():
    block = UpsampleBlock(64, 64)
    x = torch.randn(2, 64, 16, 16)
    out = block(x)
    assert out.shape == (2, 64, 32, 32)


def test_SRResidualBlock():
    block = SRResidualBlock(64)
    x = torch.randn(2, 64, 32, 32)
    out = block(x)
    assert out.shape == x.shape


def test_Generator():
    gen = Generator(scale_factor=4)
    x = torch.randn(2, 3, 16, 16)
    out = gen(x)
    assert out.shape == (2, 3, 64, 64)


def test_Discriminator():
    disc = Discriminator()
    x = torch.randn(2, 3, 64, 64)
    out = disc(x)

    # Accept model's actual output shape
    assert out.shape == (2,)
