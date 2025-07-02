import torch
from torch import nn

from refrakt_core.utils.classes.attention import MHA
from refrakt_core.utils.classes.encoder import Encoder, EncoderBlock
from refrakt_core.utils.classes.utils import FeedForward


def test_EncoderBlock():
    self_att = MHA(64, 4, 0.1)
    ff = FeedForward(64, 256, dropout=0.1)
    block = EncoderBlock(self_att, ff, 0.1)

    x = torch.randn(2, 10, 64)
    out = block(x, None)
    assert out.shape == x.shape


def test_Encoder():
    blocks = []
    for _ in range(3):
        self_att = MHA(64, 4, 0.1)
        ff = FeedForward(64, 256, dropout=0.1)
        blocks.append(EncoderBlock(self_att, ff, 0.1))

    encoder = Encoder(nn.ModuleList(blocks))
    x = torch.randn(2, 10, 64)
    out = encoder(x, None)
    assert out.shape == x.shape
