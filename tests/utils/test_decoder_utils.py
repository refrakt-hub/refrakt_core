import torch
from torch import nn

from refrakt_core.utils.classes.attention import MHA
from refrakt_core.utils.classes.decoder import Decoder, DecoderBlock
from refrakt_core.utils.classes.utils import FeedForward


def test_DecoderBlock():
    masked_att = MHA(64, 4, 0.1)
    cross_att = MHA(64, 4, 0.1)
    ff = FeedForward(64, 256, dropout=0.1)
    block = DecoderBlock(masked_att, cross_att, ff, 0.1)
    
    x = torch.randn(2, 10, 64)
    enc_out = torch.randn(2, 5, 64)
    out = block(x, enc_out, None, None)
    assert out.shape == x.shape

def test_Decoder():
    blocks = []
    for _ in range(3):
        masked_att = MHA(64, 4, 0.1)
        cross_att = MHA(64, 4, 0.1)
        ff = FeedForward(64, 256, dropout=0.1)
        blocks.append(DecoderBlock(masked_att, cross_att, ff, 0.1))
    
    decoder = Decoder(nn.ModuleList(blocks))
    x = torch.randn(2, 10, 64)
    enc_out = torch.randn(2, 5, 64)
    out = decoder(x, enc_out, None, None)
    assert out.shape == x.shape