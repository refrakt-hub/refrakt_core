# test_convnext.py
import pytest
import torch
import torch.nn as nn

from refrakt_core.models.convnext import ConvNeXt, ConvNeXtBlock

@pytest.fixture
def small_convnext():
    return ConvNeXt(in_channels=3, num_classes=10)

def test_convnext_init(small_convnext):
    assert small_convnext.model_name == "convnext"
    assert small_convnext.num_classes == 10

def test_convnext_block():
    block = ConvNeXtBlock(96, 192)
    x = torch.randn(1, 96, 56, 56)
    output = block(x)
    assert output.shape == (1, 192, 56, 56)

def test_convnext_forward(small_convnext):
    x = torch.randn(2, 3, 224, 224)
    output = small_convnext(x)
    assert output.shape == (2, 10)

def test_convnext_different_input_sizes(small_convnext):
    sizes = [(128, 128), (256, 256), (64, 64)]
    for h, w in sizes:
        x = torch.randn(2, 3, h, w)
        output = small_convnext(x)
        assert output.shape == (2, 10)