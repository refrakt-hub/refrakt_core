# test_swin.py
import pytest
import torch

from refrakt_core.models.swin import SwinTransformer


@pytest.fixture
def small_swin():
    return SwinTransformer(in_channels=3, num_classes=10)


def test_swin_init(small_swin):
    assert isinstance(small_swin.embedding, torch.nn.Module)
    assert isinstance(small_swin.head, torch.nn.Linear)


def test_swin_forward(small_swin):
    # Use 224x224 as this is the standard size for Swin Transformers
    x = torch.randn(2, 3, 224, 224)
    output = small_swin(x)
    assert output.shape == (2, 10)
