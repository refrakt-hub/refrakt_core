import pytest
import torch

from refrakt_core.losses.cross_entropy import CrossEntropyLoss


def test_cross_entropy_basic():
    loss_fn = CrossEntropyLoss()
    pred = torch.randn(4, 10)
    target = torch.tensor([1, 2, 3, 4])
    loss = loss_fn(pred, target)
    assert loss.item() > 0


def test_cross_entropy_invalid_shape_pred():
    loss_fn = CrossEntropyLoss()
    pred = torch.randn(4)  # invalid shape
    target = torch.tensor([1, 2, 3, 4])
    with pytest.raises(ValueError):
        loss_fn(pred, target)


def test_cross_entropy_invalid_shape_target():
    loss_fn = CrossEntropyLoss()
    pred = torch.randn(4, 10)
    target = torch.randn(4, 1)
    with pytest.raises(ValueError):
        loss_fn(pred, target)


def test_cross_entropy_mismatched_batch():
    loss_fn = CrossEntropyLoss()
    pred = torch.randn(4, 10)
    target = torch.tensor([1, 2])
    with pytest.raises(ValueError):
        loss_fn(pred, target)
