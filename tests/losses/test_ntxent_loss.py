import pytest
import torch
import torch.nn.functional as F

from refrakt_core.losses.ntxent import NTXentLoss


@pytest.fixture
def loss_fn():
    return NTXentLoss(temperature=0.5)


def test_identical_views_low_loss(loss_fn):
    torch.manual_seed(42)
    z = F.normalize(torch.randn(8, 128), dim=1)
    loss = loss_fn(z, z)
    assert (
        loss.item() < 1.2
    ), f"Expected low loss for identical views, got {loss.item()}"


def test_symmetric_property(loss_fn):
    torch.manual_seed(0)
    z1 = F.normalize(torch.randn(16, 128), dim=1)
    z2 = F.normalize(torch.randn(16, 128), dim=1)
    loss1 = loss_fn(z1, z2)
    loss2 = loss_fn(z2, z1)
    assert torch.allclose(
        loss1, loss2, atol=1e-5
    ), f"Loss not symmetric: {loss1.item()} vs {loss2.item()}"


def test_random_projections_high_loss(loss_fn):
    torch.manual_seed(99)
    z1 = F.normalize(torch.randn(64, 128), dim=1)
    z2 = F.normalize(torch.randn(64, 128), dim=1)
    loss = loss_fn(z1, z2)
    assert (
        loss.item() > 2.0
    ), f"Expected high loss for random vectors, got {loss.item()}"


def test_batch_size_one_handling(loss_fn):
    z1 = F.normalize(torch.randn(1, 128), dim=1)
    z2 = F.normalize(torch.randn(1, 128), dim=1)
    with pytest.raises(ValueError, match="Batch size must be > 1"):
        loss_fn(z1, z2)
