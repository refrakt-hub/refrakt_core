import pytest
import torch

from refrakt_core.losses.gan import GANLoss


@pytest.mark.parametrize("use_lsgan", [True, False])
def test_gan_loss_basic(use_lsgan):
    loss_fn = GANLoss(use_lsgan=use_lsgan)
    pred = torch.randn(4, 1)
    loss = loss_fn(pred, target_is_real=True)
    assert loss.item() >= 0


def test_gan_loss_fake():
    loss_fn = GANLoss()
    pred = torch.randn(4, 1)
    loss = loss_fn(pred, target_is_real=False)
    assert loss.item() >= 0


def test_gan_loss_invalid_target_flag():
    loss_fn = GANLoss()
    pred = torch.randn(4, 1)
    with pytest.raises(TypeError):
        loss_fn(pred, target_is_real="yes")


def test_gan_loss_non_tensor_input():
    loss_fn = GANLoss()
    pred = [[0.1, 0.9]]  # not a tensor
    with pytest.raises(TypeError):
        loss_fn(pred, target_is_real=True)
