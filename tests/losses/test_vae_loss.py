import pytest
import torch

from refrakt_core.losses.vae import VAELoss


def test_vae_loss_basic():
    loss_fn = VAELoss()
    B, C, H, W = 4, 3, 32, 32

    model_output = {
        "recon": torch.randn(B, C, H, W),
        "mu": torch.randn(B, 16),
        "logvar": torch.randn(B, 16),
    }
    target = torch.randn(B, C, H, W)

    loss = loss_fn(model_output, target)
    assert loss.item() > 0


def test_vae_ae_mode():
    loss_fn = VAELoss()
    B, C, H, W = 4, 3, 32, 32

    model_output = torch.randn(B, C, H, W)
    target = torch.randn(B, C, H, W)

    loss = loss_fn(model_output, target)
    assert loss.item() > 0


def test_vae_loss_recon_types():
    loss_fn_mse = VAELoss(recon_loss_type="mse")
    loss_fn_l1 = VAELoss(recon_loss_type="l1")

    B, C, H, W = 4, 3, 32, 32
    model_output = {
        "recon": torch.randn(B, C, H, W),
        "mu": torch.randn(B, 16),
        "logvar": torch.randn(B, 16),
    }
    target = torch.randn(B, C, H, W)

    loss_mse = loss_fn_mse(model_output, target)
    loss_l1 = loss_fn_l1(model_output, target)

    assert not torch.isclose(loss_mse, loss_l1)


def test_vae_loss_kld_weight():
    loss_fn_low_kld = VAELoss(kld_weight=0.1)
    loss_fn_high_kld = VAELoss(kld_weight=1.0)

    B, C, H, W = 4, 3, 32, 32

    # Use non-zero KL divergence components
    model_output = {
        "recon": torch.zeros(B, C, H, W),
        "mu": torch.ones(B, 16),
        "logvar": torch.ones(B, 16),
    }
    target = torch.zeros(B, C, H, W)

    loss_low = loss_fn_low_kld(model_output, target)
    loss_high = loss_fn_high_kld(model_output, target)

    assert loss_high > loss_low


def test_vae_registry_integration():
    from refrakt_core.registry.loss_registry import get_loss

    vae_loss = get_loss("vae", recon_loss_type="l1", kld_weight=0.5)
    assert isinstance(vae_loss, VAELoss)
    assert vae_loss.recon_loss_type == "l1"
    assert vae_loss.kld_weight == 0.5
