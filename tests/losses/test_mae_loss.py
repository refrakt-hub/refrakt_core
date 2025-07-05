import pytest
import torch

from refrakt_core.losses.mae import MAELoss


def test_mae_loss_basic():
    loss_fn = MAELoss()
    B, C, H, W = 4, 3, 32, 32
    N = (H // 16) * (W // 16)  # Number of patches
    D = 16 * 16 * C  # Patch dimension

    predictions = {
        "recon": torch.randn(B, C, H, W),  # Image tensor
        "mask": torch.ones(B, N),
        "original_patches": torch.randn(B, N, D),
    }

    loss = loss_fn(predictions)
    assert loss.item() > 0


def test_mae_loss_partial_masking():
    loss_fn = MAELoss()
    B, C, H, W = 4, 3, 32, 32
    N = (H // 16) * (W // 16)  # Number of patches
    D = 16 * 16 * C  # Patch dimension

    mask = torch.zeros(B, N)
    mask[:, : N // 2] = 1

    # Controlled values for predictable loss
    predictions = {
        "recon": torch.ones(B, C, H, W),  # Image tensor
        "mask": mask,
        "original_patches": torch.zeros(B, N, D),
    }

    loss = loss_fn(predictions)
    # Expected loss = (1^2 * D) = D
    assert torch.isclose(loss, torch.tensor(float(D)))


def test_mae_loss_normalization():
    loss_fn_normalized = MAELoss(normalize_target=True)
    loss_fn_standard = MAELoss(normalize_target=False)

    B, C, H, W = 4, 3, 32, 32
    N = (H // 16) * (W // 16)  # Number of patches
    D = 16 * 16 * C  # Patch dimension
    predictions = {
        "recon": torch.randn(B, C, H, W),  # Image tensor
        "mask": torch.ones(B, N),
        "original_patches": torch.randn(B, N, D),
    }

    loss_norm = loss_fn_normalized(predictions)
    loss_std = loss_fn_standard(predictions)

    assert not torch.isclose(loss_norm, loss_std)


def test_mae_loss_missing_keys():
    loss_fn = MAELoss()
    predictions = {
        "recon": torch.randn(4, 3, 32, 32),
        "original_patches": torch.randn(4, 4, 768),  # 4 patches, 768 dims
    }

    with pytest.raises(KeyError):
        loss_fn(predictions)


def test_mae_registry_integration():
    from refrakt_core.registry.loss_registry import get_loss

    mae_loss = get_loss("mae", normalize_target=True)
    assert isinstance(mae_loss, MAELoss)
    assert mae_loss.normalize_target is True
