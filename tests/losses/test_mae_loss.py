import pytest
import torch

from refrakt_core.losses.mae import MAELoss

def test_mae_loss_basic():
    loss_fn = MAELoss()
    B, N, D = 4, 16, 64
    
    predictions = {
        "recon_patches": torch.randn(B, N, D),
        "mask": torch.ones(B, N),
        "original_patches": torch.randn(B, N, D)
    }
    
    loss = loss_fn(predictions)
    assert loss.item() > 0

def test_mae_loss_partial_masking():
    loss_fn = MAELoss()
    B, N, D = 4, 16, 64
    
    mask = torch.zeros(B, N)
    mask[:, :N//2] = 1
    
    # Controlled values for predictable loss
    predictions = {
        "recon_patches": torch.ones(B, N, D),
        "mask": mask,
        "original_patches": torch.zeros(B, N, D)
    }
    
    loss = loss_fn(predictions)
    # Expected loss = (1^2 * D) = D
    assert torch.isclose(loss, torch.tensor(float(D)))

def test_mae_loss_normalization():
    loss_fn_normalized = MAELoss(normalize_target=True)
    loss_fn_standard = MAELoss(normalize_target=False)
    
    B, N, D = 4, 16, 64
    predictions = {
        "recon_patches": torch.randn(B, N, D),
        "mask": torch.ones(B, N),
        "original_patches": torch.randn(B, N, D)
    }
    
    loss_norm = loss_fn_normalized(predictions)
    loss_std = loss_fn_standard(predictions)
    
    assert not torch.isclose(loss_norm, loss_std)

def test_mae_loss_missing_keys():
    loss_fn = MAELoss()
    predictions = {
        "recon_patches": torch.randn(4, 16, 64),
        "original_patches": torch.randn(4, 16, 64)
    }
    
    with pytest.raises(KeyError):
        loss_fn(predictions)

def test_mae_registry_integration():
    from refrakt_core.registry.loss_registry import get_loss
    mae_loss = get_loss("mae", normalize_target=True)
    assert isinstance(mae_loss, MAELoss)
    assert mae_loss.normalize_target is True