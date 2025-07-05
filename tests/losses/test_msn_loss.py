import pytest
import torch
import torch.nn.functional as F

from refrakt_core.losses.msn import MSNLoss


def test_msn_loss_basic():
    loss_fn = MSNLoss()
    B, M, D, K = 4, 3, 64, 100

    # Normalize inputs to prevent NaNs
    z_anchor = F.normalize(torch.randn(B * M, D), dim=-1)
    z_target = F.normalize(torch.randn(B, D), dim=-1)
    prototypes = F.normalize(torch.randn(K, D), dim=-1)

    loss = loss_fn(z_anchor, z_target, prototypes)
    assert not torch.isnan(loss)
    # MSN loss can be negative due to entropy term, so just check it's finite
    assert torch.isfinite(loss)


import pytest
import torch
import torch.nn.functional as F

from refrakt_core.losses.msn import MSNLoss


def test_msn_loss_shape_mismatch():
    loss_fn = MSNLoss()
    B, M, D, K = 4, 3, 64, 100

    z_anchor = F.normalize(torch.randn(B * M, D), dim=-1)
    z_target = F.normalize(torch.randn(B, D), dim=-1)
    prototypes = F.normalize(torch.randn(K, D), dim=-1)

    # Anchor dim mismatch
    with pytest.raises(ValueError, match="Feature dimensions.*match"):
        loss_fn(F.normalize(torch.randn(B * M, 32), dim=-1), z_target, prototypes)

    # Target dim mismatch
    with pytest.raises(ValueError, match="Feature dimensions.*match"):
        loss_fn(z_anchor, F.normalize(torch.randn(B, 32), dim=-1), prototypes)

    # Prototypes dim mismatch
    with pytest.raises(ValueError, match="Feature dimensions.*match"):
        loss_fn(z_anchor, z_target, F.normalize(torch.randn(K, 32), dim=-1))

    # Invalid anchor batch size (not divisible by B)
    with pytest.raises(ValueError, match="Anchor batch size must be a multiple"):
        loss_fn(F.normalize(torch.randn(B * M + 1, D), dim=-1), z_target, prototypes)


def test_msn_loss_components():
    loss_fn = MSNLoss(lambda_me_max=0.5)
    B, M, D, K = 4, 3, 64, 100

    # Create aligned representations
    base = F.normalize(torch.randn(B, D), dim=-1)
    z_target = base
    z_anchor = base.repeat_interleave(M, dim=0)
    prototypes = F.normalize(torch.randn(K, D), dim=-1)

    aligned_loss = loss_fn(z_anchor, z_target, prototypes)

    # Random representations
    z_anchor_rand = F.normalize(torch.randn(B * M, D), dim=-1)
    z_target_rand = F.normalize(torch.randn(B, D), dim=-1)
    rand_loss = loss_fn(z_anchor_rand, z_target_rand, prototypes)

    assert rand_loss > aligned_loss


def test_msn_registry_integration():
    from refrakt_core.registry.loss_registry import get_loss

    msn_loss = get_loss("msn", temp_anchor=0.2, temp_target=0.05)
    assert isinstance(msn_loss, MSNLoss)
    assert msn_loss.temp_anchor == 0.2
