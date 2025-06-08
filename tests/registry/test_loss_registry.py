import torch
from torch import nn
import pytest
from unittest.mock import patch, MagicMock
from refrakt_core.registry.loss_registry import register_loss, get_loss, LOSS_REGISTRY

    
def test_custom_loss_registration():
    """Test custom loss registration and retrieval"""
    import torch
    from refrakt_core.registry.loss_registry import register_loss, get_loss
    from refrakt_core.losses.templates.base import BaseLoss

    @register_loss("custom_mse")
    class CustomMSELoss(BaseLoss):
        def forward(self, x, y):
            return torch.mean((x - y) ** 2)

    x = torch.randn(2, 3)
    y = torch.randn(2, 3)

    loss_fn = get_loss("custom_mse")
    result = loss_fn(x, y)

    assert isinstance(result, torch.Tensor)
    assert result.shape == ()




def test_loss_not_found():
    """Test handling of unregistered loss"""
    with pytest.raises(ValueError) as excinfo:
        get_loss("unknown_loss")
    assert "Loss 'unknown_loss' not found" in str(excinfo.value)