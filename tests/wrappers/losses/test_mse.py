import pytest
import torch

from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.wrappers.losses.mse import MSELossWrapper


@pytest.fixture
def recon():
    # Shape: (B, C, H, W) - 2 batch size, 3 channels, 64x64 image
    return torch.randn(2, 3, 64, 64)


@pytest.fixture
def target():
    # Shape: (B, C, H, W) - 2 batch size, 3 channels, 64x64 image
    return torch.randn(2, 3, 64, 64)


@pytest.fixture
def model_output(recon):
    return ModelOutput(reconstruction=recon)


@pytest.fixture
def model_output_with_target(recon, target):
    return ModelOutput(reconstruction=recon, targets=target)


# Smoke Tests
def test_mse_loss_wrapper_smoke_initialization():
    """Test that MSELossWrapper can be initialized without errors."""
    wrapper = MSELossWrapper()
    assert hasattr(wrapper, "loss_fn")
    assert isinstance(wrapper.loss_fn, torch.nn.MSELoss)


def test_mse_loss_wrapper_smoke_forward_with_reconstruction(model_output):
    """Test that MSELossWrapper forward pass works with reconstruction only."""
    wrapper = MSELossWrapper()
    loss = wrapper(model_output)
    assert hasattr(loss, "total")
    assert hasattr(loss, "components")
    assert isinstance(loss.total, torch.Tensor)
    assert isinstance(loss.components, dict)


def test_mse_loss_wrapper_smoke_forward_with_target(model_output, target):
    """Test that MSELossWrapper forward pass works with explicit target."""
    wrapper = MSELossWrapper()
    loss = wrapper(model_output, target)
    assert hasattr(loss, "total")
    assert hasattr(loss, "components")
    assert isinstance(loss.total, torch.Tensor)
    assert isinstance(loss.components, dict)


# Sanity Tests
def test_mse_loss_wrapper_sanity_loss_value(model_output):
    """Test that MSE loss produces reasonable loss values."""
    wrapper = MSELossWrapper()
    loss = wrapper(model_output)
    assert loss.total is not None
    assert "mse" in loss.components
    assert loss.total.item() >= 0  # Loss should be non-negative
    assert loss.components["mse"].item() >= 0


def test_mse_loss_wrapper_sanity_loss_with_target(model_output, target):
    """Test that MSE loss with explicit target produces reasonable values."""
    wrapper = MSELossWrapper()
    loss = wrapper(model_output, target)
    assert loss.total is not None
    assert "mse" in loss.components
    assert loss.total.item() >= 0
    assert loss.components["mse"].item() >= 0


def test_mse_loss_wrapper_sanity_loss_consistency(model_output, target):
    """Test that MSE loss is consistent between reconstruction and target modes."""
    wrapper = MSELossWrapper()
    
    # Loss with reconstruction as target (self-reconstruction)
    loss_self = wrapper(model_output)
    
    # Loss with explicit target
    loss_target = wrapper(model_output, target)
    
    # Both should produce valid loss values
    assert loss_self.total.item() >= 0
    assert loss_target.total.item() >= 0


# Unit Tests
def test_mse_loss_wrapper_unit_missing_reconstruction():
    """Test that MSELossWrapper raises ValueError when reconstruction is missing."""
    wrapper = MSELossWrapper()
    bad_output = ModelOutput(reconstruction=None)
    with pytest.raises(ValueError, match="Missing reconstruction field in ModelOutput"):
        wrapper(bad_output)


def test_mse_loss_wrapper_unit_custom_loss_params():
    """Test that MSELossWrapper accepts custom loss parameters."""
    wrapper = MSELossWrapper(reduction='sum')
    assert wrapper.loss_fn.reduction == 'sum'


def test_mse_loss_wrapper_unit_empty_loss_params():
    """Test that MSELossWrapper handles empty loss parameters."""
    wrapper = MSELossWrapper()
    assert wrapper.loss_fn.reduction == 'mean'  # Default reduction


def test_mse_loss_wrapper_unit_loss_components_structure(model_output):
    """Test that loss output has correct structure."""
    wrapper = MSELossWrapper()
    loss = wrapper(model_output)
    
    # Check that components contain the expected MSE loss
    assert "mse" in loss.components
    assert torch.equal(loss.total, loss.components["mse"])


def test_mse_loss_wrapper_unit_different_shapes():
    """Test that MSELossWrapper handles different tensor shapes."""
    wrapper = MSELossWrapper()
    
    # Test with 1D tensors
    recon_1d = torch.randn(10)
    output_1d = ModelOutput(reconstruction=recon_1d)
    loss_1d = wrapper(output_1d)
    assert loss_1d.total.item() >= 0
    
    # Test with 2D tensors
    recon_2d = torch.randn(5, 10)
    output_2d = ModelOutput(reconstruction=recon_2d)
    loss_2d = wrapper(output_2d)
    assert loss_2d.total.item() >= 0
    
    # Test with 3D tensors
    recon_3d = torch.randn(3, 4, 5)
    output_3d = ModelOutput(reconstruction=recon_3d)
    loss_3d = wrapper(output_3d)
    assert loss_3d.total.item() >= 0


def test_mse_loss_wrapper_unit_target_vs_self_target(model_output, target):
    """Test that using explicit target vs self-target produces different results."""
    wrapper = MSELossWrapper()
    
    # Loss with reconstruction as target (self-reconstruction)
    loss_self = wrapper(model_output)
    
    # Loss with explicit target
    loss_target = wrapper(model_output, target)
    
    # These should be different since target != reconstruction
    assert not torch.equal(loss_self.total, loss_target.total)


def test_mse_loss_wrapper_unit_registration():
    """Test that MSELossWrapper is properly registered."""
    from refrakt_core.registry.loss_registry import get_loss
    
    # Should be able to get the loss by name
    loss_instance = get_loss("mse_wrapped")
    assert isinstance(loss_instance, MSELossWrapper)


def test_mse_loss_wrapper_unit_gradient_flow(model_output):
    """Test that gradients can flow through the loss wrapper."""
    wrapper = MSELossWrapper()
    
    # Ensure reconstruction requires gradients
    model_output.reconstruction.requires_grad_(True)
    
    loss = wrapper(model_output)
    loss.total.backward()
    
    # Check that gradients were computed
    assert model_output.reconstruction.grad is not None
    assert model_output.reconstruction.grad.shape == model_output.reconstruction.shape 