import pytest
import torch
import torch.nn as nn
from refrakt_core.wrappers.schema.default_loss import DefaultLossWrapper
from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.schema.loss_output import LossOutput
from refrakt_core.wrappers.losses.mae import MAELossWrapper
from refrakt_core.wrappers.losses.vae import VAELossWrapper

class DummyLoss(nn.Module):
    def forward(self, x, y):
        return torch.abs(x - y).sum()

@pytest.fixture
def dummy_loss():
    return DummyLoss()

@pytest.fixture
def sample_output():
    return torch.randn(2, 10)

@pytest.fixture
def sample_target():
    return torch.randn(2, 10)

# Smoke Tests
def test_default_loss_wrapper_smoke_initialization(dummy_loss):
    wrapper = DefaultLossWrapper(dummy_loss)
    assert hasattr(wrapper, 'loss_fn')

def test_default_loss_wrapper_smoke_forward_tensor(dummy_loss, sample_output, sample_target):
    wrapper = DefaultLossWrapper(dummy_loss)
    loss = wrapper(sample_output, sample_target)
    assert hasattr(loss, 'total')
    assert hasattr(loss, 'components')

# Sanity Tests
def test_default_loss_wrapper_sanity_model_output(dummy_loss, sample_output, sample_target):
    wrapper = DefaultLossWrapper(dummy_loss)
    model_output = ModelOutput(logits=sample_output)
    loss = wrapper(model_output, sample_target)
    assert loss.total is not None

# Unit Tests
def test_default_loss_wrapper_unit_mae_dispatch():
    mae_loss = MAELossWrapper()
    wrapper = DefaultLossWrapper(mae_loss)
    # Use a 4D tensor for reconstruction: (B, C, H, W)
    recon = torch.randn(2, 3, 64, 64)
    # For mask and original_patches, use shapes compatible with patchify (e.g., 16 patches, 768 features)
    mask = torch.ones(2, 16)
    original_patches = torch.randn(2, 16, 768)
    model_output = ModelOutput(reconstruction=recon, extra={"mask": mask, "original_patches": original_patches})
    loss = wrapper(model_output)
    assert hasattr(loss, 'total')

def test_default_loss_wrapper_unit_vae_dispatch(sample_output):
    vae_loss = VAELossWrapper()
    wrapper = DefaultLossWrapper(vae_loss)
    model_output = ModelOutput(reconstruction=sample_output, extra={"mu": sample_output, "logvar": sample_output})
    loss = wrapper(model_output, sample_output)
    assert hasattr(loss, 'total')

def test_default_loss_wrapper_unit_loss_output_return(dummy_loss, sample_output, sample_target):
    class LossReturnsLossOutput(nn.Module):
        def forward(self, x, y):
            return LossOutput(total=torch.tensor(1.0), components={"foo": torch.tensor(1.0)})
    wrapper = DefaultLossWrapper(LossReturnsLossOutput())
    loss = wrapper(sample_output, sample_target)
    assert isinstance(loss, LossOutput)

def test_default_loss_wrapper_unit_type_error(dummy_loss, sample_output, sample_target):
    class BadLoss(nn.Module):
        def forward(self, x, y):
            return "bad"
    wrapper = DefaultLossWrapper(BadLoss())
    with pytest.raises(TypeError):
        wrapper(sample_output, sample_target) 