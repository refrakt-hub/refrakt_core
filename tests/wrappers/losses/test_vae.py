import pytest
import torch
from refrakt_core.wrappers.losses.vae import VAELossWrapper
from refrakt_core.schema.model_output import ModelOutput

@pytest.fixture
def recon():
    return torch.randn(2, 10)

@pytest.fixture
def mu():
    return torch.randn(2, 10)

@pytest.fixture
def logvar():
    return torch.randn(2, 10)

@pytest.fixture
def target():
    return torch.randn(2, 10)

@pytest.fixture
def model_output(recon, mu, logvar):
    return ModelOutput(reconstruction=recon, extra={"mu": mu, "logvar": logvar})

# Smoke Tests
def test_vae_loss_wrapper_smoke_initialization():
    wrapper = VAELossWrapper()
    assert hasattr(wrapper, 'loss_fn')

def test_vae_loss_wrapper_smoke_forward(model_output, target):
    wrapper = VAELossWrapper()
    loss = wrapper(model_output, target)
    assert hasattr(loss, 'total')
    assert hasattr(loss, 'components')

# Sanity Tests
def test_vae_loss_wrapper_sanity_loss_value(model_output, target):
    wrapper = VAELossWrapper()
    loss = wrapper(model_output, target)
    assert loss.total is not None
    assert 'recon_loss' in loss.components
    assert 'kld_loss' in loss.components

# Unit Tests
def test_vae_loss_wrapper_unit_missing_fields(target):
    wrapper = VAELossWrapper()
    bad_output = ModelOutput(reconstruction=None, extra={})
    with pytest.raises(ValueError):
        wrapper(bad_output, target)

def test_vae_loss_wrapper_unit_kld_weight_and_type():
    wrapper = VAELossWrapper()
    assert hasattr(wrapper, 'kld_weight')
    assert hasattr(wrapper, 'recon_loss_type') 