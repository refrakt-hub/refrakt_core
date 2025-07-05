import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from refrakt_core.wrappers.models.autoencoder import AutoencoderWrapper
from refrakt_core.schema.model_output import ModelOutput


class MockAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(10, 5)
        self.decoder = nn.Linear(5, 10)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return {"recon": decoded}


class MockVAEAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(10, 5)
        self.decoder = nn.Linear(5, 10)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return {
            "recon": decoded,
            "mu": torch.randn_like(encoded),
            "logvar": torch.randn_like(encoded)
        }


class MockMAEAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(10, 5)
        self.decoder = nn.Linear(5, 10)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return {
            "recon": decoded,
            "mask": torch.randint(0, 2, (x.shape[0], 5)),
            "original_patches": torch.randn_like(encoded)
        }


@pytest.fixture
def mock_model():
    return MockAutoencoder()


@pytest.fixture
def mock_vae_model():
    return MockVAEAutoencoder()


@pytest.fixture
def mock_mae_model():
    return MockMAEAutoencoder()


@pytest.fixture
def sample_input():
    return torch.randn(2, 10)


# Smoke Tests
def test_autoencoder_wrapper_smoke_initialization(mock_model):
    """Test that AutoencoderWrapper can be initialized without errors."""
    wrapper = AutoencoderWrapper(mock_model)
    assert wrapper.backbone == mock_model
    assert wrapper.variant == "simple"


def test_autoencoder_wrapper_smoke_forward_pass(mock_model, sample_input):
    """Test that AutoencoderWrapper can perform a forward pass without errors."""
    wrapper = AutoencoderWrapper(mock_model)
    output = wrapper(sample_input)
    assert isinstance(output, ModelOutput)
    assert output.reconstruction is not None


def test_autoencoder_wrapper_smoke_forward_for_graph(mock_model, sample_input):
    """Test that forward_for_graph method works without errors."""
    wrapper = AutoencoderWrapper(mock_model)
    output = wrapper.forward_for_graph(sample_input)
    assert isinstance(output, torch.Tensor)


# Sanity Tests
def test_autoencoder_wrapper_sanity_vae_variant(mock_vae_model, sample_input):
    """Test that VAE variant returns correct ModelOutput structure."""
    wrapper = AutoencoderWrapper(mock_vae_model, variant="vae")
    output = wrapper(sample_input)
    
    assert isinstance(output, ModelOutput)
    assert output.reconstruction is not None
    assert "mu" in output.extra
    assert "logvar" in output.extra
    assert isinstance(output.extra["mu"], torch.Tensor)
    assert isinstance(output.extra["logvar"], torch.Tensor)


def test_autoencoder_wrapper_sanity_mae_variant(mock_mae_model, sample_input):
    """Test that MAE variant returns correct ModelOutput structure."""
    wrapper = AutoencoderWrapper(mock_mae_model, variant="mae")
    output = wrapper(sample_input)
    
    assert isinstance(output, ModelOutput)
    assert output.reconstruction is not None
    assert "mask" in output.extra
    assert "original_patches" in output.extra
    assert isinstance(output.extra["mask"], torch.Tensor)
    assert isinstance(output.extra["original_patches"], torch.Tensor)


def test_autoencoder_wrapper_sanity_simple_variant(mock_model, sample_input):
    """Test that simple variant returns correct ModelOutput structure."""
    wrapper = AutoencoderWrapper(mock_model, variant="simple")
    output = wrapper(sample_input)
    
    assert isinstance(output, ModelOutput)
    assert output.reconstruction is not None
    assert output.extra == {}


def test_autoencoder_wrapper_sanity_forward_for_graph_returns_reconstruction(mock_model, sample_input):
    """Test that forward_for_graph returns the reconstruction tensor."""
    wrapper = AutoencoderWrapper(mock_model)
    output = wrapper.forward_for_graph(sample_input)
    
    # Should return the same as reconstruction from full forward
    full_output = wrapper(sample_input)
    assert torch.equal(output, full_output.reconstruction)


# Unit Tests
def test_autoencoder_wrapper_unit_default_variant():
    """Test that default variant is 'simple' when not specified."""
    mock_model = Mock()
    wrapper = AutoencoderWrapper(mock_model)
    assert wrapper.variant == "simple"


def test_autoencoder_wrapper_unit_custom_variant():
    """Test that custom variant is properly set."""
    mock_model = Mock()
    wrapper = AutoencoderWrapper(mock_model, variant="custom_variant")
    assert wrapper.variant == "custom_variant"


def test_autoencoder_wrapper_unit_backbone_assignment():
    """Test that backbone is properly assigned."""
    mock_model = Mock()
    wrapper = AutoencoderWrapper(mock_model)
    assert wrapper.backbone == mock_model


def test_autoencoder_wrapper_unit_inherits_from_nn_module():
    """Test that AutoencoderWrapper inherits from nn.Module."""
    mock_model = Mock()
    wrapper = AutoencoderWrapper(mock_model)
    assert isinstance(wrapper, nn.Module)


def test_autoencoder_wrapper_unit_simple_variant_output_structure(mock_model, sample_input):
    """Test that simple variant produces correct output structure."""
    wrapper = AutoencoderWrapper(mock_model, variant="simple")
    output = wrapper(sample_input)
    
    assert hasattr(output, 'reconstruction')
    assert hasattr(output, 'extra')
    assert output.extra == {}


def test_autoencoder_wrapper_unit_vae_variant_output_structure(mock_vae_model, sample_input):
    """Test that VAE variant produces correct output structure."""
    wrapper = AutoencoderWrapper(mock_vae_model, variant="vae")
    output = wrapper(sample_input)
    
    assert hasattr(output, 'reconstruction')
    assert hasattr(output, 'extra')
    assert 'mu' in output.extra
    assert 'logvar' in output.extra


def test_autoencoder_wrapper_unit_mae_variant_output_structure(mock_mae_model, sample_input):
    """Test that MAE variant produces correct output structure."""
    wrapper = AutoencoderWrapper(mock_mae_model, variant="mae")
    output = wrapper(sample_input)
    
    assert hasattr(output, 'reconstruction')
    assert hasattr(output, 'extra')
    assert 'mask' in output.extra
    assert 'original_patches' in output.extra


def test_autoencoder_wrapper_unit_forward_calls_backbone(mock_model, sample_input):
    """Test that forward method calls the backbone model."""
    mock_model.forward = Mock(return_value={"recon": torch.randn(2, 10)})
    wrapper = AutoencoderWrapper(mock_model)
    wrapper(sample_input)
    
    mock_model.forward.assert_called_once_with(sample_input)


def test_autoencoder_wrapper_unit_forward_for_graph_calls_forward(mock_model, sample_input):
    """Test that forward_for_graph calls forward and returns reconstruction."""
    mock_model.forward = Mock(return_value={"recon": torch.randn(2, 10)})
    wrapper = AutoencoderWrapper(mock_model)
    
    with patch.object(wrapper, 'forward') as mock_forward:
        mock_forward.return_value = ModelOutput(
            reconstruction=torch.randn(2, 3, 64, 64),
            embeddings=torch.randn(2, 512)
        )
        result = wrapper.forward_for_graph(sample_input)
        
        mock_forward.assert_called_once_with(sample_input)
        assert mock_forward.return_value.reconstruction is not None
        assert torch.equal(result, mock_forward.return_value.reconstruction)


def test_autoencoder_wrapper_unit_registration():
    """Test that AutoencoderWrapper is properly registered."""
    from refrakt_core.registry.wrapper_registry import WRAPPER_REGISTRY
    assert "autoencoder" in WRAPPER_REGISTRY 