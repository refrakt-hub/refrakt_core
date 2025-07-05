import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from refrakt_core.wrappers.models.mae import MAEWrapper
from refrakt_core.schema.model_output import ModelOutput

class MockMAEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_size = (3, 224, 224)
    def forward(self, x):
        recon = torch.randn(x.shape[0], 16, 16*3)
        patches = torch.randn_like(recon)
        return {"recon": recon, "mask": torch.randint(0, 2, (x.shape[0], 16)), "original_patches": patches}

@pytest.fixture
def mock_mae_model():
    return MockMAEModel()

@pytest.fixture
def sample_input():
    return torch.randn(2, 3, 224, 224)

# Smoke Tests
def test_mae_wrapper_smoke_initialization(mock_mae_model):
    wrapper = MAEWrapper(mock_mae_model)
    assert wrapper.model == mock_mae_model

def test_mae_wrapper_smoke_forward_pass(mock_mae_model, sample_input):
    wrapper = MAEWrapper(mock_mae_model)
    output = wrapper(sample_input)
    assert isinstance(output, ModelOutput)
    assert output.reconstruction is not None
    assert "mask" in output.extra
    assert "original_patches" in output.extra

# Sanity Tests
def test_mae_wrapper_sanity_unpatchify_shape(mock_mae_model, sample_input):
    wrapper = MAEWrapper(mock_mae_model)
    model_output = mock_mae_model(sample_input)
    recon = model_output["recon"]
    patches = model_output["original_patches"]
    result = wrapper._unpatchify(recon, patches)
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] == recon.shape[0]

# Unit Tests
def test_mae_wrapper_unit_img_size_assignment(mock_mae_model):
    wrapper = MAEWrapper(mock_mae_model)
    assert wrapper.expected_input_dim == mock_mae_model.img_size

def test_mae_wrapper_unit_forward_calls_model(mock_mae_model, sample_input):
    wrapper = MAEWrapper(mock_mae_model)
    with patch.object(mock_mae_model, 'forward', wraps=mock_mae_model.forward) as mock_forward:
        wrapper(sample_input)
        mock_forward.assert_called_once_with(sample_input)

def test_mae_wrapper_unit_output_structure(mock_mae_model, sample_input):
    wrapper = MAEWrapper(mock_mae_model)
    output = wrapper(sample_input)
    assert hasattr(output, 'reconstruction')
    assert hasattr(output, 'extra')
    assert 'mask' in output.extra
    assert 'original_patches' in output.extra 