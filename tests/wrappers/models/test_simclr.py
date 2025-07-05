import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from refrakt_core.wrappers.models.simclr import SimCLRWrapper
from refrakt_core.schema.model_output import ModelOutput

class MockSimCLRModel(nn.Module):
    def forward(self, x):
        return torch.randn(x.shape[0], 128)

@pytest.fixture
def mock_simclr_model():
    return MockSimCLRModel()

@pytest.fixture
def sample_input():
    return torch.randn(2, 3, 224, 224)

# Smoke Tests
def test_simclr_wrapper_smoke_initialization(mock_simclr_model):
    wrapper = SimCLRWrapper(mock_simclr_model)
    assert wrapper.model == mock_simclr_model

def test_simclr_wrapper_smoke_forward_pass(mock_simclr_model, sample_input):
    wrapper = SimCLRWrapper(mock_simclr_model)
    output = wrapper(sample_input)
    assert isinstance(output, ModelOutput)
    assert output.embeddings is not None
    assert output.extra["wrapper_type"] == "simclr"

# Sanity Tests
def test_simclr_wrapper_sanity_output_shape(mock_simclr_model, sample_input):
    wrapper = SimCLRWrapper(mock_simclr_model)
    output = wrapper(sample_input)
    assert output.embeddings.shape[0] == sample_input.shape[0]

# Unit Tests
def test_simclr_wrapper_unit_forward_calls_model(mock_simclr_model, sample_input):
    wrapper = SimCLRWrapper(mock_simclr_model)
    with patch.object(mock_simclr_model, 'forward', wraps=mock_simclr_model.forward) as mock_forward:
        wrapper(sample_input)
        mock_forward.assert_called_once_with(sample_input)

def test_simclr_wrapper_unit_output_structure(mock_simclr_model, sample_input):
    wrapper = SimCLRWrapper(mock_simclr_model)
    output = wrapper(sample_input)
    assert hasattr(output, 'embeddings')
    assert hasattr(output, 'extra')
    assert 'wrapper_type' in output.extra 