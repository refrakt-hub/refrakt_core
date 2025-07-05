import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from refrakt_core.wrappers.models.srgan import SRGANWrapper
from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.schema.loss_output import LossOutput

class MockSRGANModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = nn.Linear(10, 10)
        self.discriminator = nn.Linear(10, 1)
        self.scale_factor = 4
    def generate(self, x):
        return torch.randn(x.shape[0], 10)
    def discriminate(self, x):
        return torch.randn(x.shape[0], 1)

@pytest.fixture
def mock_srgan_model():
    return MockSRGANModel()

@pytest.fixture
def sample_input():
    return torch.randn(2, 10)

@pytest.fixture
def sample_batch():
    return {"lr": torch.randn(2, 10), "hr": torch.randn(2, 10)}

@pytest.fixture
def mock_loss_fn():
    return {"discriminator": Mock(return_value=LossOutput(total=torch.tensor(1.0), components={"d": torch.tensor(1.0)})),
            "generator": Mock(return_value=LossOutput(total=torch.tensor(0.5), components={"g": torch.tensor(0.5)}))}

@pytest.fixture
def mock_optimizer():
    return {"generator": Mock(), "discriminator": Mock()}

# Smoke Tests
def test_srgan_wrapper_smoke_initialization(mock_srgan_model):
    wrapper = SRGANWrapper(mock_srgan_model)
    assert wrapper.model == mock_srgan_model

def test_srgan_wrapper_smoke_generate(mock_srgan_model, sample_input):
    wrapper = SRGANWrapper(mock_srgan_model)
    output = wrapper.generate(sample_input)
    assert isinstance(output, ModelOutput)
    assert output.image is not None
    assert output.extra["wrapper_type"] == "srgan"
    assert output.extra["scale_factor"] == mock_srgan_model.scale_factor

def test_srgan_wrapper_smoke_discriminate(mock_srgan_model, sample_input):
    wrapper = SRGANWrapper(mock_srgan_model)
    output = wrapper.discriminate(sample_input)
    assert isinstance(output, ModelOutput)
    assert output.logits is not None
    assert output.extra["wrapper_type"] == "srgan"

# Sanity Tests
def test_srgan_wrapper_sanity_forward_and_properties(mock_srgan_model, sample_input):
    wrapper = SRGANWrapper(mock_srgan_model)
    output = wrapper(sample_input)
    assert isinstance(output, ModelOutput)
    assert output.image is not None
    assert wrapper.generator == mock_srgan_model.generator
    assert wrapper.discriminator == mock_srgan_model.discriminator

def test_srgan_wrapper_sanity_training_step_discriminator(mock_srgan_model, sample_batch, mock_loss_fn, mock_optimizer):
    wrapper = SRGANWrapper(mock_srgan_model)
    result = wrapper.training_step(sample_batch, optimizer=mock_optimizer, loss_fn=mock_loss_fn, phase="discriminator")
    assert "d_loss" in result
    assert isinstance(result["d_loss"], LossOutput)

def test_srgan_wrapper_sanity_training_step_generator(mock_srgan_model, sample_batch, mock_loss_fn, mock_optimizer):
    wrapper = SRGANWrapper(mock_srgan_model)
    result = wrapper.training_step(sample_batch, optimizer=mock_optimizer, loss_fn=mock_loss_fn, phase="generator")
    assert "g_loss" in result
    assert isinstance(result["g_loss"], LossOutput)

# Unit Tests
def test_srgan_wrapper_unit_forward_calls_generate(mock_srgan_model, sample_input):
    wrapper = SRGANWrapper(mock_srgan_model)
    with patch.object(wrapper, 'generate', wraps=wrapper.generate) as mock_generate:
        wrapper(sample_input)
        mock_generate.assert_called_once_with(sample_input)

def test_srgan_wrapper_unit_forward_for_graph_dict_input(mock_srgan_model):
    wrapper = SRGANWrapper(mock_srgan_model)
    input_dict = {"lr": torch.randn(1, 10)}
    result = wrapper.forward_for_graph(input_dict)
    assert isinstance(result, torch.Tensor)

def test_srgan_wrapper_unit_forward_for_graph_tensor_input(mock_srgan_model, sample_input):
    wrapper = SRGANWrapper(mock_srgan_model)
    result = wrapper.forward_for_graph(sample_input)
    assert isinstance(result, torch.Tensor)

def test_srgan_wrapper_unit_forward_for_graph_none_input(mock_srgan_model):
    wrapper = SRGANWrapper(mock_srgan_model)
    # Should not raise
    result = wrapper.forward_for_graph(None)
    assert isinstance(result, torch.Tensor)

def test_srgan_wrapper_unit_training_step_invalid_phase(mock_srgan_model, sample_batch, mock_loss_fn, mock_optimizer):
    wrapper = SRGANWrapper(mock_srgan_model)
    with pytest.raises(ValueError):
        wrapper.training_step(sample_batch, optimizer=mock_optimizer, loss_fn=mock_loss_fn, phase="invalid") 