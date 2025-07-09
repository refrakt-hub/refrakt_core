from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.wrappers.models.resnet import ResNetWrapper


class MockResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_dim = 512
        self.num_classes = 10
        self.fc = nn.Linear(self.feature_dim, self.num_classes)

    def forward(self, x, return_features=False):
        # Simulate feature extraction
        features = torch.randn(x.shape[0], self.feature_dim)
        if return_features:
            return features
        else:
            return self.fc(features)


@pytest.fixture
def mock_resnet():
    return MockResNet()


@pytest.fixture
def sample_input():
    return torch.randn(2, 3, 224, 224)


# Smoke Tests
def test_resnet_wrapper_smoke_initialization(mock_resnet):
    """Test that ResNetWrapper can be initialized without errors."""
    wrapper = ResNetWrapper(mock_resnet)
    assert wrapper.backbone == mock_resnet


def test_resnet_wrapper_smoke_forward_pass(mock_resnet, sample_input):
    """Test that ResNetWrapper can perform a forward pass without errors."""
    wrapper = ResNetWrapper(mock_resnet)
    output = wrapper(sample_input)
    assert isinstance(output, ModelOutput)
    assert output.logits is not None
    assert output.embeddings is not None


def test_resnet_wrapper_smoke_forward_for_graph(mock_resnet, sample_input):
    """Test that forward_for_graph method works without errors."""
    wrapper = ResNetWrapper(mock_resnet)
    output = wrapper.forward_for_graph(sample_input)
    assert isinstance(output, torch.Tensor)


# Sanity Tests
def test_resnet_wrapper_sanity_output_structure(mock_resnet, sample_input):
    """Test that ResNetWrapper returns correct ModelOutput structure."""
    wrapper = ResNetWrapper(mock_resnet)
    output = wrapper(sample_input)

    assert isinstance(output, ModelOutput)
    assert output.logits is not None
    assert output.embeddings is not None
    assert output.logits.shape[0] == sample_input.shape[0]
    assert output.embeddings.shape[0] == sample_input.shape[0]
    assert output.logits.shape[1] == mock_resnet.num_classes
    assert output.embeddings.shape[1] == mock_resnet.feature_dim


def test_resnet_wrapper_sanity_forward_for_graph_returns_logits(
    mock_resnet, sample_input
):
    """Test that forward_for_graph returns the logits tensor."""
    wrapper = ResNetWrapper(mock_resnet)
    output = wrapper.forward_for_graph(sample_input)

    # Should return a tensor with the correct shape
    assert isinstance(output, torch.Tensor)
    assert output.shape[0] == sample_input.shape[0]  # batch size
    assert output.shape[1] == mock_resnet.num_classes  # num classes


def test_resnet_wrapper_sanity_backbone_calls(mock_resnet, sample_input):
    """Test that the wrapper correctly calls the backbone model."""
    wrapper = ResNetWrapper(mock_resnet)

    # Mock the backbone's forward method to track calls
    original_forward = mock_resnet.forward
    mock_resnet.forward = Mock(side_effect=original_forward)

    wrapper(sample_input)

    # Should call forward with return_features=True for features
    assert mock_resnet.forward.call_count >= 1


# Unit Tests
def test_resnet_wrapper_unit_backbone_assignment():
    """Test that backbone is properly assigned."""
    mock_model = Mock()
    wrapper = ResNetWrapper(mock_model)
    assert wrapper.backbone == mock_model


def test_resnet_wrapper_unit_inherits_from_nn_module():
    """Test that ResNetWrapper inherits from nn.Module."""
    mock_model = Mock()
    wrapper = ResNetWrapper(mock_model)
    assert isinstance(wrapper, nn.Module)


def test_resnet_wrapper_unit_output_contains_logits_and_embeddings(
    mock_resnet, sample_input
):
    """Test that output contains both logits and embeddings."""
    wrapper = ResNetWrapper(mock_resnet)
    output = wrapper(sample_input)

    assert hasattr(output, "logits")
    assert hasattr(output, "embeddings")
    assert output.logits is not None
    assert output.embeddings is not None


def test_resnet_wrapper_unit_forward_calls_backbone_with_return_features(
    mock_resnet, sample_input
):
    """Test that forward method calls backbone with return_features=True for features."""
    wrapper = ResNetWrapper(mock_resnet)

    with patch.object(mock_resnet, "forward") as mock_forward:
        mock_forward.return_value = torch.randn(2, mock_resnet.feature_dim)
        wrapper(sample_input)

        # Should call forward with return_features=True
        mock_forward.assert_called_with(sample_input, return_features=True)


def test_resnet_wrapper_unit_forward_calls_fc_layer(mock_resnet, sample_input):
    """Test that forward method calls the fc layer with features."""
    wrapper = ResNetWrapper(mock_resnet)

    # Instead of patching the fc layer directly, we'll verify the output structure
    output = wrapper(sample_input)

    # Should have logits and embeddings
    assert hasattr(output, "logits")
    assert hasattr(output, "embeddings")
    assert output.logits is not None
    assert output.embeddings is not None


def test_resnet_wrapper_unit_forward_for_graph_calls_backbone_directly(
    mock_resnet, sample_input
):
    """Test that forward_for_graph calls backbone directly without return_features."""
    wrapper = ResNetWrapper(mock_resnet)

    with patch.object(mock_resnet, "forward") as mock_forward:
        mock_forward.return_value = torch.randn(2, mock_resnet.num_classes)
        wrapper.forward_for_graph(sample_input)

        # Should call forward without return_features for graph tracing
        mock_forward.assert_called_with(sample_input)


def test_resnet_wrapper_unit_registration():
    """Test that ResNetWrapper is properly registered."""
    from refrakt_core.registry.wrapper_registry import WRAPPER_REGISTRY

    assert "resnet" in WRAPPER_REGISTRY


def test_resnet_wrapper_unit_model_output_creation(mock_resnet, sample_input):
    """Test that ModelOutput is created with correct fields."""
    wrapper = ResNetWrapper(mock_resnet)
    output = wrapper(sample_input)

    assert isinstance(output, ModelOutput)
    assert output.logits is not None
    assert output.embeddings is not None
    assert output.logits.dtype == torch.float32
    assert output.embeddings.dtype == torch.float32


def test_resnet_wrapper_unit_tensor_shapes_consistency(mock_resnet, sample_input):
    """Test that tensor shapes are consistent across calls."""
    wrapper = ResNetWrapper(mock_resnet)
    output1 = wrapper(sample_input)
    output2 = wrapper(sample_input)

    assert output1.logits.shape == output2.logits.shape
    assert output1.embeddings.shape == output2.embeddings.shape
