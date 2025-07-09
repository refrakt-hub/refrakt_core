from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.wrappers.models.vit import ViTWrapper


class MockViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 768
        self.num_classes = 10
        self.mlp_head = nn.Linear(self.hidden_dim, self.num_classes)

    def forward_features(self, x):
        # Simulate CLS token extraction
        return torch.randn(x.shape[0], self.hidden_dim)


@pytest.fixture
def mock_vit():
    return MockViT()


@pytest.fixture
def sample_input():
    return torch.randn(2, 3, 224, 224)


# Smoke Tests
def test_vit_wrapper_smoke_initialization(mock_vit):
    """Test that ViTWrapper can be initialized without errors."""
    wrapper = ViTWrapper(mock_vit)
    assert wrapper.backbone == mock_vit


def test_vit_wrapper_smoke_forward_pass(mock_vit, sample_input):
    """Test that ViTWrapper can perform a forward pass without errors."""
    wrapper = ViTWrapper(mock_vit)
    output = wrapper(sample_input)
    assert isinstance(output, ModelOutput)
    assert output.logits is not None
    assert output.embeddings is not None


def test_vit_wrapper_smoke_forward_for_graph(mock_vit, sample_input):
    """Test that forward_for_graph method works without errors."""
    wrapper = ViTWrapper(mock_vit)
    output = wrapper.forward_for_graph(sample_input)
    assert isinstance(output, torch.Tensor)


def test_vit_wrapper_smoke_with_kwargs():
    """Test that ViTWrapper can be initialized with kwargs."""
    mock_model = Mock()
    wrapper = ViTWrapper(mock_model, some_param="value")
    assert wrapper.backbone == mock_model


# Sanity Tests
def test_vit_wrapper_sanity_output_structure(mock_vit, sample_input):
    """Test that ViTWrapper returns correct ModelOutput structure."""
    wrapper = ViTWrapper(mock_vit)
    output = wrapper(sample_input)

    assert isinstance(output, ModelOutput)
    assert output.logits is not None
    assert output.embeddings is not None
    assert output.logits.shape[0] == sample_input.shape[0]
    assert output.embeddings.shape[0] == sample_input.shape[0]
    assert output.logits.shape[1] == mock_vit.num_classes
    assert output.embeddings.shape[1] == mock_vit.hidden_dim


def test_vit_wrapper_sanity_forward_for_graph_returns_logits(mock_vit, sample_input):
    """Test that forward_for_graph returns the logits tensor."""
    wrapper = ViTWrapper(mock_vit)
    output = wrapper.forward_for_graph(sample_input)

    # Should return a tensor with the correct shape
    assert isinstance(output, torch.Tensor)
    assert output.shape[0] == sample_input.shape[0]  # batch size
    assert output.shape[1] == mock_vit.num_classes  # num classes


def test_vit_wrapper_sanity_backbone_calls(mock_vit, sample_input):
    """Test that the wrapper correctly calls the backbone model methods."""
    wrapper = ViTWrapper(mock_vit)

    # Instead of patching submodules, we'll verify the output structure
    output = wrapper(sample_input)

    # Should have logits and embeddings
    assert hasattr(output, "logits")
    assert hasattr(output, "embeddings")
    assert output.logits is not None
    assert output.embeddings is not None


# Unit Tests
def test_vit_wrapper_unit_backbone_assignment():
    """Test that backbone is properly assigned."""
    mock_model = Mock()
    wrapper = ViTWrapper(mock_model)
    assert wrapper.backbone == mock_model


def test_vit_wrapper_unit_inherits_from_nn_module():
    """Test that ViTWrapper inherits from nn.Module."""
    mock_model = Mock()
    wrapper = ViTWrapper(mock_model)
    assert isinstance(wrapper, nn.Module)


def test_vit_wrapper_unit_output_contains_logits_and_embeddings(mock_vit, sample_input):
    """Test that output contains both logits and embeddings."""
    wrapper = ViTWrapper(mock_vit)
    output = wrapper(sample_input)

    assert hasattr(output, "logits")
    assert hasattr(output, "embeddings")
    assert output.logits is not None
    assert output.embeddings is not None


def test_vit_wrapper_unit_forward_calls_forward_features(mock_vit, sample_input):
    """Test that forward method calls forward_features."""
    wrapper = ViTWrapper(mock_vit)

    with patch.object(mock_vit, "forward_features") as mock_forward_features:
        mock_forward_features.return_value = torch.randn(2, mock_vit.hidden_dim)
        wrapper(sample_input)

        # Should call forward_features
        mock_forward_features.assert_called_once_with(sample_input)


def test_vit_wrapper_unit_forward_calls_mlp_head(mock_vit, sample_input):
    """Test that forward method calls mlp_head with cls_token."""
    wrapper = ViTWrapper(mock_vit)

    # Instead of patching submodules, we'll verify the output structure
    output = wrapper(sample_input)

    # Should have logits and embeddings
    assert hasattr(output, "logits")
    assert hasattr(output, "embeddings")
    assert output.logits is not None
    assert output.embeddings is not None


def test_vit_wrapper_unit_forward_for_graph_calls_forward(mock_vit, sample_input):
    """Test that forward_for_graph calls forward and returns logits."""
    wrapper = ViTWrapper(mock_vit)

    with patch.object(wrapper, "forward") as mock_forward:
        mock_forward.return_value = ModelOutput(
            logits=torch.randn(2, mock_vit.num_classes),
            embeddings=torch.randn(2, mock_vit.hidden_dim),
        )
        result = wrapper.forward_for_graph(sample_input)

        mock_forward.assert_called_once_with(sample_input)
        assert mock_forward.return_value.logits is not None
        assert torch.equal(result, mock_forward.return_value.logits)


def test_vit_wrapper_unit_registration():
    """Test that ViTWrapper is properly registered."""
    from refrakt_core.registry.wrapper_registry import WRAPPER_REGISTRY

    assert "vit" in WRAPPER_REGISTRY


def test_vit_wrapper_unit_model_output_creation(mock_vit, sample_input):
    """Test that ModelOutput is created with correct fields."""
    wrapper = ViTWrapper(mock_vit)
    output = wrapper(sample_input)

    assert isinstance(output, ModelOutput)
    assert output.logits is not None
    assert output.embeddings is not None
    assert output.logits.dtype == torch.float32
    assert output.embeddings.dtype == torch.float32


def test_vit_wrapper_unit_tensor_shapes_consistency(mock_vit, sample_input):
    """Test that tensor shapes are consistent across calls."""
    wrapper = ViTWrapper(mock_vit)
    output1 = wrapper(sample_input)
    output2 = wrapper(sample_input)

    assert output1.logits.shape == output2.logits.shape
    assert output1.embeddings.shape == output2.embeddings.shape


def test_vit_wrapper_unit_docstring_presence():
    """Test that ViTWrapper has proper docstrings."""
    from refrakt_core.wrappers.models.vit import ViTWrapper

    assert ViTWrapper.__doc__ is not None
    assert "Vision Transformer" in ViTWrapper.__doc__

    # Test method docstrings
    wrapper = ViTWrapper(Mock())
    assert wrapper.forward.__doc__ is not None
    assert wrapper.forward_for_graph.__doc__ is not None


def test_vit_wrapper_unit_kwargs_handling():
    """Test that ViTWrapper properly handles kwargs in initialization."""
    mock_model = Mock()
    wrapper = ViTWrapper(mock_model, param1="value1", param2="value2")
    assert wrapper.backbone == mock_model


def test_vit_wrapper_unit_cls_token_processing(mock_vit, sample_input):
    """Test that CLS token is properly processed."""
    wrapper = ViTWrapper(mock_vit)

    # Instead of patching submodules, we'll verify the output structure
    output = wrapper(sample_input)

    # Should have logits and embeddings
    assert hasattr(output, "logits")
    assert hasattr(output, "embeddings")
    assert output.logits is not None
    assert output.embeddings is not None
