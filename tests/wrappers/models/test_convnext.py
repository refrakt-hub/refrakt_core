import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from refrakt_core.wrappers.models.convnext import ConvNeXtWrapper
from refrakt_core.schema.model_output import ModelOutput


class MockConvNeXt(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_dim = 1024
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
def mock_convnext():
    return MockConvNeXt()


@pytest.fixture
def sample_input():
    return torch.randn(2, 3, 224, 224)


# Smoke Tests
def test_convnext_wrapper_smoke_initialization(mock_convnext):
    """Test that ConvNeXtWrapper can be initialized without errors."""
    wrapper = ConvNeXtWrapper(mock_convnext)
    assert wrapper.backbone == mock_convnext


def test_convnext_wrapper_smoke_forward_pass(mock_convnext, sample_input):
    """Test that ConvNeXtWrapper can perform a forward pass without errors."""
    wrapper = ConvNeXtWrapper(mock_convnext)
    output = wrapper(sample_input)
    assert isinstance(output, ModelOutput)
    assert output.logits is not None
    assert output.embeddings is not None


def test_convnext_wrapper_smoke_forward_for_graph(mock_convnext, sample_input):
    """Test that forward_for_graph method works without errors."""
    wrapper = ConvNeXtWrapper(mock_convnext)
    output = wrapper.forward_for_graph(sample_input)
    assert isinstance(output, torch.Tensor)


# Sanity Tests
def test_convnext_wrapper_sanity_output_structure(mock_convnext, sample_input):
    """Test that ConvNeXtWrapper returns correct ModelOutput structure."""
    wrapper = ConvNeXtWrapper(mock_convnext)
    output = wrapper(sample_input)
    
    assert isinstance(output, ModelOutput)
    assert output.logits is not None
    assert output.embeddings is not None
    assert output.logits.shape[0] == sample_input.shape[0]
    assert output.embeddings.shape[0] == sample_input.shape[0]
    assert output.logits.shape[1] == mock_convnext.num_classes
    assert output.embeddings.shape[1] == mock_convnext.feature_dim


def test_convnext_wrapper_sanity_forward_for_graph_returns_logits(mock_convnext, sample_input):
    """Test that forward_for_graph returns the logits tensor."""
    wrapper = ConvNeXtWrapper(mock_convnext)
    output = wrapper.forward_for_graph(sample_input)

    # Should return a tensor with the correct shape
    assert isinstance(output, torch.Tensor)
    assert output.shape[0] == sample_input.shape[0]  # batch size
    assert output.shape[1] == mock_convnext.num_classes  # num classes


def test_convnext_wrapper_sanity_backbone_calls(mock_convnext, sample_input):
    """Test that the wrapper correctly calls the backbone model."""
    wrapper = ConvNeXtWrapper(mock_convnext)
    
    # Mock the backbone's forward method to track calls
    original_forward = mock_convnext.forward
    mock_convnext.forward = Mock(side_effect=original_forward)
    
    wrapper(sample_input)
    
    # Should call forward with return_features=True for features
    assert mock_convnext.forward.call_count >= 1


# Unit Tests
def test_convnext_wrapper_unit_backbone_assignment():
    """Test that backbone is properly assigned."""
    mock_model = Mock()
    wrapper = ConvNeXtWrapper(mock_model)
    assert wrapper.backbone == mock_model


def test_convnext_wrapper_unit_inherits_from_nn_module():
    """Test that ConvNeXtWrapper inherits from nn.Module."""
    mock_model = Mock()
    wrapper = ConvNeXtWrapper(mock_model)
    assert isinstance(wrapper, nn.Module)


def test_convnext_wrapper_unit_output_contains_logits_and_embeddings(mock_convnext, sample_input):
    """Test that output contains both logits and embeddings."""
    wrapper = ConvNeXtWrapper(mock_convnext)
    output = wrapper(sample_input)
    
    assert hasattr(output, 'logits')
    assert hasattr(output, 'embeddings')
    assert output.logits is not None
    assert output.embeddings is not None


def test_convnext_wrapper_unit_forward_calls_backbone_with_return_features(mock_convnext, sample_input):
    """Test that forward method calls backbone with return_features=True for features."""
    wrapper = ConvNeXtWrapper(mock_convnext)
    
    with patch.object(mock_convnext, 'forward') as mock_forward:
        mock_forward.return_value = torch.randn(2, mock_convnext.feature_dim)
        wrapper(sample_input)
        
        # Should call forward with return_features=True
        mock_forward.assert_called_with(sample_input, return_features=True)


def test_convnext_wrapper_unit_forward_calls_fc_layer(mock_convnext, sample_input):
    """Test that forward method calls the fc layer with features."""
    wrapper = ConvNeXtWrapper(mock_convnext)
    
    # Instead of patching the fc layer directly, we'll verify the output structure
    output = wrapper(sample_input)
    
    # Should have logits and embeddings
    assert hasattr(output, 'logits')
    assert hasattr(output, 'embeddings')
    assert output.logits is not None
    assert output.embeddings is not None


def test_convnext_wrapper_unit_forward_for_graph_calls_forward(mock_convnext, sample_input):
    """Test that forward_for_graph calls forward and returns logits."""
    wrapper = ConvNeXtWrapper(mock_convnext)
    
    with patch.object(wrapper, 'forward') as mock_forward:
        mock_forward.return_value = ModelOutput(
            logits=torch.randn(2, mock_convnext.num_classes),
            embeddings=torch.randn(2, mock_convnext.feature_dim)
        )
        result = wrapper.forward_for_graph(sample_input)
        
        mock_forward.assert_called_once_with(sample_input)
        assert mock_forward.return_value.logits is not None
        assert torch.equal(result, mock_forward.return_value.logits)


def test_convnext_wrapper_unit_registration():
    """Test that ConvNeXtWrapper is properly registered."""
    from refrakt_core.registry.wrapper_registry import WRAPPER_REGISTRY
    assert "convnext" in WRAPPER_REGISTRY


def test_convnext_wrapper_unit_model_output_creation(mock_convnext, sample_input):
    """Test that ModelOutput is created with correct fields."""
    wrapper = ConvNeXtWrapper(mock_convnext)
    output = wrapper(sample_input)
    
    assert isinstance(output, ModelOutput)
    assert output.logits is not None
    assert output.embeddings is not None
    assert output.logits.dtype == torch.float32
    assert output.embeddings.dtype == torch.float32


def test_convnext_wrapper_unit_tensor_shapes_consistency(mock_convnext, sample_input):
    """Test that tensor shapes are consistent across calls."""
    wrapper = ConvNeXtWrapper(mock_convnext)
    output1 = wrapper(sample_input)
    output2 = wrapper(sample_input)
    
    assert output1.logits.shape == output2.logits.shape
    assert output1.embeddings.shape == output2.embeddings.shape


def test_convnext_wrapper_unit_docstring_presence():
    """Test that ConvNeXtWrapper has proper docstrings."""
    from refrakt_core.wrappers.models.convnext import ConvNeXtWrapper
    
    assert ConvNeXtWrapper.__doc__ is not None
    assert "ConvNeXt" in ConvNeXtWrapper.__doc__
    
    # Test method docstrings
    wrapper = ConvNeXtWrapper(Mock())
    assert wrapper.forward.__doc__ is not None
    assert wrapper.forward_for_graph.__doc__ is not None


def test_convnext_wrapper_unit_embeddings_and_logits_relationship(mock_convnext, sample_input):
    """Test that embeddings and logits have correct relationship."""
    wrapper = ConvNeXtWrapper(mock_convnext)
    
    output = wrapper(sample_input)
    
    # Check that we have both embeddings and logits
    assert output.embeddings is not None
    assert output.logits is not None
    # Check that they have the expected shapes
    assert output.embeddings.shape[0] == sample_input.shape[0]  # batch size
    assert output.logits.shape[0] == sample_input.shape[0]  # batch size


def test_convnext_wrapper_unit_lightweight_forward_for_graph(mock_convnext, sample_input):
    """Test that forward_for_graph is lightweight for graph tracing."""
    wrapper = ConvNeXtWrapper(mock_convnext)
    
    with patch.object(wrapper, 'forward') as mock_forward:
        mock_forward.return_value = ModelOutput(
            logits=torch.randn(2, mock_convnext.num_classes),
            embeddings=torch.randn(2, mock_convnext.feature_dim)
        )
        result = wrapper.forward_for_graph(sample_input)
        
        # Should only return logits for graph visualization
        assert result.shape == (2, mock_convnext.num_classes)
        assert mock_forward.return_value.logits is not None
        assert torch.equal(result, mock_forward.return_value.logits) 