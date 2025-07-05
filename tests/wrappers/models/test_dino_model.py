import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch
from omegaconf import DictConfig

from refrakt_core.wrappers.models.dino import DINOWrapper
from refrakt_core.schema.model_output import ModelOutput


class MockDINOModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.student_head = nn.Linear(512, 1024)
        # Create a mock backbone without get_attention_maps method
        self.backbone = Mock()
        # Explicitly remove get_attention_maps to ensure it doesn't exist
        if hasattr(self.backbone, 'get_attention_maps'):
            delattr(self.backbone, 'get_attention_maps')
    
    def forward(self, x, teacher=False):
        return torch.randn(x.shape[0], 1024)
    
    def update_teacher(self):
        return True


class MockDINOModelWithAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.student_head = nn.Linear(512, 1024)
        self.backbone = Mock()
        self.backbone.get_attention_maps = Mock(return_value=torch.randn(2, 4, 224, 224))
    
    def forward(self, x, teacher=False):
        return torch.randn(x.shape[0], 1024)
    
    def update_teacher(self):
        return True


@pytest.fixture
def mock_dino_model():
    return MockDINOModel()


@pytest.fixture
def mock_dino_model_with_attention():
    return MockDINOModelWithAttention()


@pytest.fixture
def sample_input():
    return torch.randn(2, 3, 224, 224)


@pytest.fixture
def model_config():
    return {
        "backbone": "resnet18",
        "out_dim": 1024
    }


@pytest.fixture
def dict_config():
    return DictConfig({
        "backbone": "resnet18",
        "out_dim": 1024
    })


# Smoke Tests
def test_dino_wrapper_smoke_initialization_with_model(mock_dino_model):
    """Test that DINOWrapper can be initialized with a model without errors."""
    wrapper = DINOWrapper(mock_dino_model)
    assert wrapper.dino_model == mock_dino_model


def test_dino_wrapper_smoke_initialization_with_dict(model_config):
    """Test that DINOWrapper can be initialized with a dict config without errors."""
    with patch('refrakt_core.models.dino.DINOModelWrapper') as mock_dino_class:
        mock_dino_class.return_value = Mock()
        wrapper = DINOWrapper(model_config)
        assert wrapper.dino_model is not None


def test_dino_wrapper_smoke_initialization_with_dictconfig(dict_config):
    """Test that DINOWrapper can be initialized with a DictConfig without errors."""
    with patch('refrakt_core.models.dino.DINOModelWrapper') as mock_dino_class:
        mock_dino_class.return_value = Mock()
        wrapper = DINOWrapper(dict_config)
        assert wrapper.dino_model is not None


def test_dino_wrapper_smoke_forward_pass(mock_dino_model, sample_input):
    """Test that DINOWrapper can perform a forward pass without errors."""
    wrapper = DINOWrapper(mock_dino_model)
    output = wrapper(sample_input)
    assert isinstance(output, ModelOutput)
    assert output.embeddings is not None


def test_dino_wrapper_smoke_forward_pass_with_teacher(mock_dino_model, sample_input):
    """Test that DINOWrapper can perform a forward pass with teacher=True without errors."""
    wrapper = DINOWrapper(mock_dino_model)
    output = wrapper(sample_input, teacher=True)
    assert isinstance(output, ModelOutput)
    assert output.embeddings is not None


def test_dino_wrapper_smoke_update_teacher(mock_dino_model):
    """Test that update_teacher method works without errors."""
    wrapper = DINOWrapper(mock_dino_model)
    result = wrapper.update_teacher()
    assert result is True


# Sanity Tests
def test_dino_wrapper_sanity_output_structure(mock_dino_model, sample_input):
    """Test that DINOWrapper returns correct ModelOutput structure."""
    wrapper = DINOWrapper(mock_dino_model)
    output = wrapper(sample_input)
    
    assert isinstance(output, ModelOutput)
    assert output.embeddings is not None
    assert output.loss_components == {}
    assert "wrapper_config" in output.extra
    assert output.extra["wrapper_config"]["wrapper_type"] == "dino"


def test_dino_wrapper_sanity_attention_maps_included(mock_dino_model_with_attention, sample_input):
    """Test that attention maps are included when available."""
    wrapper = DINOWrapper(mock_dino_model_with_attention)
    output = wrapper(sample_input)
    
    assert hasattr(output, 'attention_maps')
    assert output.attention_maps is not None
    assert output.attention_maps.shape == (2, 4, 224, 224)


def test_dino_wrapper_sanity_teacher_parameter_filtering(mock_dino_model, sample_input):
    """Test that teacher parameter is properly passed to the model."""
    wrapper = DINOWrapper(mock_dino_model)
    
    with patch.object(mock_dino_model, 'forward') as mock_forward:
        mock_forward.return_value = torch.randn(2, 1024)
        wrapper(sample_input, teacher=True)
        
        # Should call forward with teacher=True
        mock_forward.assert_called_with(sample_input, teacher=True)


def test_dino_wrapper_sanity_parameters_method(mock_dino_model):
    """Test that parameters method returns student_head parameters."""
    wrapper = DINOWrapper(mock_dino_model)
    params = list(wrapper.parameters())
    assert len(params) > 0


def test_dino_wrapper_sanity_named_parameters_method(mock_dino_model):
    """Test that named_parameters method returns student_head named parameters."""
    wrapper = DINOWrapper(mock_dino_model)
    named_params = list(wrapper.named_parameters())
    assert len(named_params) > 0


# Unit Tests
def test_dino_wrapper_unit_backbone_assignment():
    """Test that dino_model is properly assigned."""
    # Create a mock that inherits from nn.Module
    mock_model = Mock(spec=nn.Module)
    wrapper = DINOWrapper(mock_model)
    assert wrapper.dino_model == mock_model


def test_dino_wrapper_unit_inherits_from_nn_module():
    """Test that DINOWrapper inherits from nn.Module."""
    mock_model = Mock(spec=nn.Module)
    wrapper = DINOWrapper(mock_model)
    assert isinstance(wrapper, nn.Module)


def test_dino_wrapper_unit_dict_config_processing(model_config):
    """Test that dict config is properly processed."""
    with patch('refrakt_core.models.dino.DINOModelWrapper') as mock_dino_class:
        mock_dino_class.return_value = Mock()
        wrapper = DINOWrapper(model_config)
        assert wrapper.dino_model is not None


def test_dino_wrapper_unit_dictconfig_processing(dict_config):
    """Test that DictConfig is properly processed."""
    with patch('refrakt_core.models.dino.DINOModelWrapper') as mock_dino_class:
        mock_dino_class.return_value = Mock()
        wrapper = DINOWrapper(dict_config)
        assert wrapper.dino_model is not None


def test_dino_wrapper_unit_invalid_model_type():
    """Test that invalid model type raises TypeError."""
    with pytest.raises(TypeError):
        DINOWrapper("invalid_model_type")


def test_dino_wrapper_unit_wrapper_config_filtering():
    """Test that wrapper config filters out model initialization parameters."""
    mock_model = Mock(spec=nn.Module)
    wrapper = DINOWrapper(mock_model, backbone="resnet18", out_dim=1024, custom_param="value")
    assert wrapper.wrapper_config["custom_param"] == "value"
    assert "backbone" not in wrapper.wrapper_config
    assert "out_dim" not in wrapper.wrapper_config


def test_dino_wrapper_unit_forward_calls_dino_model(mock_dino_model, sample_input):
    """Test that forward method calls the DINO model."""
    wrapper = DINOWrapper(mock_dino_model)
    
    with patch.object(mock_dino_model, 'forward') as mock_forward:
        mock_forward.return_value = torch.randn(2, 1024)
        wrapper(sample_input)
        
        # Should call forward with valid args only
        mock_forward.assert_called_with(sample_input, teacher=False)


def test_dino_wrapper_unit_forward_filters_kwargs(mock_dino_model, sample_input):
    """Test that forward method filters out unexpected kwargs."""
    wrapper = DINOWrapper(mock_dino_model)
    
    with patch.object(mock_dino_model, 'forward') as mock_forward:
        mock_forward.return_value = torch.randn(2, 1024)
        wrapper(sample_input, teacher=True, unexpected_param="value")
        
        # Should only pass valid args
        mock_forward.assert_called_with(sample_input, teacher=True)


def test_dino_wrapper_unit_update_teacher_calls_model(mock_dino_model):
    """Test that update_teacher calls the model's update_teacher method."""
    wrapper = DINOWrapper(mock_dino_model)
    
    with patch.object(mock_dino_model, 'update_teacher') as mock_update:
        mock_update.return_value = True
        result = wrapper.update_teacher()
        
        mock_update.assert_called_once()
        assert result is True


def test_dino_wrapper_unit_update_teacher_raises_error():
    """Test that update_teacher raises AttributeError when model doesn't have the method."""
    mock_model = Mock(spec=nn.Module)
    # Remove update_teacher method
    if hasattr(mock_model, 'update_teacher'):
        delattr(mock_model, 'update_teacher')
    
    wrapper = DINOWrapper(mock_model)
    
    with pytest.raises(AttributeError):
        wrapper.update_teacher()


def test_dino_wrapper_unit_parameters_returns_student_head_params(mock_dino_model):
    """Test that parameters returns student_head parameters."""
    wrapper = DINOWrapper(mock_dino_model)
    params = list(wrapper.parameters())
    
    # Should return student_head parameters
    student_params = list(mock_dino_model.student_head.parameters())
    assert len(params) == len(student_params)


def test_dino_wrapper_unit_named_parameters_returns_student_head_params(mock_dino_model):
    """Test that named_parameters returns student_head named parameters."""
    wrapper = DINOWrapper(mock_dino_model)
    named_params = list(wrapper.named_parameters())
    
    # Should return student_head named parameters
    student_named_params = list(mock_dino_model.student_head.named_parameters())
    assert len(named_params) == len(student_named_params)


def test_dino_wrapper_unit_registration():
    """Test that DINOWrapper is properly registered."""
    from refrakt_core.registry.wrapper_registry import WRAPPER_REGISTRY
    assert "dino" in WRAPPER_REGISTRY


def test_dino_wrapper_unit_attention_maps_handling(mock_dino_model_with_attention, sample_input):
    """Test that attention maps are properly handled when available."""
    wrapper = DINOWrapper(mock_dino_model_with_attention)
    output = wrapper(sample_input)
    
    assert hasattr(output, 'attention_maps')
    assert output.attention_maps is not None


def test_dino_wrapper_unit_no_attention_maps_when_not_available(mock_dino_model, sample_input):
    """Test that attention maps are not included when not available."""
    wrapper = DINOWrapper(mock_dino_model)
    output = wrapper(sample_input)
    
    # Should not have attention_maps when backbone doesn't have get_attention_maps
    assert output.attention_maps is None


def test_dino_wrapper_unit_extra_fields_consistency(mock_dino_model, sample_input):
    """Test that extra fields are consistent across calls."""
    wrapper = DINOWrapper(mock_dino_model)
    output1 = wrapper(sample_input)
    output2 = wrapper(sample_input)
    
    assert "wrapper_config" in output1.extra
    assert "wrapper_config" in output2.extra
    assert output1.extra["wrapper_config"]["wrapper_type"] == output2.extra["wrapper_config"]["wrapper_type"] 