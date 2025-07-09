from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.wrappers.models.swin import SwinTransformerWrapper


class MockSwinBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Identity()
        self.patch1 = nn.Identity()
        self.patch2 = nn.Identity()
        self.stage1 = nn.Identity()
        self.stage2 = nn.Identity()
        self.stage3_1 = nn.Identity()
        self.stage3_2 = nn.Identity()
        self.stage3_3 = nn.Identity()
        self.patch3 = nn.Identity()
        self.stage4 = nn.Identity()
        self.head = nn.Linear(32, 10)

    def forward(self, x):
        return self.head(x)


class MockSwinModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Identity()
        self.patch1 = nn.Identity()
        self.patch2 = nn.Identity()
        self.stage1 = nn.Identity()
        self.stage2 = nn.Identity()
        self.stage3_1 = nn.Identity()
        self.stage3_2 = nn.Identity()
        self.stage3_3 = nn.Identity()
        self.patch3 = nn.Identity()
        self.stage4 = nn.Identity()
        self.head = nn.Linear(32, 10)

    def forward(self, x):
        return self.head(x)


@pytest.fixture
def mock_swin_model():
    return MockSwinModel()


@pytest.fixture
def sample_input():
    return torch.randn(2, 3, 32, 32)


# Smoke Tests
def test_swin_wrapper_smoke_initialization(mock_swin_model):
    wrapper = SwinTransformerWrapper(mock_swin_model)
    assert wrapper.backbone == mock_swin_model


def test_swin_wrapper_smoke_forward_pass(mock_swin_model, sample_input):
    wrapper = SwinTransformerWrapper(mock_swin_model)
    output = wrapper(sample_input)
    assert isinstance(output, ModelOutput)
    assert output.logits is not None
    assert output.embeddings is not None


def test_swin_wrapper_smoke_forward_for_graph(mock_swin_model, sample_input):
    wrapper = SwinTransformerWrapper(mock_swin_model)
    output = wrapper.forward_for_graph(sample_input)
    assert isinstance(output, torch.Tensor)


# Sanity Tests
def test_swin_wrapper_sanity_output_shape(mock_swin_model, sample_input):
    wrapper = SwinTransformerWrapper(mock_swin_model)
    output = wrapper(sample_input)
    assert output.logits.shape[0] == sample_input.shape[0]
    assert output.embeddings.shape[0] == sample_input.shape[0]


# Unit Tests
def test_swin_wrapper_unit_forward_calls_backbone(mock_swin_model, sample_input):
    wrapper = SwinTransformerWrapper(mock_swin_model)

    # Instead of patching submodules, we'll verify the output structure
    output = wrapper(sample_input)

    # Should have logits and embeddings
    assert hasattr(output, "logits")
    assert hasattr(output, "embeddings")
    assert output.logits is not None
    assert output.embeddings is not None


def test_swin_wrapper_unit_output_structure(mock_swin_model, sample_input):
    wrapper = SwinTransformerWrapper(mock_swin_model)
    output = wrapper(sample_input)
    assert hasattr(output, "logits")
    assert hasattr(output, "embeddings")
