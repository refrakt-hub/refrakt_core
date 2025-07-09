import pytest
import torch
from torch import nn

from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.wrappers.losses.ce import CrossEntropyLossWrapper


@pytest.fixture
def sample_logits():
    return torch.randn(4, 3, requires_grad=True)


@pytest.fixture
def sample_targets():
    return torch.tensor([0, 1, 2, 1])


@pytest.fixture
def model_output(sample_logits):
    return ModelOutput(logits=sample_logits)


# Smoke Tests
def test_ce_wrapper_smoke_initialization():
    wrapper = CrossEntropyLossWrapper()
    assert isinstance(wrapper, nn.Module)


def test_ce_wrapper_smoke_forward(model_output, sample_targets):
    wrapper = CrossEntropyLossWrapper()
    loss = wrapper(model_output, sample_targets)
    assert hasattr(loss, "total")
    assert hasattr(loss, "components")


# Sanity Tests
def test_ce_wrapper_sanity_loss_value(model_output, sample_targets):
    wrapper = CrossEntropyLossWrapper()
    loss = wrapper(model_output, sample_targets)
    assert loss.total is not None
    assert "cross_entropy" in loss.components


# Unit Tests
def test_ce_wrapper_unit_missing_logits(sample_targets):
    wrapper = CrossEntropyLossWrapper()
    bad_output = ModelOutput(logits=None)
    with pytest.raises(ValueError):
        wrapper(bad_output, sample_targets)


def test_ce_wrapper_unit_missing_target(model_output):
    wrapper = CrossEntropyLossWrapper()
    with pytest.raises(ValueError):
        wrapper(model_output, None)


def test_ce_wrapper_unit_config_and_repr():
    wrapper = CrossEntropyLossWrapper(label_smoothing=0.1, device="cpu")
    config = wrapper.get_config()
    assert config["label_smoothing"] == 0.1
    assert config["device"] == "cpu"
    assert "label_smoothing" in wrapper.extra_repr()
