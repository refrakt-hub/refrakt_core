import pytest
import torch

# Skip all tests in this module if no GPU is available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA GPU is not available, skipping perceptual loss tests."
)

from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.wrappers.losses.perceptual import PerceptualLossWrapper


@pytest.fixture
def sr():
    return torch.randn(2, 3, 64, 64)


@pytest.fixture
def hr():
    return torch.randn(2, 3, 64, 64)


@pytest.fixture
def model_output(sr):
    return ModelOutput(reconstruction=sr)


# Smoke Tests
def test_perceptual_loss_wrapper_smoke_initialization():
    wrapper = PerceptualLossWrapper()
    assert hasattr(wrapper, "loss_fn")


def test_perceptual_loss_wrapper_smoke_forward(model_output, hr):
    wrapper = PerceptualLossWrapper()
    loss = wrapper(model_output, hr)
    assert hasattr(loss, "total")
    assert hasattr(loss, "components")


# Sanity Tests
def test_perceptual_loss_wrapper_sanity_loss_value(model_output, hr):
    wrapper = PerceptualLossWrapper()
    loss = wrapper(model_output, hr)
    assert loss.total is not None
    assert "perceptual" in loss.components


# Unit Tests
def test_perceptual_loss_wrapper_unit_missing_fields():
    wrapper = PerceptualLossWrapper()
    bad_output = ModelOutput(reconstruction=None)
    with pytest.raises(ValueError):
        wrapper(bad_output, None)


def test_perceptual_loss_wrapper_unit_required_fields(model_output, hr):
    wrapper = PerceptualLossWrapper()
    assert "sr" in wrapper.required_fields
    assert "hr" in wrapper.required_fields
