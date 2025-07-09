import pytest
import torch

from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.wrappers.losses.gan import GANLossWrapper


@pytest.fixture
def sample_logits():
    return torch.randn(4, 1, requires_grad=True)


@pytest.fixture
def model_output_real(sample_logits):
    return ModelOutput(logits=sample_logits, extra={"target_is_real": True})


@pytest.fixture
def model_output_fake(sample_logits):
    return ModelOutput(logits=sample_logits, extra={"target_is_real": False})


# Smoke Tests
def test_gan_loss_wrapper_smoke_initialization():
    wrapper = GANLossWrapper()
    assert hasattr(wrapper, "loss_fn")


def test_gan_loss_wrapper_smoke_forward(model_output_real):
    wrapper = GANLossWrapper()
    loss = wrapper(model_output_real)
    assert hasattr(loss, "total")
    assert hasattr(loss, "components")


# Sanity Tests
def test_gan_loss_wrapper_sanity_loss_value(model_output_fake):
    wrapper = GANLossWrapper()
    loss = wrapper(model_output_fake)
    assert loss.total is not None
    assert "gan" in loss.components


# Unit Tests
def test_gan_loss_wrapper_unit_missing_fields():
    wrapper = GANLossWrapper()
    bad_output = ModelOutput(logits=None, extra={})
    with pytest.raises(ValueError):
        wrapper(bad_output)


def test_gan_loss_wrapper_unit_required_fields(model_output_real):
    wrapper = GANLossWrapper()
    assert set(wrapper.required_fields) == {"logits", "target_is_real"}
