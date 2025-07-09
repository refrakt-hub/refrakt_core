import pytest
import torch

from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.wrappers.losses.msn import MSNLossWrapper


@pytest.fixture
def z_anchor():
    # Shape: (B, D) - 2 batch size, 128 features
    return torch.randn(2, 128)


@pytest.fixture
def z_target():
    # Shape: (B, D) - 2 batch size, 128 features (same as z_anchor)
    return torch.randn(2, 128)


@pytest.fixture
def prototypes():
    # Shape: (K, D) - 10 prototypes, 128 features (same feature dimension)
    return torch.randn(10, 128)


@pytest.fixture
def model_output(z_anchor, z_target, prototypes):
    return ModelOutput(
        embeddings=z_anchor, extra={"z_target": z_target, "prototypes": prototypes}
    )


# Smoke Tests
def test_msn_loss_wrapper_smoke_initialization():
    wrapper = MSNLossWrapper()
    assert hasattr(wrapper, "loss_fn")


def test_msn_loss_wrapper_smoke_forward(model_output):
    wrapper = MSNLossWrapper()
    loss = wrapper(model_output)
    assert hasattr(loss, "total")
    assert hasattr(loss, "components")


# Sanity Tests
def test_msn_loss_wrapper_sanity_loss_value(model_output):
    wrapper = MSNLossWrapper()
    loss = wrapper(model_output)
    assert loss.total is not None
    assert isinstance(loss.components, dict)


# Unit Tests
def test_msn_loss_wrapper_unit_missing_fields():
    wrapper = MSNLossWrapper()
    bad_output = ModelOutput(embeddings=None, extra={})
    with pytest.raises(ValueError):
        wrapper(bad_output)


def test_msn_loss_wrapper_unit_extra_fields(model_output):
    wrapper = MSNLossWrapper()
    assert "z_target" in model_output.extra
    assert "prototypes" in model_output.extra
