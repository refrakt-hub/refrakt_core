import pytest
import torch

from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.wrappers.losses.mae import MAELossWrapper


@pytest.fixture
def recon():
    # Shape: (B, C, H, W) - 2 batch size, 3 channels, 64x64 image
    return torch.randn(2, 3, 64, 64)


@pytest.fixture
def mask():
    # Shape: (B, N) - 2 batch size, 16 patches (4x4 grid)
    return torch.randint(0, 2, (2, 16))


@pytest.fixture
def patches():
    # Shape: (B, N, patch_dim) - 2 batch size, 16 patches, 768 features (16*16*3)
    return torch.randn(2, 16, 768)


@pytest.fixture
def model_output(recon, mask, patches):
    return ModelOutput(
        reconstruction=recon, extra={"mask": mask, "original_patches": patches}
    )


# Smoke Tests
def test_mae_loss_wrapper_smoke_initialization():
    wrapper = MAELossWrapper()
    assert hasattr(wrapper, "loss_fn")


def test_mae_loss_wrapper_smoke_forward(model_output):
    wrapper = MAELossWrapper()
    loss = wrapper(model_output)
    assert hasattr(loss, "total")
    assert hasattr(loss, "components")


# Sanity Tests
def test_mae_loss_wrapper_sanity_loss_value(model_output):
    wrapper = MAELossWrapper()
    loss = wrapper(model_output)
    assert loss.total is not None
    assert "masked_mse" in loss.components


# Unit Tests
def test_mae_loss_wrapper_unit_missing_fields():
    wrapper = MAELossWrapper()
    bad_output = ModelOutput(reconstruction=None, extra={})
    with pytest.raises(ValueError):
        wrapper(bad_output)


def test_mae_loss_wrapper_unit_extra_fields(model_output):
    wrapper = MAELossWrapper()
    assert "mask" in model_output.extra
    assert "original_patches" in model_output.extra
