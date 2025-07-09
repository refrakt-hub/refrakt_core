import pytest
import torch

from refrakt_core.wrappers.losses.ntxent import NTXentLossWrapper


@pytest.fixture
def z1():
    return torch.randn(4, 128, requires_grad=True)


@pytest.fixture
def z2():
    return torch.randn(4, 128, requires_grad=True)


# Smoke Tests
def test_ntxent_loss_wrapper_smoke_initialization():
    wrapper = NTXentLossWrapper()
    assert hasattr(wrapper, "inner_loss")


def test_ntxent_loss_wrapper_smoke_forward(z1, z2):
    wrapper = NTXentLossWrapper()
    loss = wrapper(z1, z2)
    assert hasattr(loss, "total")
    assert hasattr(loss, "components")


# Sanity Tests
def test_ntxent_loss_wrapper_sanity_loss_value(z1, z2):
    wrapper = NTXentLossWrapper()
    loss = wrapper(z1, z2)
    assert loss.total is not None
    assert "ntxent" in loss.components


# Unit Tests
def test_ntxent_loss_wrapper_unit_temperature():
    wrapper = NTXentLossWrapper(temperature=0.7)
    assert wrapper.temperature == 0.7


def test_ntxent_loss_wrapper_unit_config():
    wrapper = NTXentLossWrapper(name="test_loss", temperature=0.3, custom_param=42)
    assert wrapper.config["custom_param"] == 42
