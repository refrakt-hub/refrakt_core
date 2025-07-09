from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.wrappers.schema.default_model import DefaultModelWrapper


class DummyModel(nn.Module):
    def forward(self, x, **kwargs):
        return torch.randn(x.shape[0], 10)


def get_model(model_name, **params):
    return DummyModel()


@pytest.fixture
def model_name():
    return "dummy"


@pytest.fixture
def model_params():
    return {"param1": 1}


@pytest.fixture
def modules():
    return {"get_model": get_model}


@pytest.fixture
def sample_input():
    return torch.randn(2, 5)


# Smoke Tests
def test_default_model_wrapper_smoke_initialization(model_name, model_params, modules):
    wrapper = DefaultModelWrapper(model_name, model_params, modules)
    assert isinstance(wrapper, nn.Module)
    assert hasattr(wrapper, "model")


def test_default_model_wrapper_smoke_forward(
    model_name, model_params, modules, sample_input
):
    wrapper = DefaultModelWrapper(model_name, model_params, modules)
    output = wrapper(sample_input)
    assert isinstance(output, ModelOutput)
    assert output.logits is not None


# Sanity Tests
def test_default_model_wrapper_sanity_tensor_output(
    model_name, model_params, modules, sample_input
):
    class TensorModel(nn.Module):
        def forward(self, x, **kwargs):
            return torch.randn(x.shape[0], 10)

    modules = {"get_model": lambda n, **p: TensorModel()}
    wrapper = DefaultModelWrapper(model_name, model_params, modules)
    output = wrapper(sample_input)
    assert isinstance(output, ModelOutput)
    assert output.logits is not None


def test_default_model_wrapper_sanity_dict_output(
    model_name, model_params, modules, sample_input
):
    class DictModel(nn.Module):
        def forward(self, x, **kwargs):
            return {
                "logits": torch.randn(x.shape[0], 10),
                "embeddings": torch.randn(x.shape[0], 5),
            }

    modules = {"get_model": lambda n, **p: DictModel()}
    wrapper = DefaultModelWrapper(model_name, model_params, modules)
    output = wrapper(sample_input)
    assert isinstance(output, ModelOutput)
    assert output.logits is not None
    assert output.embeddings is not None


# Unit Tests
def test_default_model_wrapper_unit_parameters_method(
    model_name, model_params, modules
):
    wrapper = DefaultModelWrapper(model_name, model_params, modules)
    params = list(wrapper.parameters())
    assert isinstance(params, list)


def test_default_model_wrapper_unit_raises_on_missing_get_model(
    model_name, model_params
):
    with pytest.raises(ValueError):
        DefaultModelWrapper(model_name, model_params, modules={})


def test_default_model_wrapper_unit_raises_on_unsupported_output(
    model_name, model_params, modules, sample_input
):
    class BadModel(nn.Module):
        def forward(self, x, **kwargs):
            return 42

    modules = {"get_model": lambda n, **p: BadModel()}
    wrapper = DefaultModelWrapper(model_name, model_params, modules)
    with pytest.raises(ValueError):
        wrapper(sample_input)
