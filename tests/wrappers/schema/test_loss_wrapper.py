import pytest
import torch
from refrakt_core.wrappers.schema.loss_wrapper import LossWrapper
from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.schema.loss_output import LossOutput

class DummyLoss:
    def __call__(self, output, target):
        return torch.abs(output - target).sum()

class DummyLossOutput:
    def __call__(self, output, target):
        return LossOutput(total=torch.tensor(1.0), components={"foo": torch.tensor(1.0)})

@pytest.fixture
def sample_output():
    return torch.randn(2, 10)

@pytest.fixture
def sample_target():
    return torch.randn(2, 10)

@pytest.fixture
def model_output(sample_output):
    return ModelOutput(logits=sample_output)

# Smoke Tests
def test_loss_wrapper_smoke_initialization():
    wrapper = LossWrapper(DummyLoss())
    assert hasattr(wrapper, 'fn')

def test_loss_wrapper_smoke_call_tensor(sample_output, sample_target):
    wrapper = LossWrapper(DummyLoss())
    loss = wrapper(sample_output, sample_target)
    assert hasattr(loss, 'total')

# Sanity Tests
def test_loss_wrapper_sanity_model_output(model_output, sample_target):
    wrapper = LossWrapper(DummyLoss(), field_map={"output": "logits", "target": None})
    loss = wrapper(model_output, sample_target)
    assert loss.total is not None

# Unit Tests
def test_loss_wrapper_unit_field_map(model_output, sample_target):
    def custom_loss(logits, target):
        return torch.abs(logits - target).sum()
    wrapper = LossWrapper(custom_loss, field_map={"logits": "logits"})
    loss = wrapper(model_output, sample_target)
    assert hasattr(loss, 'total')

def test_loss_wrapper_unit_loss_output_return(sample_output, sample_target):
    wrapper = LossWrapper(DummyLossOutput())
    loss = wrapper(sample_output, sample_target)
    assert isinstance(loss, LossOutput)

def test_loss_wrapper_unit_dict_return(sample_output, sample_target):
    def dict_loss(output, target):
        return {"a": torch.tensor(1.0), "b": torch.tensor(2.0)}
    wrapper = LossWrapper(dict_loss)
    loss = wrapper(sample_output, sample_target)
    assert isinstance(loss, LossOutput)
    assert set(loss.components.keys()) == {"a", "b"}
    assert loss.total == 3.0

def test_loss_wrapper_unit_tuple_return(sample_output, sample_target):
    def tuple_loss(output, target):
        return (torch.tensor(3.0), {"foo": torch.tensor(3.0)})
    wrapper = LossWrapper(tuple_loss)
    loss = wrapper(sample_output, sample_target)
    assert isinstance(loss, LossOutput)
    assert loss.total.item() == 3.0
    assert "foo" in loss.components

def test_loss_wrapper_unit_error_on_bad_return(sample_output, sample_target):
    def bad_loss(output, target):
        return None
    wrapper = LossWrapper(bad_loss)
    loss = wrapper(sample_output, sample_target)
    assert isinstance(loss, LossOutput)
    assert torch.isnan(loss.total) 