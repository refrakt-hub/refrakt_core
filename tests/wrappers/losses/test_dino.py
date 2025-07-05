import pytest
import torch
from unittest.mock import patch
from refrakt_core.wrappers.losses.dino import DINOLossWrapper

@pytest.fixture
def student_out():
    # Shape: (B, num_views, D) - 4 batch size, 2 views, 1024 features
    return torch.randn(4, 2, 1024, requires_grad=True)

@pytest.fixture
def teacher_out():
    # Shape: (B, 1, D) - 4 batch size, 1 view, 1024 features
    return torch.randn(4, 1, 1024, requires_grad=True)

# Smoke Tests
def test_dino_loss_wrapper_smoke_initialization():
    wrapper = DINOLossWrapper()
    assert hasattr(wrapper, 'loss_fn')

def test_dino_loss_wrapper_smoke_forward(student_out, teacher_out):
    wrapper = DINOLossWrapper()
    loss = wrapper(student_out, teacher_out)
    assert hasattr(loss, 'total')
    assert hasattr(loss, 'components')

# Sanity Tests
def test_dino_loss_wrapper_sanity_loss_value(student_out, teacher_out):
    wrapper = DINOLossWrapper()
    loss = wrapper(student_out, teacher_out)
    assert loss.total is not None
    assert 'dino' in loss.components

# Unit Tests
def test_dino_loss_wrapper_unit_loss_fn_called(student_out, teacher_out):
    wrapper = DINOLossWrapper()
    with patch.object(wrapper.loss_fn, 'forward', wraps=wrapper.loss_fn.forward) as mock_call:
        wrapper(student_out, teacher_out)
        assert mock_call.called 