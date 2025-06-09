import pytest
import torch

from refrakt_core.losses.dino import DINOLoss


def test_dino_loss_basic():
    loss_fn = DINOLoss(out_dim=256)
    student_output = torch.randn(4, 3, 256)
    teacher_output = torch.randn(4, 1, 256)
    
    loss = loss_fn(student_output, teacher_output)
    assert loss.item() > 0
    assert isinstance(loss, torch.Tensor)

def test_dino_loss_center_update():
    loss_fn = DINOLoss(out_dim=256)
    original_center = loss_fn.center.clone()
    
    student_output = torch.randn(4, 3, 256)
    teacher_output = torch.randn(4, 1, 256)
    loss_fn(student_output, teacher_output)
    
    assert not torch.equal(loss_fn.center, original_center)

def test_dino_loss_shape_mismatch():
    loss_fn = DINOLoss(out_dim=256)
    
    # Batch size mismatch
    student_output = torch.randn(4, 3, 256)
    teacher_output = torch.randn(3, 1, 256)
    with pytest.raises((RuntimeError, ValueError)):
        loss_fn(student_output, teacher_output)
    
    # Feature dim mismatch
    student_output = torch.randn(4, 3, 256)
    teacher_output = torch.randn(4, 1, 128)
    with pytest.raises((RuntimeError, ValueError)):
        loss_fn(student_output, teacher_output)

def test_dino_loss_temperature_effects():
    loss_fn_high_temp = DINOLoss(out_dim=256, student_temp=0.5, teacher_temp=0.5)
    loss_fn_low_temp = DINOLoss(out_dim=256, student_temp=0.01, teacher_temp=0.01)
    
    student_output = torch.randn(4, 3, 256)
    teacher_output = torch.randn(4, 1, 256)
    
    loss_high = loss_fn_high_temp(student_output, teacher_output)
    loss_low = loss_fn_low_temp(student_output, teacher_output)
    
    assert not torch.isclose(loss_high, loss_low)

def test_dino_registry_integration():
    from refrakt_core.registry.loss_registry import get_loss
    dino_loss = get_loss("dino", out_dim=128)
    assert isinstance(dino_loss, DINOLoss)
    assert dino_loss.center.shape[1] == 128