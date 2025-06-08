# test_dino.py
import pytest
import torch
import copy

from refrakt_core.wrappers.dino import DINOBackboneWrapper, DINOModelWrapper
from refrakt_core.models.resnet import ResNet18

@pytest.fixture
def dino_model():
    backbone = ResNet18(num_classes=1)
    wrapped_backbone = DINOBackboneWrapper(backbone)
    return DINOModelWrapper(wrapped_backbone)

def test_dino_init(dino_model):
    assert dino_model.model_name == "dino"
    assert dino_model.model_type == "contrastive"

def test_dino_forward(dino_model):
    x = torch.randn(2, 3, 224, 224)
    
    # Student forward
    student_out = dino_model(x, teacher=False)
    assert student_out.shape == (2, 65536)
    
    # Teacher forward
    teacher_out = dino_model(x, teacher=True)
    assert teacher_out.shape == (2, 65536)

import torch.nn as nn

def test_dino_update_teacher(dino_model):
    # Grab the first Linear layer from the teacher_head
    linear_layers = [layer for layer in dino_model.teacher_head.mlp if isinstance(layer, nn.Linear)]
    assert linear_layers, "No Linear layers found in teacher_head.mlp"

    init_weights = copy.deepcopy(linear_layers[0].weight.data)
    dino_model.update_teacher(momentum=0.9)
    updated_weights = linear_layers[0].weight.data

    assert not torch.equal(init_weights, updated_weights)
