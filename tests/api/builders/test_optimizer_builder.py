import pytest
from omegaconf import OmegaConf
import torch
from torch import nn
from refrakt_core.api.builders.optimizer_builder import build_optimizer

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
    def forward(self, x):
        return self.linear(x)

def test_build_optimizer_smoke():
    cfg = OmegaConf.create({'optimizer': {'name': 'adam', 'params': OmegaConf.create({'lr': 0.01})}})
    model = DummyModel()
    opt = build_optimizer(cfg, model)
    assert isinstance(opt, torch.optim.Optimizer)

def test_build_optimizer_bad_type():
    cfg = OmegaConf.create({'optimizer': 'not_a_dict'})
    model = DummyModel()
    with pytest.raises(TypeError):
        build_optimizer(cfg, model) 