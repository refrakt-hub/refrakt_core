import pytest
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from refrakt_core.api.core.components import ModelComponents


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)


class DummyLoss(nn.Module):
    def forward(self, x, y):
        return ((x - y) ** 2).mean()


def test_model_components_smoke():
    model = DummyModel()
    loss_fn = DummyLoss()
    optimizer = SGD(model.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=1)
    mc = ModelComponents(model, loss_fn, optimizer, scheduler, device="cpu")
    assert mc.model is model
    assert mc.loss_fn is loss_fn
    assert mc.optimizer is optimizer
    assert mc.scheduler is scheduler
    assert mc.device == "cpu"


def test_model_components_sanity():
    model = DummyModel()
    loss_fn = DummyLoss()
    optimizer = SGD(model.parameters(), lr=0.01)
    mc = ModelComponents(model, loss_fn, optimizer)
    assert isinstance(mc.model, nn.Module)
    assert isinstance(mc.loss_fn, nn.Module)
    assert isinstance(mc.optimizer, SGD)
    assert mc.scheduler is None
    assert mc.device == "cuda" or mc.device == "cpu"


def test_model_components_unit():
    model = DummyModel()
    loss_fn = DummyLoss()
    optimizer = SGD(model.parameters(), lr=0.01)
    mc = ModelComponents(model, loss_fn, optimizer, device="cpu")
    # Test attribute assignment
    mc.device = "cuda"
    assert mc.device == "cuda"
