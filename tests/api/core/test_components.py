import importlib

import pytest
import torch
from torch.optim.lr_scheduler import StepLR, _LRScheduler

from refrakt_core.api.core.components import ModelComponents
import refrakt_core.api.core.components as components


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)


class DummyLoss(torch.nn.Module):
    def forward(self, x, y):
        return (x - y).abs().mean()


class TestComponents:
    # Smoke Tests
    def test_import_components(self):
        import refrakt_core.api.core.components as components

        importlib.reload(components)

    def test_components_has_any_symbol(self):
        import refrakt_core.api.core.components as components

        symbols = [s for s in dir(components) if not s.startswith("__")]
        assert symbols

    # Sanity Tests
    def test_model_components_construction(self):
        model = DummyModel()
        loss_fn = DummyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler: _LRScheduler = StepLR(optimizer, step_size=1)  # type: ignore
        device = "cpu"
        mc = ModelComponents(model, loss_fn, optimizer, scheduler, device)
        assert mc.model is model
        assert mc.loss_fn is loss_fn
        assert mc.optimizer is optimizer
        assert mc.scheduler is scheduler
        assert mc.device == device

    # Unit Tests
    def test_model_components_defaults(self):
        model = DummyModel()
        loss_fn = DummyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        mc = ModelComponents(model, loss_fn, optimizer)
        assert mc.scheduler is None
        assert mc.device == "cuda" or mc.device == "cpu"
