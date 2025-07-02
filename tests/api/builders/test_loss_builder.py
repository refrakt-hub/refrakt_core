import pytest
from omegaconf import OmegaConf
from torch import nn
from refrakt_core.api.builders.loss_builder import build_loss

class DummyLoss(nn.Module):
    def forward(self, x, y):
        return x.sum() - y.sum()

def dummy_get_loss(name, **params):
    return DummyLoss()

def test_build_loss_single():
    cfg = OmegaConf.create({'loss': {'name': 'dummy', 'params': {}}})
    modules = {'get_loss': dummy_get_loss}
    result = build_loss(cfg, modules, 'cpu')
    assert isinstance(result, nn.Module)

def test_build_loss_multi():
    cfg = OmegaConf.create({'loss': {'components': {'a': {'name': 'dummy', 'params': {}}}}})
    modules = {'get_loss': dummy_get_loss}
    result = build_loss(cfg, modules, 'cpu')
    assert isinstance(result, dict)
    assert 'a' in result
    assert isinstance(result['a'], nn.Module)

def test_build_loss_bad_type():
    cfg = OmegaConf.create({'loss': 123})  # not a dict
    modules = {'get_loss': dummy_get_loss}
    with pytest.raises(TypeError):
        build_loss(cfg, modules, 'cpu') 