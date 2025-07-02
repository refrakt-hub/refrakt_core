import pytest
from omegaconf import OmegaConf
from torch import nn
from refrakt_core.api.builders.model_builder import build_model

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
    def forward(self, x):
        return self.linear(x)

def dummy_get_model(name):
    return DummyModel

def dummy_get_wrapper(name):
    class Wrapper(nn.Module):
        def __init__(self, model, **kwargs):
            super().__init__()
            self.model = model
        def forward(self, x):
            return self.model(x)
    return Wrapper

def test_build_model_smoke():
    cfg = OmegaConf.create({'model': {'name': 'dummy', 'params': {}}})
    modules = {'get_model': dummy_get_model, 'get_wrapper': dummy_get_wrapper}
    result = build_model(cfg, modules, 'cpu')
    assert isinstance(result, nn.Module)

def test_build_model_with_wrapper():
    cfg = OmegaConf.create({'model': {'name': 'dummy', 'params': {}, 'wrapper': 'wrap'}})
    modules = {'get_model': dummy_get_model, 'get_wrapper': dummy_get_wrapper}
    result = build_model(cfg, modules, 'cpu')
    assert isinstance(result, nn.Module)

def test_build_model_bad_type():
    cfg = OmegaConf.create({'model': 'not_a_dict'})
    modules = {'get_model': dummy_get_model}
    with pytest.raises(TypeError):
        build_model(cfg, modules, 'cpu') 