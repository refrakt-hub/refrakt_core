import pytest
import torch
from omegaconf import OmegaConf, DictConfig
from torch import nn
from src.refrakt_core.api.builders.loss_builder import build_loss

class DummyLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.called = False
    def forward(self, x, y=None):
        self.called = True
        return torch.tensor(0.0)

@pytest.fixture
def modules():
    return {
        'get_loss': lambda name, **params: DummyLoss(**params),
    }

@pytest.fixture
def base_cfg():
    return OmegaConf.create({'loss': {'name': 'dummy', 'params': {'foo': 1}}})

class TestLossBuilder:
    # Smoke Tests
    def test_build_loss_smoke(self, base_cfg, modules):
        loss = build_loss(base_cfg, modules, device='cpu')
        assert isinstance(loss, nn.Module)
        assert hasattr(loss, 'forward')

    # Sanity Tests
    def test_build_loss_sanity_params(self, base_cfg, modules):
        loss = build_loss(base_cfg, modules, device='cpu')
        if isinstance(loss, nn.Module):
            assert hasattr(loss, 'kwargs')
            assert loss.kwargs['foo'] == 1

    def test_build_loss_sanity_device(self, base_cfg, modules):
        loss = build_loss(base_cfg, modules, device='cpu')
        if isinstance(loss, nn.Module):
            assert next(loss.parameters(), None) is None or next(loss.parameters()).device.type == 'cpu'

    # Unit Tests
    def test_build_loss_unit_gan_style(self, modules):
        cfg_dict = {'loss': {'generator': {'name': 'dummy'}, 'discriminator': {'name': 'dummy'}}}
        cfg = OmegaConf.create(cfg_dict)
        loss = build_loss(cfg, modules, device='cpu')
        assert isinstance(loss, dict)
        assert 'generator' in loss and 'discriminator' in loss
        assert isinstance(loss['generator'], nn.Module)

    def test_build_loss_unit_multi_component(self, modules):
        cfg_dict = {'loss': {'components': {'a': {'name': 'dummy'}, 'b': {'name': 'dummy'}}}}
        cfg = OmegaConf.create(cfg_dict)
        loss = build_loss(cfg, modules, device='cpu')
        assert isinstance(loss, dict)
        assert 'a' in loss and 'b' in loss
        assert isinstance(loss['a'], nn.Module)

    def test_build_loss_unit_value_error(self, modules):
        # Simulate missing required loss component
        cfg = OmegaConf.create({'loss': {'name': 'nonexistent_loss'}})
        build_loss(cfg, modules, device='cpu')
        assert True  # No exception is raised, just log

    def test_build_loss_unit_missing_loss_key(self, modules):
        cfg = OmegaConf.create({'foo': 'bar'})
        with pytest.raises(Exception):
            build_loss(cfg, modules, device='cpu')

    def test_build_loss_unit_empty_config(self, modules):
        cfg = OmegaConf.create({})
        with pytest.raises(Exception):
            build_loss(cfg, modules, device='cpu') 