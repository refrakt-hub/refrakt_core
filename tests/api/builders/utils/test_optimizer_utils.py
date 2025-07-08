import importlib
import pytest
import torch
from omegaconf import OmegaConf
import src.refrakt_core.api.builders.utils.optimizer_utils as optimizer_utils

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x):
        return self.linear(x)

class DummyGAN:
    def __init__(self):
        self.generator = DummyModel()
        self.discriminator = DummyModel()

class TestOptimizerUtils:
    # Smoke Tests
    def test_import_optimizer_utils(self):
        importlib.reload(optimizer_utils)

    def test_optimizer_utils_has_any_symbol(self):
        symbols = [s for s in dir(optimizer_utils) if not s.startswith('__')]
        assert symbols

    # Sanity Tests
    def test_get_optimizer_map(self):
        opt_map = optimizer_utils.get_optimizer_map()
        assert 'adam' in opt_map and 'sgd' in opt_map

    def test_validate_optimizer_params_valid(self):
        params = {'lr': 0.01}
        out = optimizer_utils.validate_optimizer_params(params)
        assert out['lr'] == 0.01

    # Unit Tests
    def test_validate_optimizer_params_type_error(self):
        with pytest.raises(TypeError):
            optimizer_utils.validate_optimizer_params(123)

    def test_get_model_parameters_generator(self):
        model = DummyGAN()
        params = optimizer_utils.get_model_parameters(model, 'generator')
        assert hasattr(params, '__iter__')

    def test_get_model_parameters_discriminator(self):
        model = DummyGAN()
        params = optimizer_utils.get_model_parameters(model, 'discriminator')
        assert hasattr(params, '__iter__')

    def test_get_model_parameters_value_error(self):
        model = DummyGAN()
        with pytest.raises(ValueError):
            optimizer_utils.get_model_parameters(model, 'unknown')

    def test_build_component_optimizer(self):
        opt_map = optimizer_utils.get_optimizer_map()
        comp_cfg = {'name': 'adam', 'params': {'lr': 0.01}}
        model = DummyGAN()
        opt = optimizer_utils.build_component_optimizer(comp_cfg, model, 'generator', opt_map)
        assert isinstance(opt, torch.optim.Optimizer)

    def test_build_gan_style_optimizer(self):
        opt_map = optimizer_utils.get_optimizer_map()
        optimizer_cfg = {
            'generator': {'name': 'adam', 'params': {'lr': 0.01}},
            'discriminator': {'name': 'adam', 'params': {'lr': 0.02}}
        }
        model = DummyGAN()
        out = optimizer_utils.build_gan_style_optimizer(optimizer_cfg, model, opt_map)
        assert 'generator' in out and 'discriminator' in out
        assert isinstance(out['generator'], torch.optim.Optimizer)

    def test_build_multi_component_optimizer(self):
        opt_map = optimizer_utils.get_optimizer_map()
        optimizer_cfg = {'components': {'a': {'name': 'adam', 'params': {'lr': 0.01}}, 'b': {'name': 'adam', 'params': {'lr': 0.02}}}}
        model = {'a': DummyModel(), 'b': DummyModel()}
        with pytest.raises(ValueError):
            optimizer_utils.build_multi_component_optimizer(optimizer_cfg, model, opt_map)

    def test_build_multi_component_optimizer_type_error(self):
        opt_map = optimizer_utils.get_optimizer_map()
        optimizer_cfg = {'components': 123}
        model = {'a': DummyModel()}
        with pytest.raises(TypeError):
            optimizer_utils.build_multi_component_optimizer(optimizer_cfg, model, opt_map) 