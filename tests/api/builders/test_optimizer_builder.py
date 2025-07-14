import pytest
import torch
from omegaconf import OmegaConf

from refrakt_core.api.builders.optimizer_builder import build_optimizer


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


@pytest.fixture
def base_cfg():
    return OmegaConf.create({"optimizer": {"name": "adam", "params": {"lr": 0.01}}})


class TestOptimizerBuilder:
    # Smoke Tests
    def test_build_optimizer_smoke(self, base_cfg):
        model = DummyModel()
        opt = build_optimizer(base_cfg, model)
        assert isinstance(opt, torch.optim.Optimizer)

    # Sanity Tests
    def test_build_optimizer_sanity_params(self, base_cfg):
        model = DummyModel()
        opt = build_optimizer(base_cfg, model)
        if isinstance(opt, torch.optim.Optimizer):
            assert opt.defaults["lr"] == 0.01

    def test_build_optimizer_sanity_device(self, base_cfg):
        model = DummyModel()
        opt = build_optimizer(base_cfg, model)
        if isinstance(opt, torch.optim.Optimizer):
            for group in opt.param_groups:
                for p in group["params"]:
                    assert p.device.type == "cpu"

    # Unit Tests
    def test_build_optimizer_unit_missing_name(self, base_cfg):
        base_cfg.optimizer["name"] = None
        model = DummyModel()
        with pytest.raises(TypeError):
            build_optimizer(base_cfg, model)

    def test_build_optimizer_unit_params_not_dict(self, base_cfg):
        base_cfg.optimizer["params"] = 123
        model = DummyModel()
        with pytest.raises(TypeError):
            build_optimizer(base_cfg, model)

    def test_build_optimizer_unit_gan_style(self):
        cfg = OmegaConf.create(
            {
                "optimizer": {
                    "generator": {"name": "adam", "params": {"lr": 0.001}},
                    "discriminator": {"name": "adam", "params": {"lr": 0.002}},
                }
            }
        )
        model = DummyGAN()
        opt = build_optimizer(cfg, model)
        assert isinstance(opt, dict)
        assert "generator" in opt and "discriminator" in opt
        assert isinstance(opt["generator"], torch.optim.Optimizer)

    def test_build_optimizer_unit_multi_component(self):
        cfg = OmegaConf.create(
            {
                "optimizer": {
                    "components": {
                        "a": {"name": "adam", "params": {"lr": 0.01}},
                        "b": {"name": "adam", "params": {"lr": 0.02}},
                    }
                }
            }
        )
        model = {"a": DummyModel(), "b": DummyModel()}
        with pytest.raises(ValueError):
            build_optimizer(cfg, model)

    def test_build_optimizer_unit_invalid_model_type(self, base_cfg):
        with pytest.raises(AttributeError):
            build_optimizer(base_cfg, "not_a_model")

    def test_build_optimizer_unit_missing_optimizer_key(self):
        cfg = OmegaConf.create({"foo": "bar"})
        model = DummyModel()
        with pytest.raises(Exception):
            build_optimizer(cfg, model)

    def test_build_optimizer_unit_empty_config(self):
        cfg = OmegaConf.create({})
        model = DummyModel()
        with pytest.raises(Exception):
            build_optimizer(cfg, model)
