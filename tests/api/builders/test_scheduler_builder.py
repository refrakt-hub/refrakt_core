import pytest
import torch
from omegaconf import OmegaConf

from src.refrakt_core.api.builders.scheduler_builder import build_scheduler


@pytest.fixture
def optimizer():
    model = torch.nn.Linear(2, 2)
    return torch.optim.Adam(model.parameters(), lr=0.01)


@pytest.fixture
def base_cfg():
    return OmegaConf.create({"scheduler": {"name": "cosine", "params": {"T_max": 10}}})


class TestSchedulerBuilder:
    # Smoke Tests
    def test_build_scheduler_smoke(self, base_cfg, optimizer):
        sched = build_scheduler(base_cfg, optimizer)
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_build_scheduler_none_smoke(self, optimizer):
        cfg = OmegaConf.create({})
        sched = build_scheduler(cfg, optimizer)
        assert sched is None

    # Sanity Tests
    def test_build_scheduler_sanity_params(self, base_cfg, optimizer):
        sched = build_scheduler(base_cfg, optimizer)
        if hasattr(sched, "T_max"):
            assert sched.T_max == 10

    def test_build_scheduler_sanity_device(self, base_cfg, optimizer):
        sched = build_scheduler(base_cfg, optimizer)
        if hasattr(sched, "optimizer"):
            for group in sched.optimizer.param_groups:
                for p in group["params"]:
                    assert p.device.type == "cpu"

    # Unit Tests
    def test_build_scheduler_unit_missing_name(self, base_cfg, optimizer):
        base_cfg.scheduler["name"] = 123  # Use a non-string type to trigger TypeError
        with pytest.raises(TypeError):
            build_scheduler(base_cfg, optimizer)

    def test_build_scheduler_unit_params_not_dict(self, base_cfg, optimizer):
        base_cfg.scheduler["params"] = 123
        with pytest.raises(TypeError):
            build_scheduler(base_cfg, optimizer)

    def test_build_scheduler_unit_unsupported_scheduler(self, optimizer):
        cfg = OmegaConf.create({"scheduler": {"name": "unknown", "params": {}}})
        with pytest.raises(ValueError):
            build_scheduler(cfg, optimizer)

    def test_build_scheduler_unit_exponential(self, optimizer):
        cfg = OmegaConf.create(
            {"scheduler": {"name": "exponential", "params": {"gamma": 0.9}}}
        )
        sched = build_scheduler(cfg, optimizer)
        assert isinstance(sched, torch.optim.lr_scheduler.ExponentialLR)

    def test_build_scheduler_unit_steplr(self, optimizer):
        cfg = OmegaConf.create(
            {"scheduler": {"name": "steplr", "params": {"step_size": 2, "gamma": 0.5}}}
        )
        sched = build_scheduler(cfg, optimizer)
        assert isinstance(sched, torch.optim.lr_scheduler.StepLR)

    def test_build_scheduler_unit_multisteplr(self, optimizer):
        cfg = OmegaConf.create(
            {
                "scheduler": {
                    "name": "multisteplr",
                    "params": {"milestones": [1, 2], "gamma": 0.5},
                }
            }
        )
        sched = build_scheduler(cfg, optimizer)
        assert isinstance(sched, torch.optim.lr_scheduler.MultiStepLR)

    def test_build_scheduler_unit_invalid_optimizer_type(self, base_cfg):
        with pytest.raises(Exception):
            build_scheduler(base_cfg, "not_an_optimizer")
