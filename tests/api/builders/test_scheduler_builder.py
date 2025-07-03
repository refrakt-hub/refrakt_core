import pytest
import torch
from omegaconf import OmegaConf

from refrakt_core.api.builders.scheduler_builder import build_scheduler


def test_build_scheduler_smoke():
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    cfg = OmegaConf.create({"scheduler": {"name": "cosine", "params": {"T_max": 10}}})
    sched = build_scheduler(cfg, optimizer)
    assert sched is not None
    assert hasattr(sched, "step")


def test_build_scheduler_bad_type():
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    cfg = OmegaConf.create({"scheduler": 123})
    sched = build_scheduler(cfg, optimizer)
    assert sched is None
