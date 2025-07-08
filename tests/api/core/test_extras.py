import importlib
import pytest
import types
import torch
from typing import Any, cast
from src.refrakt_core.api.core import extras

class TestExtras:
    # Smoke Tests
    def test_import_extras(self):
        import src.refrakt_core.api.core.extras as extras_mod
        importlib.reload(extras_mod)

    # Sanity Tests
    def test_import_modules_returns_dict(self):
        # Patch registry imports to avoid import errors
        import builtins
        import types
        orig_import = builtins.__import__
        def fake_import(name, *args, **kwargs):
            if name.startswith("refrakt_core.registry"):
                # Create a mock module with all expected functions
                mock_module = types.SimpleNamespace()
                mock_module.get_loss = lambda: "mock_loss"
                mock_module.get_model = lambda: "mock_model"
                mock_module.get_trainer = lambda: "mock_trainer"
                mock_module.get_wrapper = lambda: "mock_wrapper"
                mock_module.get_transform = lambda: "mock_transform"
                return mock_module
            return orig_import(name, *args, **kwargs)
        builtins.__import__ = fake_import
        try:
            result = extras.import_modules()
            assert isinstance(result, dict)
        finally:
            builtins.__import__ = orig_import

    def test_setup_device_returns_str(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert extras.setup_device() == "cpu"
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert extras.setup_device() == "cuda"

    # Unit Tests
    def test_flatten_and_filter_config(self):
        cfg = {"a": 1, "b": {"c": 2.0, "d": {"e": "x"}}, "f": [1,2], "g": None}
        flat = extras.flatten_and_filter_config(cfg)
        assert flat["a"] == 1
        assert flat["b.c"] == 2.0
        assert flat["b.d.e"] == "x"
        assert "f" not in flat and "g" not in flat

    def test_build_datasets_and_dataloaders(self, monkeypatch):
        # Patch build_dataset and build_dataloader
        monkeypatch.setattr(extras, "build_dataset", lambda cfg: "ds" if getattr(cfg, "dataset", None) is None else "ds2")
        monkeypatch.setattr(extras, "OmegaConf", type("OmegaConf", (), {
            "merge": staticmethod(lambda a, b: a),
            "create": staticmethod(lambda x: x)
        }))
        DummyCfg = types.SimpleNamespace
        dummy_cfg = DummyCfg(dataset=DummyCfg(), dataloader=DummyCfg())
        train, val = extras.build_datasets(cast(Any, dummy_cfg))
        assert train in ("ds", "ds2")
        assert val in ("ds", "ds2")
        monkeypatch.setattr(extras, "build_dataloader", lambda ds, cfg: [1,2,3])
        train_loader, val_loader = extras.build_dataloaders("ds", "ds", cast(Any, dummy_cfg))
        assert len(train_loader) == 3
        assert len(val_loader) == 3

    def test_build_model_components(self, monkeypatch):
        # Patch all builder functions and ModelComponents
        monkeypatch.setattr(extras, "import_modules", lambda: {})
        monkeypatch.setattr(extras, "setup_device", lambda: "cpu")
        monkeypatch.setattr(extras, "build_model", lambda cfg, modules, device: "model")
        monkeypatch.setattr(extras, "build_loss", lambda cfg, modules, device: "loss")
        monkeypatch.setattr(extras, "build_optimizer", lambda cfg, model: "optimizer")
        monkeypatch.setattr(extras, "build_scheduler", lambda cfg, optimizer: "scheduler")
        class DummyMC:
            def __init__(self, model, loss_fn, optimizer, scheduler, device):
                self.model = model; self.loss_fn = loss_fn; self.optimizer = optimizer
                self.scheduler = scheduler; self.device = device
        monkeypatch.setattr(extras, "ModelComponents", DummyMC)
        DummyCfg = types.SimpleNamespace
        dummy_cfg = DummyCfg()
        mc = extras.build_model_components(cast(Any, dummy_cfg))
        assert mc.model == "model"
        assert mc.loss_fn == "loss"
        assert mc.optimizer == "optimizer"
        assert mc.scheduler == "scheduler"
        assert mc.device == "cpu" 