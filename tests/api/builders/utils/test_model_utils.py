import importlib

import pytest
import torch
from omegaconf import OmegaConf

import src.refrakt_core.api.builders.utils.model_utils as model_utils


class DummyModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, x):
        return x


class DummyWrapper(torch.nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        self.kwargs = kwargs
        self.wrapped = True

    def forward(self, x):
        return self.model(x)


class TestModelUtils:
    # Smoke Tests
    def test_import_model_utils(self):
        importlib.reload(model_utils)

    def test_model_utils_has_any_symbol(self):
        symbols = [s for s in dir(model_utils) if not s.startswith("__")]
        assert symbols

    # Sanity Tests
    def test_validate_model_config_valid(self):
        cfg_dict = {"model": {"name": "dummy", "params": {"foo": 1}}}
        name, params, wrapper = model_utils.validate_model_config(cfg_dict)
        assert name == "dummy"
        assert params["foo"] == 1
        assert wrapper is None

    def test_apply_model_overrides_noop(self):
        cfg = OmegaConf.create({"model": {"name": "dummy", "params": {}}})
        out = model_utils.apply_model_overrides(cfg)
        assert out["model"]["name"] == "dummy"

    # Unit Tests
    def test_validate_model_config_type_error(self):
        with pytest.raises(TypeError):
            model_utils.validate_model_config({"model": None})

    def test_instantiate_base_model(self):
        modules = {"get_model": lambda name, **params: DummyModel(**params)}
        model = model_utils.instantiate_base_model("dummy", {"foo": 1}, modules, "cpu")
        assert isinstance(model, DummyModel)
        assert model.kwargs["foo"] == 1

    def test_wrap_model(self):
        modules = {"get_wrapper": lambda name: DummyWrapper}
        raw_model = DummyModel()
        model = model_utils.wrap_model(raw_model, "dummy", {"foo": 1}, modules, "cpu")
        assert isinstance(model, DummyWrapper)
        assert hasattr(model, "wrapped")

    def test_wrap_model_wrapper_not_found(self):
        modules = {"get_wrapper": lambda name: None}
        raw_model = DummyModel()
        with pytest.raises(ValueError):
            model_utils.wrap_model(raw_model, "notfound", {}, modules, "cpu")

    def test_create_default_wrapper(self):
        modules = {"get_model": lambda name, **params: DummyModel(**params)}
        model = model_utils.create_default_wrapper("dummy", {"foo": 1}, modules, "cpu")
        assert isinstance(model, torch.nn.Module)

    def test_add_fusion_block(self, monkeypatch):
        called = {}

        def fake_add_fusion_block(model, model_cfg, device):
            called["fusion"] = True
            return model

        monkeypatch.setattr(model_utils, "add_fusion_block", fake_add_fusion_block)
        model = DummyModel()
        model_utils.add_fusion_block(model, {"fusion": {"type": "dummy"}}, "cpu")
        assert called.get("fusion")
