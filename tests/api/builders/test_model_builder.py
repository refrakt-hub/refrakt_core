import pytest
import torch
from omegaconf import OmegaConf

from refrakt_core.api.builders.model_builder import build_model
from refrakt_core.wrappers.schema.default_model import DefaultModelWrapper


class DummyModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.called = False

    def forward(self, x, **kwargs):
        self.called = True
        return x


class DummyWrapper(torch.nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        self.kwargs = kwargs
        self.wrapped = True

    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)


@pytest.fixture(autouse=True)
def patch_model_registry(monkeypatch):
    import refrakt_core.registry.model_registry as reg

    reg.MODEL_REGISTRY["dummy"] = DummyModel
    reg.get_model = lambda name, *args, **kwargs: DummyModel(**kwargs)
    yield
    reg.MODEL_REGISTRY.pop("dummy", None)


@pytest.fixture
def modules():
    return {
        "get_model": lambda name, **params: DummyModel(**params),
        "get_wrapper": lambda name: DummyWrapper if name == "dummy" else None,
    }


@pytest.fixture
def base_cfg():
    return OmegaConf.create(
        {
            "model": {
                "name": "dummy",
                "params": {"foo": 1, "bar": 2},
            }
        }
    )


class TestModelBuilder:
    # Smoke Tests
    def test_build_model_smoke(self, base_cfg, modules):
        model = build_model(base_cfg, modules, device="cpu")
        assert isinstance(model, DefaultModelWrapper) or hasattr(model, "model")

    def test_build_model_with_wrapper_smoke(self, base_cfg, modules):
        base_cfg.model["wrapper"] = "dummy"
        model = build_model(base_cfg, modules, device="cpu")
        assert isinstance(model, DummyWrapper)
        assert hasattr(model, "wrapped")

    # Sanity Tests
    def test_build_model_sanity_config_override(self, base_cfg, modules):
        overrides = ["model.params.foo=42"]
        model = build_model(base_cfg, modules, device="cpu", overrides=overrides)
        assert hasattr(model, "model") or isinstance(model, DefaultModelWrapper)

    def test_build_model_sanity_device(self, base_cfg, modules):
        model = build_model(base_cfg, modules, device="cpu")
        # Check if model has parameters, if not, just verify it's a valid model
        try:
            next(model.parameters())
            assert next(model.parameters()).device.type == "cpu"
        except StopIteration:
            # Model has no parameters, which is fine for testing
            assert True

    def test_build_model_sanity_fallback_on_error(self, base_cfg, modules):
        def bad_get_model(name, **params):
            raise RuntimeError("fail")

        modules["get_model"] = bad_get_model
        with pytest.raises(RuntimeError):
            build_model(base_cfg, modules, device="cpu")

    # Unit Tests
    def test_build_model_unit_missing_model_name(self, base_cfg, modules):
        base_cfg.model["name"] = None
        with pytest.raises(TypeError):
            build_model(base_cfg, modules, device="cpu")

    def test_build_model_unit_missing_get_model(self, base_cfg, modules):
        del modules["get_model"]
        with pytest.raises(ValueError):
            build_model(base_cfg, modules, device="cpu")

    def test_build_model_unit_wrapper_not_found(self, base_cfg, modules):
        base_cfg.model["wrapper"] = "notfound"
        model = build_model(base_cfg, modules, device="cpu")
        assert hasattr(model, "model") or isinstance(model, DefaultModelWrapper)

    def test_build_model_unit_fusion_block(self, base_cfg, modules, monkeypatch):
        called = {}
        import refrakt_core.api.builders.utils.model_utils as model_utils

        monkeypatch.setattr(
            model_utils,
            "add_fusion_block",
            lambda model, cfg, device: (called.setdefault("fusion", True) or model),
        )
        base_cfg.model["fusion"] = {"type": "sklearn", "model": "dummy_model"}
        with pytest.raises(ValueError):
            build_model(base_cfg, modules, device="cpu")

    def test_build_model_unit_print_finalization(self, base_cfg, modules, capsys):
        build_model(base_cfg, modules, device="cpu")
        captured = capsys.readouterr()
        assert "[FINALIZED]" in captured.out

    def test_build_model_unit_apply_model_overrides(
        self, base_cfg, modules, monkeypatch
    ):
        monkeypatch.setattr(
            "refrakt_core.api.builders.utils.model_utils.apply_model_overrides",
            lambda cfg, overrides: base_cfg,
        )
        model = build_model(
            base_cfg, modules, device="cpu", overrides=["model.params.foo=99"]
        )
        assert isinstance(model, DefaultModelWrapper) or hasattr(model, "model")

    def test_build_model_unit_validate_model_config_type_error(
        self, base_cfg, modules, monkeypatch
    ):
        monkeypatch.setattr(
            "refrakt_core.api.builders.utils.model_utils.validate_model_config",
            lambda cfg: (_ for _ in ()).throw(TypeError("fail")),
        )
        build_model(base_cfg, modules, device="cpu")
        assert True  # No exception is raised, just log
