import pytest
from omegaconf import OmegaConf

from src.refrakt_core.api.builders.transform_builder import build_transform


class DummyTransform:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, x):
        return x + 1


@pytest.fixture(autouse=True)
def patch_transform_registry(monkeypatch):
    import src.refrakt_core.registry.transform_registry as reg

    reg.TRANSFORM_REGISTRY["dummy"] = DummyTransform
    reg.TRANSFORM_REGISTRY["dummy_transform"] = DummyTransform
    reg.get_transform = lambda name, *args, **kwargs: DummyTransform(**kwargs)
    yield
    reg.TRANSFORM_REGISTRY.pop("dummy", None)
    reg.TRANSFORM_REGISTRY.pop("dummy_transform", None)


class TestTransformBuilder:
    # Smoke Tests
    def test_build_transform_smoke(self):
        cfg = [{"name": "dummy", "params": {}}]
        t = build_transform(cfg)
        assert callable(t)

    # Sanity Tests
    def test_build_transform_sanity_list(self):
        cfg = [{"name": "dummy", "params": {}}, {"name": "dummy", "params": {}}]
        t = build_transform(cfg)
        assert callable(t)

    def test_build_transform_sanity_dict_views(self):
        cfg = {"views": [[{"name": "dummy", "params": {}}]]}
        t = build_transform(cfg)
        assert callable(t)

    def test_build_transform_sanity_dict_components(self):
        cfg = [{"name": "dummy", "params": {}}]
        t = build_transform(cfg)
        assert callable(t)

    # Unit Tests
    def test_build_transform_unit_randomapply(self, monkeypatch):
        monkeypatch.setattr(
            "src.refrakt_core.registry.transform_registry.get_transform",
            lambda name, *args, **kwargs: DummyTransform(**kwargs),
        )
        cfg = [
            {
                "name": "RandomApply",
                "params": {"transforms": [{"name": "dummy", "params": {}}], "p": 1.0},
            }
        ]
        t = build_transform(cfg)
        assert callable(t)

    def test_build_transform_unit_pairedtransform(self, monkeypatch):
        monkeypatch.setattr(
            "src.refrakt_core.registry.transform_registry.get_transform",
            lambda name, *args, **kwargs: DummyTransform(**kwargs),
        )

        # Mock PairedTransform to accept any kwargs
        class MockPairedTransform:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def __call__(self, x):
                return x

        monkeypatch.setattr(
            "src.refrakt_core.transforms.PairedTransform", MockPairedTransform
        )
        cfg = [
            {
                "name": "PairedTransform",
                "params": {"transforms": [{"name": "dummy", "params": {}}]},
            }
        ]
        t = build_transform(cfg)
        assert callable(t)

    def test_build_transform_unit_invalid_dict_format(self):
        cfg = {"foo": "bar"}
        with pytest.raises(ValueError):
            build_transform(cfg)

    def test_build_transform_unit_empty_list(self):
        cfg = []
        t = build_transform(cfg)
        assert callable(t)
        assert t(1) == 1

    def test_build_transform_unit_none(self):
        t = build_transform([])
        assert callable(t)
        assert t(1) == 1
