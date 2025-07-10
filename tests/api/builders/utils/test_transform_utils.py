import importlib

import pytest
from omegaconf import ListConfig

import refrakt_core.api.builders.utils.transform_utils as transform_utils
import refrakt_core.registry.transform_registry as reg


@pytest.fixture
def patch_transform_registry():
    reg.TRANSFORM_REGISTRY["dummy"] = DummyTransform
    yield
    reg.TRANSFORM_REGISTRY.pop("dummy", None)


class DummyTransform:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, x):
        return x + 1


def dummy_get_transform(name, *args, **kwargs):
    return DummyTransform(**kwargs)


class TestTransformUtils:
    # Smoke Tests
    def test_import_transform_utils(self):
        importlib.reload(transform_utils)

    def test_transform_utils_has_any_symbol(self):
        symbols = [s for s in dir(transform_utils) if not s.startswith("__")]
        assert symbols

    # Sanity Tests
    def test_resolve_transform_sequence_list(self):
        seq = [{"name": "dummy", "params": {}}]
        out = transform_utils._resolve_transform_sequence(seq)
        assert isinstance(out, list)

    def test_resolve_transform_sequence_dict_views(self):
        seq = {"views": [[{"name": "dummy", "params": {}}]]}
        out = transform_utils._resolve_transform_sequence(seq)
        assert isinstance(out, list)

    def test_resolve_transform_sequence_dict_components(self):
        seq = {"components": [{"name": "dummy", "params": {}}]}
        out = transform_utils._resolve_transform_sequence(seq)
        assert isinstance(out, list)

    # Unit Tests
    def test_resolve_transform_sequence_value_error(self):
        with pytest.raises(ValueError):
            transform_utils._resolve_transform_sequence({"foo": "bar"})

    def test_build_simple_transform(self, monkeypatch, patch_transform_registry):
        monkeypatch.setattr(
            "refrakt_core.registry.transform_registry.get_transform",
            dummy_get_transform,
        )
        t = transform_utils._build_simple_transform("dummy", {})
        assert callable(t)

    def test_build_nested_transform(self, monkeypatch):
        monkeypatch.setattr(
            "refrakt_core.registry.transform_registry.get_transform",
            dummy_get_transform,
        )
        t = transform_utils._build_nested_transform(
            "RandomApply",
            {"transforms": [{"name": "dummy", "params": {}}], "p": 1.0},
            lambda cfgs: [DummyTransform() for _ in cfgs],
        )
        assert callable(t)

    def test_build_transform_list(self, monkeypatch, patch_transform_registry):
        monkeypatch.setattr(
            "refrakt_core.registry.transform_registry.get_transform",
            dummy_get_transform,
        )
        seq = [{"name": "dummy", "params": {}}]
        out = transform_utils._build_transform_list(
            seq, lambda cfgs: [DummyTransform() for _ in cfgs]
        )
        assert isinstance(out, list)
        assert callable(out[0])

    def test_create_final_transform_single(self):
        t = DummyTransform()
        out = transform_utils._create_final_transform([t])
        assert callable(out)
        assert out(1) == 2

    def test_create_final_transform_multiple(self):
        t1 = DummyTransform()
        t2 = DummyTransform()
        out = transform_utils._create_final_transform([t1, t2])
        assert callable(out)
        assert out(1) == 3
