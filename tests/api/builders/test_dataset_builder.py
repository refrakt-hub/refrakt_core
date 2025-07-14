import pytest
from omegaconf import OmegaConf

from refrakt_core.api.builders.dataset_builder import build_dataset
from refrakt_core.registry import dataset_registry as reg
from refrakt_core.registry import transform_registry as treg


class DummyDataset:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.data = list(range(10))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DummyWrapper:
    def __init__(self, base, transform=None):
        self.base = base
        self.transform = transform
        self.wrapped = True

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        return self.transform(item) if self.transform else item


class DummyTransform:
    def __init__(self, **kwargs):
        pass

    def __call__(self, x):
        return x + 1


@pytest.fixture(autouse=True)
def patch_dataset_registry(monkeypatch):
    reg.DATASET_REGISTRY["dummy"] = DummyDataset
    reg.DATASET_REGISTRY["wrapper"] = DummyWrapper
    monkeypatch.setattr(
        "refrakt_core.registry.dataset_registry.get_dataset",
        lambda name, **params: DummyDataset(**params),
    )
    yield
    reg.DATASET_REGISTRY.pop("dummy", None)
    reg.DATASET_REGISTRY.pop("wrapper", None)


@pytest.fixture
def patch_transform_registry(monkeypatch):
    treg.TRANSFORM_REGISTRY["dummy_transform"] = DummyTransform
    treg.TRANSFORM_REGISTRY["dummy"] = DummyTransform
    monkeypatch.setattr(
        "refrakt_core.registry.transform_registry.get_transform",
        lambda name, *args, **kwargs: DummyTransform(**kwargs),
    )
    monkeypatch.setattr(
        "refrakt_core.api.builders.transform_builder.build_transform",
        lambda cfg: lambda x: x,
    )
    yield
    treg.TRANSFORM_REGISTRY.pop("dummy_transform", None)
    treg.TRANSFORM_REGISTRY.pop("dummy", None)


@pytest.fixture
def base_cfg():
    return OmegaConf.create({"name": "dummy", "params": {"foo": 1, "bar": 2}})


class TestDatasetBuilder:
    # Smoke Tests
    def test_build_dataset_smoke(self, base_cfg):
        ds = build_dataset(base_cfg)
        assert hasattr(ds, "__len__") and hasattr(ds, "__getitem__")

    def test_build_dataset_with_wrapper_smoke(self, base_cfg, monkeypatch):
        base_cfg["wrapper"] = "wrapper"
        # Patch at the builder's import location
        monkeypatch.setitem(
            __import__(
                "refrakt_core.api.builders.dataset_builder",
                fromlist=["DATASET_REGISTRY"],
            ).__dict__["DATASET_REGISTRY"],
            "wrapper",
            DummyWrapper,
        )
        monkeypatch.setattr(
            "refrakt_core.api.builders.dataset_builder.get_dataset",
            lambda name, **params: DummyDataset(**params),
        )
        ds = build_dataset(base_cfg)
        assert hasattr(ds, "wrapped")
        assert isinstance(ds, DummyWrapper)

    def test_build_dataset_sanity_params_dict(self, base_cfg, monkeypatch):
        # Patch get_dataset at the builder's import location
        monkeypatch.setattr(
            "refrakt_core.api.builders.dataset_builder.get_dataset",
            lambda name, **params: DummyDataset(**params),
        )
        ds = build_dataset(base_cfg)
        assert hasattr(ds, "kwargs")
        assert ds.kwargs["foo"] == 1

    # Unit Tests
    def test_build_dataset_unit_missing_name(self, base_cfg):
        base_cfg["name"] = None
        with pytest.raises(TypeError):
            build_dataset(base_cfg)

    def test_build_dataset_unit_params_not_dict(self, base_cfg):
        base_cfg["params"] = 123
        with pytest.raises(TypeError):
            build_dataset(base_cfg)

    def test_build_dataset_unit_wrapper_not_found(self, base_cfg):
        base_cfg["wrapper"] = "notfound"
        with pytest.raises(ValueError):
            build_dataset(base_cfg)

    def test_build_dataset_unit_params_missing(self, monkeypatch):
        cfg = OmegaConf.create({"name": "dummy"})
        monkeypatch.setattr(
            "refrakt_core.registry.dataset_registry.get_dataset",
            lambda name, **params: DummyDataset(**params),
        )
        ds = build_dataset(cfg)
        assert hasattr(ds, "__len__") and hasattr(ds, "__getitem__")

    def test_build_dataset_unit_transform_none(self, base_cfg, monkeypatch):
        base_cfg["transform"] = None
        ds = build_dataset(base_cfg)
        assert hasattr(ds, "__len__") and hasattr(ds, "__getitem__")

    def test_build_dataset_unit_wrapper_type_error(self, base_cfg):
        base_cfg["wrapper"] = 123
        with pytest.raises(TypeError):
            build_dataset(base_cfg)
