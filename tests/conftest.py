import pytest

from refrakt_core import datasets  # Also register under non-src namespace
from refrakt_core import transforms  # Also register under non-src namespace
from refrakt_core.registry.dataset_registry import DATASET_REGISTRY
from refrakt_core.registry.transform_registry import (
    TRANSFORM_REGISTRY,
    _import_transforms,
)
from refrakt_core.registry import transform_registry as reg


class DummyTransform:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def __call__(self, x):
        return x + 1


def print_registries():
    # Ensure transforms are imported before checking registry
    _import_transforms()
    print("TRANSFORM_REGISTRY:", list(TRANSFORM_REGISTRY.keys()))
    print("DATASET_REGISTRY:", list(DATASET_REGISTRY.keys()))


@pytest.fixture(scope="session", autouse=True)
def show_registries():
    print_registries()


@pytest.fixture(autouse=False)
def patch_transform_registry(monkeypatch):
    reg.TRANSFORM_REGISTRY["dummy"] = DummyTransform
    reg.TRANSFORM_REGISTRY["dummy_transform"] = DummyTransform
    yield
    reg.TRANSFORM_REGISTRY.pop("dummy", None)
    reg.TRANSFORM_REGISTRY.pop("dummy_transform", None)