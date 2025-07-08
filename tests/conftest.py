import src.refrakt_core.transforms      # Also register under non-src namespace
import src.refrakt_core.datasets        # Also register under non-src namespace

import pytest

def print_registries():
    from src.refrakt_core.registry.transform_registry import TRANSFORM_REGISTRY, _import_transforms
    from src.refrakt_core.registry.dataset_registry import DATASET_REGISTRY
    # Ensure transforms are imported before checking registry
    _import_transforms()
    print('TRANSFORM_REGISTRY:', list(TRANSFORM_REGISTRY.keys()))
    print('DATASET_REGISTRY:', list(DATASET_REGISTRY.keys()))

@pytest.fixture(scope='session', autouse=True)
def show_registries():
    print_registries()