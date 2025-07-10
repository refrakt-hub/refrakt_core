import pytest

import refrakt_core.datasets  # Also register under non-src namespace
import refrakt_core.transforms  # Also register under non-src namespace
from refrakt_core.registry.dataset_registry import DATASET_REGISTRY
from refrakt_core.registry.loss_registry import LOSS_REGISTRY
from refrakt_core.registry.trainer_registry import TRAINER_REGISTRY
from tests.helpers.fixtures import (  # Make dummy_cfg available globally
    DummyDataset,
    dummy_cfg,
)


# Always register 'dummy' dataset for tests (autouse fixture)
@pytest.fixture(autouse=True)
def ensure_dummy_dataset():
    DATASET_REGISTRY["dummy"] = DummyDataset


# Always register 'dummy_loss' for tests (autouse fixture)
@pytest.fixture(autouse=True)
def ensure_dummy_loss():
    class DummyLoss:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return 0.0

    LOSS_REGISTRY["dummy_loss"] = DummyLoss


# Always register 'dummy_trainer' for tests (autouse fixture)
@pytest.fixture(autouse=True)
def ensure_dummy_trainer():
    class DummyTrainer:
        def __init__(self, *args, **kwargs):
            self.global_step = 0

        def train(self, num_epochs=None, *args, **kwargs):
            return {"trained": True}

        def evaluate(self):
            return 1.0

    TRAINER_REGISTRY["dummy_trainer"] = DummyTrainer


@pytest.fixture
def cfg(dummy_cfg):
    return dummy_cfg


def print_registries():
    from refrakt_core.registry.dataset_registry import DATASET_REGISTRY
    from refrakt_core.registry.transform_registry import (
        TRANSFORM_REGISTRY,
        _import_transforms,
    )

    # Ensure transforms are imported before checking registry
    _import_transforms()
    print("TRANSFORM_REGISTRY:", list(TRANSFORM_REGISTRY.keys()))
    print("DATASET_REGISTRY:", list(DATASET_REGISTRY.keys()))


@pytest.fixture(scope="session", autouse=True)
def show_registries():
    print_registries()
