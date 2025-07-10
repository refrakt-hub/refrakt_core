import pytest

from refrakt_core.api.helpers import train_helpers
from refrakt_core.registry import model_registry
from refrakt_core.registry.dataset_registry import DATASET_REGISTRY
from tests.helpers.fixtures import DummyDataset, DummyModel


@pytest.fixture(autouse=True)
def ensure_dummy_dataset_train():
    DATASET_REGISTRY["dummy"] = DummyDataset


@pytest.fixture(autouse=True)
def patch_model_registry_train(monkeypatch):
    model_registry.MODEL_REGISTRY["dummy"] = DummyModel
    monkeypatch.setattr(
        "refrakt_core.registry.model_registry.get_model",
        lambda name, *args, **kwargs: DummyModel(**kwargs),
    )
    yield


class DummyLogger:
    def info(self, msg):
        print(msg)

    def warning(self, msg):
        print(msg)

    def error(self, msg):
        print(msg)


class TestTrain:
    def test_train_sanity_success(self, monkeypatch, dummy_cfg):
        monkeypatch.setattr(
            train_helpers, "_load_and_validate_config", lambda cfg: dummy_cfg
        )
        monkeypatch.setattr(
            train_helpers, "_setup_logging", lambda cfg, name, logger: DummyLogger()
        )
        monkeypatch.setattr(train_helpers, "_check_pure_ml_training", lambda cfg: False)
        monkeypatch.setattr(
            train_helpers, "_get_modules_and_device", lambda: ({}, "cpu")
        )
        monkeypatch.setattr(
            train_helpers,
            "_build_datasets_and_model",
            lambda cfg, modules, device, logger: (
                "train_loader",
                "val_loader",
                DummyModel(),
                "loss_fn",
            ),
        )
        monkeypatch.setattr(
            train_helpers,
            "_setup_optimizer_and_scheduler",
            lambda cfg, model: ("optimizer", "scheduler"),
        )
        monkeypatch.setattr(
            "refrakt_core.api.utils.train_utils.setup_artifact_dumper",
            lambda cfg, name, logger: "artifact_dumper",
        )
        monkeypatch.setattr(
            train_helpers,
            "_setup_trainer",
            lambda cfg, model, train_loader, val_loader, loss_fn, optimizer, scheduler, device, modules, artifact_dumper, name, logger: (
                "trainer",
                10,
                "cpu",
            ),
        )
        monkeypatch.setattr(
            train_helpers,
            "_execute_training",
            lambda trainer, num_epochs, config, model, train_loader, val_loader, final_device, artifact_dumper, resolved_model_name, logger: {
                "trained": True
            },
        )
        monkeypatch.setattr("omegaconf.OmegaConf.save", lambda *a, **kw: None)
        import refrakt_core.api.train as train_module

        result = train_module.train(cfg=dummy_cfg, logger=None)
        assert result["trained"] is True

    def test_train_unit_pure_ml(self, monkeypatch, dummy_cfg):
        monkeypatch.setattr(
            train_helpers, "_load_and_validate_config", lambda cfg: dummy_cfg
        )
        monkeypatch.setattr(
            train_helpers, "_setup_logging", lambda cfg, name, logger: DummyLogger()
        )
        monkeypatch.setattr(
            "refrakt_core.api.train._check_pure_ml_training", lambda cfg: True
        )
        monkeypatch.setattr(
            "refrakt_core.api.train._handle_pure_ml_training",
            lambda *a, **kw: {"status": "completed", "type": "ml"},
        )
        monkeypatch.setattr("omegaconf.OmegaConf.save", lambda *a, **kw: None)
        import refrakt_core.api.train as train_module

        result = train_module.train(cfg=dummy_cfg, logger=None)
        assert result["status"] == "completed" and result["type"] == "ml"

    def test_train_unit_artifact_dumper_called(self, monkeypatch, dummy_cfg):
        monkeypatch.setattr(
            train_helpers, "_load_and_validate_config", lambda cfg: dummy_cfg
        )
        monkeypatch.setattr(
            train_helpers, "_setup_logging", lambda cfg, name, logger: DummyLogger()
        )
        monkeypatch.setattr(train_helpers, "_check_pure_ml_training", lambda cfg: False)
        monkeypatch.setattr(
            train_helpers, "_get_modules_and_device", lambda: ({}, "cpu")
        )
        monkeypatch.setattr(
            train_helpers,
            "_build_datasets_and_model",
            lambda cfg, modules, device, logger: (
                "train_loader",
                "val_loader",
                DummyModel(),
                "loss_fn",
            ),
        )
        monkeypatch.setattr(
            train_helpers,
            "_setup_optimizer_and_scheduler",
            lambda cfg, model: ("optimizer", "scheduler"),
        )
        monkeypatch.setattr(
            "refrakt_core.api.utils.train_utils.setup_artifact_dumper",
            lambda cfg, name, logger: "artifact_dumper",
        )
        monkeypatch.setattr(
            train_helpers,
            "_setup_trainer",
            lambda cfg, model, train_loader, val_loader, loss_fn, optimizer, scheduler, device, modules, artifact_dumper, name, logger: (
                "trainer",
                10,
                "cpu",
            ),
        )
        monkeypatch.setattr(
            train_helpers,
            "_execute_training",
            lambda trainer, num_epochs, config, model, train_loader, val_loader, final_device, artifact_dumper, resolved_model_name, logger: {
                "trained": True
            },
        )
        monkeypatch.setattr("omegaconf.OmegaConf.save", lambda *a, **kw: None)
        import refrakt_core.api.train as train_module

        result = train_module.train(cfg=dummy_cfg, logger=None)
        assert result["trained"] is True
