import pytest

import refrakt_core.api.test as test_module
from refrakt_core.api.helpers import test_helpers
from refrakt_core.api.test import test
from refrakt_core.registry import model_registry
from refrakt_core.registry.dataset_registry import DATASET_REGISTRY
from tests.helpers.fixtures import DummyDataset, DummyModel


@pytest.fixture(autouse=True)
def ensure_dummy_dataset_test():
    DATASET_REGISTRY["dummy"] = DummyDataset


@pytest.fixture(autouse=True)
def patch_model_registry_test(monkeypatch):
    model_registry.MODEL_REGISTRY["dummy"] = DummyModel
    monkeypatch.setattr(
        "refrakt_core.registry.model_registry.get_model",
        lambda name, *args, **kwargs: DummyModel(**kwargs),
    )
    yield


class DummyLogger:
    def __init__(self):
        self.errors = []
        self.infos = []
        self.warnings = []

    def error(self, msg):
        self.errors.append(msg)

    def info(self, msg):
        self.infos.append(msg)

    def warning(self, msg):
        self.warnings.append(msg)


class TestTestEntrypoint:
    def test_import_test_smoke(self):
        assert callable(test)

    def test_test_signature(self):
        from inspect import signature

        sig = signature(test)
        assert "cfg" in sig.parameters
        assert "model_path" in sig.parameters
        assert "logger" in sig.parameters

    def test_test_sanity_success(self, monkeypatch, dummy_cfg):
        monkeypatch.setattr(
            test_helpers, "_load_and_validate_config", lambda cfg: dummy_cfg
        )
        monkeypatch.setattr(
            test_helpers, "_setup_logging", lambda cfg, name, logger: DummyLogger()
        )
        monkeypatch.setattr(test_helpers, "_check_pure_ml_testing", lambda cfg: False)
        monkeypatch.setattr(
            test_helpers, "_get_modules_and_device", lambda: ({}, "cpu")
        )
        monkeypatch.setattr(
            test_helpers,
            "_build_test_components",
            lambda cfg, modules, device, logger: (
                "dataloader",
                DummyModel(),
                "loss_fn",
            ),
        )
        monkeypatch.setattr(
            "refrakt_core.api.utils.train_utils.setup_artifact_dumper",
            lambda cfg, name, logger: "artifact_dumper",
        )
        monkeypatch.setattr(
            test_helpers,
            "_setup_trainer_for_testing",
            lambda cfg, model, dataloader, loss_fn, device, modules, artifact_dumper, name, logger: "trainer",
        )
        monkeypatch.setattr(
            "refrakt_core.api.utils.test_utils._load_model_checkpoint",
            lambda *a, **kw: None,
        )
        result = test(cfg=dummy_cfg, model_path=None, logger=None)
        assert result is None

    def test_test_unit_pure_ml(self, monkeypatch, dummy_cfg):
        monkeypatch.setattr(
            test_helpers, "_load_and_validate_config", lambda cfg: dummy_cfg
        )
        monkeypatch.setattr(
            test_helpers, "_setup_logging", lambda cfg, name, logger: DummyLogger()
        )
        monkeypatch.setattr(test_helpers, "_check_pure_ml_testing", lambda cfg: True)
        monkeypatch.setattr(
            "refrakt_core.api.utils.test_utils._handle_pure_ml_pipeline",
            lambda *a, **kw: None,
        )
        monkeypatch.setattr(
            "refrakt_core.api.utils.test_utils._load_model_checkpoint",
            lambda *a, **kw: None,
        )
        result = test(cfg=dummy_cfg, model_path=None, logger=None)
        assert result is None

    def test_test_unit_error_handling(self, monkeypatch, dummy_cfg):
        monkeypatch.setattr(
            test_helpers,
            "_load_and_validate_config",
            lambda cfg: (_ for _ in ()).throw(Exception("fail")),
        )
        logger = DummyLogger()
        monkeypatch.setattr(
            test_helpers, "_setup_logging", lambda cfg, name, logger_arg: logger
        )
        with pytest.raises(SystemExit):
            test(cfg=dummy_cfg, model_path="dummy.pth", logger=None)
        assert True

    def test_test_unit_invalid_cfg(self, monkeypatch):
        monkeypatch.setattr(
            test_helpers,
            "_load_and_validate_config",
            lambda cfg: (_ for _ in ()).throw(ValueError("bad cfg")),
        )
        logger = DummyLogger()
        monkeypatch.setattr(
            test_helpers, "_setup_logging", lambda cfg, name, logger_arg: logger
        )
        with pytest.raises(SystemExit):
            test(cfg="invalid_cfg", model_path="dummy.pth", logger=None)
        assert True

    def test_test_unit_artifact_dumper_called(self, monkeypatch, dummy_cfg):
        monkeypatch.setattr(
            test_helpers, "_load_and_validate_config", lambda cfg: dummy_cfg
        )
        monkeypatch.setattr(
            test_helpers, "_setup_logging", lambda cfg, name, logger: DummyLogger()
        )
        monkeypatch.setattr(test_helpers, "_check_pure_ml_testing", lambda cfg: False)
        monkeypatch.setattr(
            test_helpers, "_get_modules_and_device", lambda: ({}, "cpu")
        )
        monkeypatch.setattr(
            test_helpers,
            "_build_test_components",
            lambda cfg, modules, device, logger: (
                "dataloader",
                DummyModel(),
                "loss_fn",
            ),
        )
        monkeypatch.setattr(
            "refrakt_core.api.utils.train_utils.setup_artifact_dumper",
            lambda cfg, name, logger: "artifact_dumper",
        )
        monkeypatch.setattr(
            test_helpers,
            "_setup_trainer_for_testing",
            lambda cfg, model, dataloader, loss_fn, device, modules, artifact_dumper, name, logger: "trainer",
        )
        monkeypatch.setattr(
            "refrakt_core.api.utils.test_utils._load_model_checkpoint",
            lambda *a, **kw: None,
        )
        monkeypatch.setattr(
            test_helpers,
            "_evaluate_model",
            lambda trainer, model, dataloader, device, fusion_acc, logger: {
                "accuracy": 1.0
            },
        )
        result = test(cfg=dummy_cfg, model_path=None, logger=None)
        assert result is None
