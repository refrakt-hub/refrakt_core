import importlib

import pytest

import refrakt_core.api.inference as inference_module
from refrakt_core.api.helpers import inference_helpers
from refrakt_core.registry.dataset_registry import DATASET_REGISTRY
from tests.helpers.fixtures import DummyDataset


@pytest.fixture(autouse=True)
def ensure_dummy_dataset_inference():
    DATASET_REGISTRY["dummy"] = DummyDataset


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


class TestInference:
    def test_inference_import_smoke(self):
        assert callable(inference_module.inference)

    def test_inference_module_importable(self):
        assert callable(inference_module.inference)

    def test_inference_sanity_success(self, monkeypatch, dummy_cfg):
        monkeypatch.setattr(
            inference_module, "_load_and_validate_config", lambda cfg: dummy_cfg
        )
        monkeypatch.setattr(
            inference_module, "_setup_logging", lambda cfg, name, logger: DummyLogger()
        )
        monkeypatch.setattr(
            inference_module, "_check_pure_ml_inference", lambda cfg: False
        )
        monkeypatch.setattr(inference_module, "_setup_device", lambda: "cpu")
        monkeypatch.setattr(
            inference_module,
            "_load_model_and_setup",
            lambda cfg, device, model_path, logger: (
                print("[DEBUG] Dummy _load_model_and_setup called"),
                ("model", {}),
            )[1],
        )
        monkeypatch.setattr(
            inference_module, "_setup_data_loader", lambda cfg, data, logger: [1, 2, 3]
        )
        result = inference_module.inference(cfg=dummy_cfg, model_path="dummy.pth")
        assert "model" in result and "results" in result and "config" in result

    def test_inference_unit_pure_ml(self, monkeypatch, dummy_cfg):
        monkeypatch.setattr(
            inference_module, "_load_and_validate_config", lambda cfg: dummy_cfg
        )
        monkeypatch.setattr(
            inference_module, "_setup_logging", lambda cfg, name, logger: DummyLogger()
        )
        monkeypatch.setattr(
            inference_module, "_check_pure_ml_inference", lambda cfg: True
        )
        monkeypatch.setattr(
            inference_module,
            "handle_pure_ml_inference",
            lambda cfg, name, logger: {"ml": True},
        )
        result = inference_module.inference(cfg=dummy_cfg, model_path="dummy.pth")
        assert result == {"ml": True}

    def test_inference_unit_error_handling(self, monkeypatch, dummy_cfg):
        monkeypatch.setattr(
            inference_helpers,
            "_load_and_validate_config",
            lambda cfg: (_ for _ in ()).throw(Exception("fail")),
        )
        logger = DummyLogger()
        monkeypatch.setattr(
            inference_helpers, "_setup_logging", lambda cfg, name, logger_arg: logger
        )
        with pytest.raises(SystemExit):
            inference_module.inference(cfg=dummy_cfg, model_path="dummy.pth")
        assert True
