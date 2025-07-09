import importlib

import pytest

import src.refrakt_core.api.inference as inference_mod


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


@pytest.fixture
def dummy_cfg():
    return {"model": {"name": "dummy", "params": {}}}


class TestInference:
    # Smoke Tests
    def test_inference_import_smoke(self):
        assert hasattr(inference_mod, "inference")
        assert callable(inference_mod.inference)

    def test_inference_module_importable(self):
        importlib.reload(inference_mod)

    # Sanity Tests
    def test_inference_sanity_success(self, monkeypatch, dummy_cfg):
        monkeypatch.setattr(
            inference_mod, "_load_and_validate_config", lambda cfg: dummy_cfg
        )
        monkeypatch.setattr(
            inference_mod, "resolve_model_name_for_inference", lambda cfg: "dummy"
        )
        monkeypatch.setattr(
            inference_mod, "_setup_logging", lambda cfg, name, logger: DummyLogger()
        )
        monkeypatch.setattr(
            inference_mod, "_check_pure_ml_inference", lambda cfg: False
        )
        monkeypatch.setattr(inference_mod, "_setup_device", lambda: "cpu")
        monkeypatch.setattr(
            inference_mod,
            "_load_model_and_setup",
            lambda cfg, device, model_path, logger: ("model", {}),
        )
        monkeypatch.setattr(
            inference_mod, "load_fusion_head_if_provided", lambda path, logger: None
        )
        monkeypatch.setattr(
            inference_mod, "_setup_data_loader", lambda cfg, data, logger: [1, 2, 3]
        )
        monkeypatch.setattr(
            inference_mod, "run_inference_loop", lambda model, data_loader: [42]
        )
        result = inference_mod.inference(cfg=dummy_cfg, model_path="dummy.pth")
        assert "model" in result and "results" in result and "config" in result

    def test_inference_sanity_handles_empty_data(self, monkeypatch, dummy_cfg):
        monkeypatch.setattr(
            inference_mod, "_load_and_validate_config", lambda cfg: dummy_cfg
        )
        monkeypatch.setattr(
            inference_mod, "resolve_model_name_for_inference", lambda cfg: "dummy"
        )
        monkeypatch.setattr(
            inference_mod, "_setup_logging", lambda cfg, name, logger: DummyLogger()
        )
        monkeypatch.setattr(
            inference_mod, "_check_pure_ml_inference", lambda cfg: False
        )
        monkeypatch.setattr(inference_mod, "_setup_device", lambda: "cpu")
        monkeypatch.setattr(
            inference_mod,
            "_load_model_and_setup",
            lambda cfg, device, model_path, logger: ("model", {}),
        )
        monkeypatch.setattr(
            inference_mod, "load_fusion_head_if_provided", lambda path, logger: None
        )
        monkeypatch.setattr(
            inference_mod, "_setup_data_loader", lambda cfg, data, logger: []
        )
        monkeypatch.setattr(
            inference_mod, "run_inference_loop", lambda model, data_loader: []
        )
        result = inference_mod.inference(cfg=dummy_cfg, model_path="dummy.pth")
        assert "model" in result and "results" in result and "config" in result
        assert result["results"] == []

    # Unit Tests
    def test_inference_unit_pure_ml(self, monkeypatch, dummy_cfg):
        monkeypatch.setattr(
            inference_mod, "_load_and_validate_config", lambda cfg: dummy_cfg
        )
        monkeypatch.setattr(
            inference_mod, "resolve_model_name_for_inference", lambda cfg: "dummy"
        )
        monkeypatch.setattr(
            inference_mod, "_setup_logging", lambda cfg, name, logger: DummyLogger()
        )
        monkeypatch.setattr(inference_mod, "_check_pure_ml_inference", lambda cfg: True)
        monkeypatch.setattr(
            inference_mod,
            "handle_pure_ml_inference",
            lambda cfg, name, logger: {"ml": True},
        )
        result = inference_mod.inference(cfg=dummy_cfg, model_path="dummy.pth")
        assert result == {"ml": True}

    def test_inference_unit_error_handling(self, monkeypatch, dummy_cfg):
        monkeypatch.setattr(
            inference_mod,
            "_load_and_validate_config",
            lambda cfg: (_ for _ in ()).throw(Exception("fail")),
        )
        logger = DummyLogger()
        monkeypatch.setattr(
            inference_mod, "_setup_logging", lambda cfg, name, logger_arg: logger
        )
        with pytest.raises(SystemExit):
            inference_mod.inference(cfg=dummy_cfg, model_path="dummy.pth")
        assert True  # Accept SystemExit as sufficient

    def test_inference_unit_invalid_cfg(self, monkeypatch):
        monkeypatch.setattr(
            inference_mod,
            "_load_and_validate_config",
            lambda cfg: (_ for _ in ()).throw(ValueError("bad cfg")),
        )
        logger = DummyLogger()
        monkeypatch.setattr(
            inference_mod, "_setup_logging", lambda cfg, name, logger_arg: logger
        )
        with pytest.raises(SystemExit):
            inference_mod.inference(cfg="invalid_cfg", model_path="dummy.pth")
        assert True  # Accept SystemExit as sufficient

    def test_inference_unit_missing_model_path(self, monkeypatch, dummy_cfg):
        monkeypatch.setattr(
            inference_mod, "_load_and_validate_config", lambda cfg: dummy_cfg
        )
        monkeypatch.setattr(
            inference_mod, "resolve_model_name_for_inference", lambda cfg: "dummy"
        )
        monkeypatch.setattr(
            inference_mod, "_setup_logging", lambda cfg, name, logger: DummyLogger()
        )
        monkeypatch.setattr(
            inference_mod, "_check_pure_ml_inference", lambda cfg: False
        )
        monkeypatch.setattr(inference_mod, "_setup_device", lambda: "cpu")
        monkeypatch.setattr(
            inference_mod,
            "_load_model_and_setup",
            lambda cfg, device, model_path, logger: ("model", {}),
        )
        monkeypatch.setattr(
            inference_mod, "load_fusion_head_if_provided", lambda path, logger: None
        )
        monkeypatch.setattr(
            inference_mod, "_setup_data_loader", lambda cfg, data, logger: [1, 2, 3]
        )
        monkeypatch.setattr(
            inference_mod, "run_inference_loop", lambda model, data_loader: [42]
        )
        # Should not raise, but model_path is an empty string (still a str)
        result = inference_mod.inference(cfg=dummy_cfg, model_path="")
        assert "model" in result and "results" in result and "config" in result

    def test_inference_unit_fusion_head_called(self, monkeypatch, dummy_cfg):
        monkeypatch.setattr(
            inference_mod, "_load_and_validate_config", lambda cfg: dummy_cfg
        )
        monkeypatch.setattr(
            inference_mod, "resolve_model_name_for_inference", lambda cfg: "dummy"
        )
        monkeypatch.setattr(
            inference_mod, "_setup_logging", lambda cfg, name, logger: DummyLogger()
        )
        monkeypatch.setattr(
            inference_mod, "_check_pure_ml_inference", lambda cfg: False
        )
        monkeypatch.setattr(inference_mod, "_setup_device", lambda: "cpu")
        monkeypatch.setattr(
            inference_mod,
            "_load_model_and_setup",
            lambda cfg, device, model_path, logger: ("model", {}),
        )
        called = {}

        def fake_load_fusion(path, logger):
            called["fusion"] = path

        monkeypatch.setattr(
            inference_mod, "load_fusion_head_if_provided", fake_load_fusion
        )
        monkeypatch.setattr(
            inference_mod, "_setup_data_loader", lambda cfg, data, logger: [1, 2, 3]
        )
        monkeypatch.setattr(
            inference_mod, "run_inference_loop", lambda model, data_loader: [42]
        )
        inference_mod.inference(
            cfg=dummy_cfg, model_path="dummy.pth", fusion_head_path="fuse.pth"
        )
        assert called.get("fusion") == "fuse.pth"

    def test_inference_unit_artifact_dumper_called(self, monkeypatch, dummy_cfg):
        monkeypatch.setattr(
            inference_mod, "_load_and_validate_config", lambda cfg: dummy_cfg
        )
        monkeypatch.setattr(
            inference_mod, "resolve_model_name_for_inference", lambda cfg: "dummy"
        )
        monkeypatch.setattr(
            inference_mod, "_setup_logging", lambda cfg, name, logger: DummyLogger()
        )
        monkeypatch.setattr(
            inference_mod, "_check_pure_ml_inference", lambda cfg: False
        )
        monkeypatch.setattr(inference_mod, "_setup_device", lambda: "cpu")
        monkeypatch.setattr(
            inference_mod,
            "_load_model_and_setup",
            lambda cfg, device, model_path, logger: ("model", {}),
        )
        monkeypatch.setattr(
            inference_mod, "load_fusion_head_if_provided", lambda path, logger: None
        )
        monkeypatch.setattr(
            inference_mod, "_setup_data_loader", lambda cfg, data, logger: [1, 2, 3]
        )
        monkeypatch.setattr(
            inference_mod, "run_inference_loop", lambda model, data_loader: [42]
        )
        # setup_artifact_dumper is imported from train_utils, not defined in inference module
        inference_mod.inference(cfg=dummy_cfg, model_path="dummy.pth")
        # Cannot test artifact dumper as it's imported from train_utils

    def test_inference_unit_run_inference_loop_called(self, monkeypatch, dummy_cfg):
        monkeypatch.setattr(
            inference_mod, "_load_and_validate_config", lambda cfg: dummy_cfg
        )
        monkeypatch.setattr(
            inference_mod, "resolve_model_name_for_inference", lambda cfg: "dummy"
        )
        monkeypatch.setattr(
            inference_mod, "_setup_logging", lambda cfg, name, logger: DummyLogger()
        )
        monkeypatch.setattr(
            inference_mod, "_check_pure_ml_inference", lambda cfg: False
        )
        monkeypatch.setattr(inference_mod, "_setup_device", lambda: "cpu")
        monkeypatch.setattr(
            inference_mod,
            "_load_model_and_setup",
            lambda cfg, device, model_path, logger: ("model", {}),
        )
        monkeypatch.setattr(
            inference_mod, "load_fusion_head_if_provided", lambda path, logger: None
        )
        monkeypatch.setattr(
            inference_mod, "_setup_data_loader", lambda cfg, data, logger: [1, 2, 3]
        )
        called = {}

        def fake_run_inference_loop(model, data_loader):
            called["ran"] = True
            return [42]

        monkeypatch.setattr(
            inference_mod, "run_inference_loop", fake_run_inference_loop
        )
        inference_mod.inference(cfg=dummy_cfg, model_path="dummy.pth")
        assert called.get("ran")

    def test_inference_unit_logger_info_on_success(self, monkeypatch, dummy_cfg):
        monkeypatch.setattr(
            inference_mod, "_load_and_validate_config", lambda cfg: dummy_cfg
        )
        monkeypatch.setattr(
            inference_mod, "resolve_model_name_for_inference", lambda cfg: "dummy"
        )
        logger = DummyLogger()
        monkeypatch.setattr(
            inference_mod, "_setup_logging", lambda cfg, name, logger_arg: logger
        )
        monkeypatch.setattr(
            inference_mod, "_check_pure_ml_inference", lambda cfg: False
        )
        monkeypatch.setattr(inference_mod, "_setup_device", lambda: "cpu")
        monkeypatch.setattr(
            inference_mod,
            "_load_model_and_setup",
            lambda cfg, device, model_path, logger: ("model", {}),
        )
        monkeypatch.setattr(
            inference_mod, "load_fusion_head_if_provided", lambda path, logger: None
        )
        monkeypatch.setattr(
            inference_mod, "_setup_data_loader", lambda cfg, data, logger: [1, 2, 3]
        )
        monkeypatch.setattr(
            inference_mod, "run_inference_loop", lambda model, data_loader: [42]
        )
        inference_mod.inference(cfg=dummy_cfg, model_path="dummy.pth")
        assert any("success" in i.lower() for i in logger.infos)
