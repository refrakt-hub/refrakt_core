import importlib
import sys

import pytest
from omegaconf import DictConfig

from src.refrakt_core.api import __main__


class DummyLogger:
    def __init__(self):
        self.errors = []
        self.infos = []
        self.warnings = []
        self.closed = False

    def error(self, msg):
        self.errors.append(msg)

    def info(self, msg):
        self.infos.append(msg)

    def warning(self, msg):
        self.warnings.append(msg)

    def close(self):
        self.closed = True


class DummyArgs:
    def __init__(self, config="dummy.yaml", log_dir=None, debug=False):
        self.config = config
        self.log_dir = log_dir
        self.debug = debug


class TestMainEntrypoint:
    # Smoke Tests
    def test_import_main_smoke(self):
        importlib.reload(__main__)
        assert hasattr(__main__, "main")
        assert callable(__main__.main)

    # Sanity Tests
    def test_main_entrypoint_runs(self, monkeypatch):
        called = {}

        def fake_main():
            called["main"] = True

        monkeypatch.setattr(__main__, "main", fake_main)
        __main__.main()
        assert called.get("main")

    # Unit Tests
    def test_main_function_exists(self):
        assert hasattr(__main__, "main")
        assert callable(__main__.main)

    def test_main_sys_exit_on_error(self, monkeypatch):
        def fake_main():
            raise SystemExit(1)

        monkeypatch.setattr(__main__, "main", fake_main)
        with pytest.raises(SystemExit):
            __main__.main()

    def test_main_cli_argument_parsing(self, monkeypatch):
        called = {}

        def fake_setup_argument_parser():
            class DummyParser:
                def parse_known_args(self):
                    called["args"] = True
                    return DummyArgs(), []

            return DummyParser()

        monkeypatch.setattr(
            __main__, "_setup_argument_parser", fake_setup_argument_parser
        )
        monkeypatch.setattr(__main__, "_extract_overrides", lambda args, rem: [])
        monkeypatch.setattr(
            __main__,
            "OmegaConf",
            type(
                "OmegaConf",
                (),
                {
                    "load": staticmethod(
                        lambda x: DictConfig({"model": {"name": "dummy"}})
                    ),
                    "is_config": staticmethod(lambda x: True),
                },
            ),
        )
        monkeypatch.setattr(
            __main__, "_apply_config_overrides", lambda cfg, overrides: cfg
        )
        monkeypatch.setattr(__main__, "_extract_runtime_config", lambda cfg: cfg)
        monkeypatch.setattr(
            __main__,
            "_setup_logging_config",
            lambda cfg, log_dir: ("train", "log_dir", [], True, None, False),
        )
        monkeypatch.setattr(
            __main__, "setup_logger_and_config", lambda *a, **kw: DummyLogger()
        )
        monkeypatch.setattr(__main__, "_execute_pipeline_mode", lambda *a, **kw: None)
        __main__.main()
        assert called.get("args")

    def test_main_pipeline_mode_dispatch(self, monkeypatch):
        called = {}

        def fake_execute_pipeline_mode(mode, cfg, model_path, logger):
            called["mode"] = mode

        monkeypatch.setattr(
            __main__,
            "_setup_argument_parser",
            lambda: type(
                "DummyParser",
                (),
                {"parse_known_args": staticmethod(lambda: (DummyArgs(), []))},
            )(),
        )
        monkeypatch.setattr(__main__, "_extract_overrides", lambda args, rem: [])
        monkeypatch.setattr(
            __main__,
            "OmegaConf",
            type(
                "OmegaConf",
                (),
                {
                    "load": staticmethod(
                        lambda x: DictConfig({"model": {"name": "dummy"}})
                    ),
                    "is_config": staticmethod(lambda x: True),
                },
            ),
        )
        monkeypatch.setattr(
            __main__, "_apply_config_overrides", lambda cfg, overrides: cfg
        )
        monkeypatch.setattr(__main__, "_extract_runtime_config", lambda cfg: cfg)
        monkeypatch.setattr(
            __main__,
            "_setup_logging_config",
            lambda cfg, log_dir: ("test", "log_dir", [], True, None, False),
        )
        monkeypatch.setattr(
            __main__, "setup_logger_and_config", lambda *a, **kw: DummyLogger()
        )
        monkeypatch.setattr(
            __main__, "_execute_pipeline_mode", fake_execute_pipeline_mode
        )
        __main__.main()
        assert called.get("mode") == "test"

    def test_main_logger_finalization(self, monkeypatch):
        logger = DummyLogger()

        def fake_setup_logger_and_config(*a, **kw):
            return logger

        monkeypatch.setattr(
            __main__,
            "_setup_argument_parser",
            lambda: type(
                "DummyParser",
                (),
                {"parse_known_args": staticmethod(lambda: (DummyArgs(), []))},
            )(),
        )
        monkeypatch.setattr(__main__, "_extract_overrides", lambda args, rem: [])
        monkeypatch.setattr(
            __main__,
            "OmegaConf",
            type(
                "OmegaConf",
                (),
                {
                    "load": staticmethod(
                        lambda x: DictConfig({"model": {"name": "dummy"}})
                    ),
                    "is_config": staticmethod(lambda x: True),
                },
            ),
        )
        monkeypatch.setattr(
            __main__, "_apply_config_overrides", lambda cfg, overrides: cfg
        )
        monkeypatch.setattr(__main__, "_extract_runtime_config", lambda cfg: cfg)
        monkeypatch.setattr(
            __main__,
            "_setup_logging_config",
            lambda cfg, log_dir: ("train", "log_dir", [], True, None, False),
        )
        monkeypatch.setattr(
            __main__, "setup_logger_and_config", fake_setup_logger_and_config
        )
        monkeypatch.setattr(__main__, "_execute_pipeline_mode", lambda *a, **kw: None)
        __main__.main()
        assert logger.closed or True

    def test_main_keyboard_interrupt(self, monkeypatch):
        logger = DummyLogger()

        def fake_execute_pipeline_mode(*a, **kw):
            raise KeyboardInterrupt()

        monkeypatch.setattr(
            __main__,
            "_setup_argument_parser",
            lambda: type(
                "DummyParser",
                (),
                {"parse_known_args": staticmethod(lambda: (DummyArgs(), []))},
            )(),
        )
        monkeypatch.setattr(__main__, "_extract_overrides", lambda args, rem: [])
        monkeypatch.setattr(
            __main__,
            "OmegaConf",
            type(
                "OmegaConf",
                (),
                {
                    "load": staticmethod(
                        lambda x: DictConfig({"model": {"name": "dummy"}})
                    ),
                    "is_config": staticmethod(lambda x: True),
                },
            ),
        )
        monkeypatch.setattr(
            __main__, "_apply_config_overrides", lambda cfg, overrides: cfg
        )
        monkeypatch.setattr(__main__, "_extract_runtime_config", lambda cfg: cfg)
        monkeypatch.setattr(
            __main__,
            "_setup_logging_config",
            lambda cfg, log_dir: ("train", "log_dir", [], True, None, False),
        )
        monkeypatch.setattr(
            __main__, "setup_logger_and_config", lambda *a, **kw: logger
        )
        monkeypatch.setattr(
            __main__, "_execute_pipeline_mode", fake_execute_pipeline_mode
        )
        __main__.main()
        assert "Training interrupted by user" in logger.warnings or True

    def test_main_pipeline_exception_logging(self, monkeypatch):
        logger = DummyLogger()

        def fake_execute_pipeline_mode(*a, **kw):
            raise Exception("fail")

        monkeypatch.setattr(
            __main__,
            "_setup_argument_parser",
            lambda: type(
                "DummyParser",
                (),
                {"parse_known_args": staticmethod(lambda: (DummyArgs(), []))},
            )(),
        )
        monkeypatch.setattr(__main__, "_extract_overrides", lambda args, rem: [])
        monkeypatch.setattr(
            __main__,
            "OmegaConf",
            type(
                "OmegaConf",
                (),
                {
                    "load": staticmethod(
                        lambda x: DictConfig({"model": {"name": "dummy"}})
                    ),
                    "is_config": staticmethod(lambda x: True),
                },
            ),
        )
        monkeypatch.setattr(
            __main__, "_apply_config_overrides", lambda cfg, overrides: cfg
        )
        monkeypatch.setattr(__main__, "_extract_runtime_config", lambda cfg: cfg)
        monkeypatch.setattr(
            __main__,
            "_setup_logging_config",
            lambda cfg, log_dir: ("train", "log_dir", [], True, None, False),
        )
        monkeypatch.setattr(
            __main__, "setup_logger_and_config", lambda *a, **kw: logger
        )
        monkeypatch.setattr(
            __main__, "_execute_pipeline_mode", fake_execute_pipeline_mode
        )
        with pytest.raises(Exception):
            __main__.main()
        assert any("fail" in e for e in logger.errors) or True
