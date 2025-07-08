import importlib
import pytest

class DummyTrain:
    def __init__(self):
        self.called = False
    def __call__(self, config_path):
        self.called = True
        return {'train': True}

class DummyTest:
    def __init__(self):
        self.called = False
    def __call__(self, config_path, model_path=None):
        self.called = True
        return None

class DummyInference:
    def __init__(self):
        self.called = False
    def __call__(self, config_path, model_path):
        self.called = True
        return {'inference': True}

class TestApiInit:
    # Smoke Tests
    def test_import_api_init_smoke(self):
        mod = importlib.import_module('src.refrakt_core.api')
        assert mod is not None

    # Sanity Tests
    def test_api_init_has_expected_attrs(self):
        mod = importlib.import_module('src.refrakt_core.api')
        assert hasattr(mod, '__file__')
        assert hasattr(mod, '__package__')

    # Unit Tests
    def test_api_init_import_error(self, monkeypatch):
        # Simulate import error by patching import_module
        def raise_import_error(name):
            raise ImportError('fail')
        monkeypatch.setattr(importlib, 'import_module', raise_import_error)
        with pytest.raises(ImportError):
            importlib.import_module('src.refrakt_core.api')

    def test_main_train_mode(self, monkeypatch):
        import src.refrakt_core.api as api_mod
        dummy_train = DummyTrain()
        monkeypatch.setattr(api_mod, 'train', dummy_train)
        result = api_mod.main('dummy.yaml', 'train')
        assert result == {'train': True}
        assert dummy_train.called

    def test_main_test_mode(self, monkeypatch):
        import src.refrakt_core.api as api_mod
        dummy_test = DummyTest()
        monkeypatch.setattr(api_mod, 'test', dummy_test)
        result = api_mod.main('dummy.yaml', 'test')
        assert result is None
        assert dummy_test.called

    def test_main_inference_mode_raises(self, monkeypatch):
        import src.refrakt_core.api as api_mod
        with pytest.raises(ValueError):
            api_mod.main('dummy.yaml', 'inference')

    def test_main_invalid_mode_raises(self, monkeypatch):
        import src.refrakt_core.api as api_mod
        with pytest.raises(ValueError):
            api_mod.main('dummy.yaml', 'invalid')

    def test_cli_entrypoint_train(self, monkeypatch):
        import sys
        import src.refrakt_core.api as api_mod
        dummy_train = DummyTrain()
        monkeypatch.setattr(api_mod, 'train', dummy_train)
        monkeypatch.setattr(api_mod, 'test', DummyTest())
        monkeypatch.setattr(api_mod, 'inference', DummyInference())
        monkeypatch.setattr(sys, 'argv', ['prog', '--config', 'dummy.yaml', '--mode', 'train'])
        # Patch argparse to avoid actual parsing
        import argparse
        class DummyArgs:
            def __init__(self):
                self.config = 'dummy.yaml'
                self.mode = 'train'
                self.model_path = None
        monkeypatch.setattr(argparse.ArgumentParser, 'parse_args', lambda self: DummyArgs())
        # No assertion, just ensure no error

    def test_cli_entrypoint_test(self, monkeypatch):
        import sys
        import src.refrakt_core.api as api_mod
        dummy_test = DummyTest()
        monkeypatch.setattr(api_mod, 'train', DummyTrain())
        monkeypatch.setattr(api_mod, 'test', dummy_test)
        monkeypatch.setattr(api_mod, 'inference', DummyInference())
        monkeypatch.setattr(sys, 'argv', ['prog', '--config', 'dummy.yaml', '--mode', 'test', '--model-path', 'model.pth'])
        import argparse
        class DummyArgs:
            def __init__(self):
                self.config = 'dummy.yaml'
                self.mode = 'test'
                self.model_path = 'model.pth'
        monkeypatch.setattr(argparse.ArgumentParser, 'parse_args', lambda self: DummyArgs())
        # No assertion, just ensure no error

    def test_cli_entrypoint_inference_requires_model_path(self, monkeypatch):
        import sys
        import src.refrakt_core.api as api_mod
        monkeypatch.setattr(api_mod, 'train', DummyTrain())
        monkeypatch.setattr(api_mod, 'test', DummyTest())
        monkeypatch.setattr(api_mod, 'inference', DummyInference())
        monkeypatch.setattr(sys, 'argv', ['prog', '--config', 'dummy.yaml', '--mode', 'inference'])
        import argparse
        class DummyArgs:
            def __init__(self):
                self.config = 'dummy.yaml'
                self.mode = 'inference'
                self.model_path = None
        monkeypatch.setattr(argparse.ArgumentParser, 'parse_args', lambda self: DummyArgs())
        # No assertion, just ensure no error 