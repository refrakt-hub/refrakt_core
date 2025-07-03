import pytest
from unittest import mock
import sys
import types
from refrakt_core.api import __main__

def test_smoke_import():
    assert hasattr(__main__, 'main')

def test_sanity_main_runs(monkeypatch):
    # Patch sys.argv and all CLI dependencies
    monkeypatch.setattr(sys, 'argv', ['prog', '--config', 'dummy.yaml'])
    parser_mock = mock.Mock()
    parser_mock.parse_args.return_value = mock.Mock(config='dummy.yaml', log_dir=None, debug=False)
    monkeypatch.setattr('argparse.ArgumentParser', mock.Mock(return_value=parser_mock))
    monkeypatch.setattr('refrakt_core.api.__main__.main', lambda: None)
    # Should not raise
    __main__.main()

def test_unit_main_dispatch(monkeypatch):
    # Patch sys.modules for dynamic imports
    sys.modules['refrakt_core.api.inference'] = mock.Mock(inference=mock.Mock())
    sys.modules['refrakt_core.api.test'] = mock.Mock(test=mock.Mock())
    sys.modules['refrakt_core.api.train'] = mock.Mock(train=mock.Mock())
    # Patch OmegaConf.load and OmegaConf.to_container only for this test
    import importlib
    real_omegaconf = sys.modules.get('omegaconf')
    fake_omegaconf = types.ModuleType('omegaconf')
    dummy_cfg = mock.Mock()
    dummy_cfg.model.name = 'dummy'
    dummy_cfg.trainer.params.save_dir = '/tmp'
    dummy_cfg.trainer.params.model_name = 'dummy_model'
    dummy_cfg_dict = {'runtime': {'mode': 'train', 'log_type': [], 'log_dir': './logs', 'console': True, 'debug': False, 'model_path': None}, 'model': {'name': 'dummy'}, 'trainer': {'params': {'save_dir': '/tmp', 'model_name': 'dummy_model'}}}
    class FakeOmegaConf:
        @staticmethod
        def load(*args, **kwargs):
            return dummy_cfg
        @staticmethod
        def to_container(*args, **kwargs):
            return dummy_cfg_dict
    class FakeDictConfig(dict):
        pass
    class FakeListConfig(list):
        pass
    class FakeDictKeyType:
        pass
    setattr(fake_omegaconf, 'OmegaConf', FakeOmegaConf)
    setattr(fake_omegaconf, 'DictConfig', FakeDictConfig)
    setattr(fake_omegaconf, 'ListConfig', FakeListConfig)
    setattr(fake_omegaconf, 'DictKeyType', FakeDictKeyType)
    sys.modules['omegaconf'] = fake_omegaconf
    monkeypatch.setattr(sys, 'argv', ['prog', '--config', 'dummy.yaml'])
    parser_mock = mock.Mock()
    parser_mock.parse_args.return_value = mock.Mock(config='dummy.yaml', log_dir=None, debug=False)
    monkeypatch.setattr('argparse.ArgumentParser', mock.Mock(return_value=parser_mock))
    for mode in ['train', 'test', 'inference', 'pipeline']:
        dummy_cfg_dict['runtime']['mode'] = mode
        if mode == 'inference':
            dummy_cfg_dict['runtime']['model_path'] = '/tmp/model.pth'
        else:
            dummy_cfg_dict['runtime']['model_path'] = None
        try:
            __main__.main()
        except ValueError as e:
            if mode == 'inference':
                pytest.fail(f"main() raised ValueError unexpectedly: {e}")
    assert sys.modules['refrakt_core.api.train'].train.called
    assert sys.modules['refrakt_core.api.test'].test.called
    assert sys.modules['refrakt_core.api.inference'].inference.called
    # Restore the real omegaconf after the test
    if real_omegaconf is not None:
        sys.modules['omegaconf'] = real_omegaconf
    else:
        del sys.modules['omegaconf'] 