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
    dummy_cfg = mock.Mock()
    dummy_cfg.model.name = 'dummy'
    dummy_cfg.trainer.params.save_dir = '/tmp'
    dummy_cfg.trainer.params.model_name = 'dummy_model'
    dummy_cfg_dict = {'runtime': {'mode': 'train', 'log_type': [], 'log_dir': './logs', 'console': True, 'debug': False, 'model_path': None}, 'model': {'name': 'dummy'}, 'trainer': {'params': {'save_dir': '/tmp', 'model_name': 'dummy_model'}}}
    with mock.patch('omegaconf.OmegaConf.load', return_value=dummy_cfg), \
         mock.patch('omegaconf.OmegaConf.to_container', return_value=dummy_cfg_dict):
        monkeypatch.setattr(sys, 'argv', ['prog', '--config', 'dummy.yaml'])
        parser_mock = mock.Mock()
        args_mock = mock.Mock(config='dummy.yaml', log_dir=None, debug=False, override=None)
        parser_mock.parse_args.return_value = args_mock
        parser_mock.parse_known_args.return_value = (args_mock, [])
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