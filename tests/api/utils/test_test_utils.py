import pytest
from unittest import mock
import types
import torch
from omegaconf import DictConfig

import sys
sys.modules['refrakt_core.api.builders.dataloader_builder'] = mock.Mock()
sys.modules['refrakt_core.api.builders.dataset_builder'] = mock.Mock()
sys.modules['refrakt_core.utils.methods'] = mock.Mock()

from refrakt_core.api.utils import test_utils

def test_smoke_import():
    assert hasattr(test_utils, '_load_config')
    assert hasattr(test_utils, '_build_test_loader')
    assert hasattr(test_utils, '_load_model_checkpoint')

def test_sanity_load_config():
    cfg = {'a': 1}
    # Should return as-is if not a string
    assert test_utils._load_config(cfg) == cfg

@mock.patch('refrakt_core.api.utils.test_utils.build_dataloader')
@mock.patch('refrakt_core.api.utils.test_utils.build_dataset')
def test_unit_build_test_loader(mock_build_dataset, mock_build_dataloader):
    # Setup mocks
    mock_build_dataset.return_value = 'dataset'
    mock_build_dataloader.return_value = 'dataloader'
    # Use a valid config with 'name' and 'params'
    config = DictConfig({'dataset': {'name': 'dummy_dataset', 'params': {}}, 'dataloader': {}})
    result = test_utils._build_test_loader(config)
    assert result == 'dataloader'
    mock_build_dataset.assert_called()
    mock_build_dataloader.assert_called()

@mock.patch('torch.load')
def test_unit_load_model_checkpoint_file_not_found(mock_torch_load):
    model = mock.Mock(spec=torch.nn.Module)
    logger = mock.Mock()
    device = torch.device('cpu')
    # Should raise FileNotFoundError if no file exists and no fallback
    with pytest.raises(FileNotFoundError):
        test_utils._load_model_checkpoint(model, '/tmp/nonexistent.pth', device, logger)

@mock.patch('torch.load')
@mock.patch('os.path.exists', return_value=True)
def test_unit_load_model_checkpoint_success(mock_exists, mock_torch_load):
    model = mock.Mock(spec=torch.nn.Module)
    logger = mock.Mock()
    device = torch.device('cpu')
    mock_torch_load.return_value = {'model_state_dict': {}, 'global_step': 42}
    step = test_utils._load_model_checkpoint(model, '/tmp/model.pth', device, logger)
    assert step == 42
    model.load_state_dict.assert_called()
    logger.info.assert_called() 