import pytest
from unittest import mock
import torch
from omegaconf import DictConfig

import sys
sys.modules['refrakt_core.api.builders.dataloader_builder'] = mock.Mock()
sys.modules['refrakt_core.api.builders.dataset_builder'] = mock.Mock()
sys.modules['refrakt_core.api.builders.loss_builder'] = mock.Mock()
sys.modules['refrakt_core.api.builders.model_builder'] = mock.Mock()
sys.modules['refrakt_core.api.builders.scheduler_builder'] = mock.Mock()
sys.modules['refrakt_core.api.core.logger'] = mock.Mock()
sys.modules['refrakt_core.registry.model_registry'] = mock.Mock()
sys.modules['refrakt_core.registry.wrapper_registry'] = mock.Mock()
sys.modules['refrakt_core.schema.artifact'] = mock.Mock()

from refrakt_core.api.utils import train_utils

def test_smoke_import():
    assert hasattr(train_utils, 'get_safe_wrapper')
    assert hasattr(train_utils, 'load_config')
    assert hasattr(train_utils, 'setup_logger')
    assert hasattr(train_utils, 'build_datasets_and_loaders')
    assert hasattr(train_utils, 'build_model_and_log_graph')
    assert hasattr(train_utils, 'build_optimizer_and_scheduler')
    assert hasattr(train_utils, 'setup_artifact_dumper')
    assert hasattr(train_utils, 'load_checkpoint')

def test_sanity_load_config():
    cfg = DictConfig({'a': 1})
    assert train_utils.load_config(cfg) == cfg

def test_unit_get_safe_wrapper():
    wrapper_name = 'dummy'
    raw_model = mock.Mock()
    model_params = {}
    modules = {'get_wrapper': lambda x: mock.Mock(return_value=mock.Mock())}
    device = 'cpu'
    with mock.patch('inspect.signature', return_value=mock.Mock(parameters={'self': None, 'model': None})):
        result = train_utils.get_safe_wrapper(wrapper_name, raw_model, model_params, modules, device)
    assert result is not None

@mock.patch('refrakt_core.api.utils.train_utils.build_dataloader')
@mock.patch('refrakt_core.api.utils.train_utils.build_dataset')
def test_unit_build_datasets_and_loaders(mock_build_dataset, mock_build_dataloader):
    mock_build_dataset.return_value = 'dataset'
    mock_build_dataloader.return_value = 'dataloader'
    cfg = DictConfig({'dataset': {'name': 'dummy_dataset', 'params': {}}, 'dataloader': {}})
    train_dataset, val_dataset, train_loader, val_loader = train_utils.build_datasets_and_loaders(cfg)
    assert train_loader == 'dataloader'
    assert val_loader == 'dataloader'

@mock.patch('refrakt_core.api.builders.model_builder.build_model')
def test_unit_build_model_and_log_graph(mock_build_model):
    cfg = DictConfig({'model': {'name': 'dummy'}})
    modules = {'get_model': lambda x: mock.Mock(), 'get_wrapper': lambda x: mock.Mock(), 'model': mock.Mock()}
    device = 'cpu'
    train_loader = [torch.zeros(1, 3, 3)]
    logger = mock.Mock()
    mock_build_model.return_value = mock.Mock()
    model = train_utils.build_model_and_log_graph(cfg, modules, device, train_loader, logger)
    assert model is not None

@mock.patch('refrakt_core.api.builders.optimizer_builder.build_optimizer')
@mock.patch('refrakt_core.api.builders.scheduler_builder.build_scheduler')
def test_unit_build_optimizer_and_scheduler(mock_build_scheduler, mock_build_optimizer):
    import torch
    cfg = DictConfig({'scheduler': {'type': 'dummy', 'name': 'cosine', 'params': {'T_max': 10}}, 'optimizer': {'name': 'adam', 'params': {}}})
    model = mock.Mock()
    real_optimizer = torch.optim.Adam([torch.zeros(1, requires_grad=True)])
    mock_build_optimizer.return_value = real_optimizer
    optimizer, scheduler = train_utils.build_optimizer_and_scheduler(cfg, model)
    assert optimizer == real_optimizer
    import torch.optim.lr_scheduler
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

@mock.patch('refrakt_core.api.utils.train_utils.ArtifactDumper')
def test_unit_setup_artifact_dumper(mock_dumper):    
    cfg = DictConfig({'artifacts': {'log_every': 1, 'enabled': True}})
    logger = mock.Mock()
    model_name = 'model'
    dumper = train_utils.setup_artifact_dumper(cfg, model_name, logger)
    assert dumper == mock_dumper.return_value

@mock.patch('torch.load')
@mock.patch('os.path.exists', return_value=True)
def test_unit_load_checkpoint_success(mock_exists, mock_torch_load):
    model = mock.Mock(spec=torch.nn.Module)
    logger = mock.Mock()
    device = torch.device('cpu')
    mock_torch_load.return_value = {'model_state_dict': {}, 'global_step': 42}
    step = train_utils.load_checkpoint(model, '/tmp/model.pth', device, logger)
    assert step == 42
    model.load_state_dict.assert_called()
    logger.info.assert_called() 