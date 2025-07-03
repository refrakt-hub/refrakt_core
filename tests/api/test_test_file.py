import pytest
from unittest import mock
import torch
from omegaconf import DictConfig
from refrakt_core.api.test import test as refrakt_test


def test_smoke_import():
    assert callable(refrakt_test)

def test_sanity_test_runs(monkeypatch):
    # Patch all dependencies to allow function to run
    dummy_cfg = DictConfig({'model': {'name': 'dummy', 'params': {}}})
    monkeypatch.setattr('refrakt_core.api.utils.train_utils.load_config', mock.Mock(return_value=dummy_cfg))
    monkeypatch.setattr('refrakt_core.api.utils.train_utils.setup_logger', mock.Mock(return_value=mock.Mock(log_config=mock.Mock(), info=mock.Mock())))
    monkeypatch.setattr('refrakt_core.api.utils.train_utils.setup_artifact_dumper', mock.Mock())
    monkeypatch.setattr('refrakt_core.api.utils.test_utils._build_test_loader', mock.Mock(return_value=[torch.zeros(1, 3, 3)]))
    monkeypatch.setattr('refrakt_core.api.utils.test_utils._load_model_checkpoint', mock.Mock())
    monkeypatch.setattr('refrakt_core.registry.model_registry.get_model', mock.Mock(return_value=mock.Mock()))
    monkeypatch.setattr('refrakt_core.registry.loss_registry.get_loss', mock.Mock(return_value=mock.Mock()))
    monkeypatch.setattr('refrakt_core.registry.trainer_registry.get_trainer', mock.Mock(return_value=mock.Mock()))
    monkeypatch.setattr('refrakt_core.registry.wrapper_registry.get_wrapper', mock.Mock(return_value=mock.Mock()))
    monkeypatch.setattr('refrakt_core.api.builders.model_builder.build_model', mock.Mock(return_value=mock.Mock(__call__=mock.Mock(return_value=torch.zeros(1, 3, 3)), eval=mock.Mock())))
    monkeypatch.setattr('refrakt_core.api.builders.loss_builder.build_loss', mock.Mock(return_value=mock.Mock()))
    monkeypatch.setattr('refrakt_core.api.builders.trainer_builder.initialize_trainer', mock.Mock(return_value=mock.Mock(evaluate=mock.Mock(return_value={'acc': 1.0}), model_name='dummy', logger=mock.Mock(), artifact_dumper=mock.Mock())))
    # Run test and catch SystemExit
    with pytest.raises(SystemExit):
        refrakt_test(cfg=dummy_cfg, model_path='/tmp/model.pth', logger=mock.Mock())

@mock.patch('refrakt_core.api.utils.train_utils.load_config')
@mock.patch('refrakt_core.api.utils.train_utils.setup_logger')
@mock.patch('refrakt_core.api.utils.train_utils.setup_artifact_dumper')
@mock.patch('refrakt_core.api.utils.test_utils._build_test_loader')
@mock.patch('refrakt_core.api.utils.test_utils._load_model_checkpoint')
@mock.patch('refrakt_core.registry.model_registry.get_model')
@mock.patch('refrakt_core.registry.loss_registry.get_loss')
@mock.patch('refrakt_core.registry.trainer_registry.get_trainer')
@mock.patch('refrakt_core.registry.wrapper_registry.get_wrapper')
@mock.patch('refrakt_core.api.builders.model_builder.build_model')
@mock.patch('refrakt_core.api.builders.loss_builder.build_loss')
@mock.patch('refrakt_core.api.builders.trainer_builder.initialize_trainer')
def test_unit_test_success(
    mock_initialize_trainer, mock_build_loss, mock_build_model, mock_get_wrapper, mock_get_trainer, mock_get_loss, mock_get_model, mock_load_model_checkpoint, mock_build_test_loader, mock_setup_artifact_dumper, mock_setup_logger, mock_load_config
):
    # Setup mocks
    dummy_cfg = DictConfig({'model': {'name': 'dummy', 'params': {}}})
    mock_load_config.return_value = dummy_cfg
    mock_setup_logger.return_value = mock.Mock(log_config=mock.Mock(), info=mock.Mock())
    mock_setup_artifact_dumper.return_value = mock.Mock()
    mock_build_test_loader.return_value = [torch.zeros(1, 3, 3)]
    mock_load_model_checkpoint.return_value = 0
    mock_get_model.return_value = mock.Mock()
    mock_get_loss.return_value = mock.Mock()
    mock_get_trainer.return_value = mock.Mock()
    mock_get_wrapper.return_value = mock.Mock()
    model_mock = mock.Mock()
    model_mock.__call__ = mock.Mock(return_value=torch.zeros(1, 3, 3))
    model_mock.eval = mock.Mock()
    mock_build_model.return_value = model_mock
    mock_build_loss.return_value = mock.Mock()
    trainer_mock = mock.Mock(evaluate=mock.Mock(return_value={'acc': 1.0}), model_name='dummy', logger=mock.Mock(), artifact_dumper=mock.Mock())
    mock_initialize_trainer.return_value = trainer_mock
    logger = mock.Mock()
    with pytest.raises(SystemExit):
        refrakt_test(cfg=dummy_cfg, model_path='/tmp/model.pth', logger=logger) 