import pytest
from unittest import mock
import torch
from omegaconf import DictConfig
from refrakt_core.api.inference import inference

def test_smoke_import():
    assert callable(inference)

def test_sanity_inference_runs(monkeypatch):
    # Use a real DictConfig
    dummy_cfg = DictConfig({'model': {'name': 'dummy', 'params': {}}})
    monkeypatch.setattr('refrakt_core.api.utils.train_utils.load_config', mock.Mock(return_value=dummy_cfg))
    monkeypatch.setattr('refrakt_core.api.utils.train_utils.setup_logger', mock.Mock(return_value=mock.Mock(log_config=mock.Mock(), info=mock.Mock())))
    monkeypatch.setattr('refrakt_core.api.utils.train_utils.setup_data_loader_for_inference', mock.Mock(return_value=[torch.zeros(1, 3, 3)]))
    monkeypatch.setattr('refrakt_core.api.utils.train_utils.setup_artifact_dumper', mock.Mock())
    monkeypatch.setattr('refrakt_core.api.utils.test_utils._load_model_checkpoint', mock.Mock())
    monkeypatch.setattr('refrakt_core.registry.model_registry.get_model', mock.Mock(return_value=mock.Mock()))
    monkeypatch.setattr('refrakt_core.registry.wrapper_registry.get_wrapper', mock.Mock(return_value=mock.Mock()))
    monkeypatch.setattr('refrakt_core.api.builders.model_builder.build_model', mock.Mock(return_value=mock.Mock(__call__=mock.Mock(return_value=torch.zeros(1, 3, 3)), eval=mock.Mock())))
    # Run inference and catch SystemExit
    with pytest.raises(SystemExit):
        inference(cfg=dummy_cfg, model_path='/tmp/model.pth', logger=mock.Mock())

@mock.patch('refrakt_core.api.utils.train_utils.load_config')
@mock.patch('refrakt_core.api.utils.train_utils.setup_logger')
@mock.patch('refrakt_core.api.utils.train_utils.setup_data_loader_for_inference')
@mock.patch('refrakt_core.api.utils.train_utils.setup_artifact_dumper')
@mock.patch('refrakt_core.api.utils.test_utils._load_model_checkpoint')
@mock.patch('refrakt_core.registry.model_registry.get_model')
@mock.patch('refrakt_core.registry.wrapper_registry.get_wrapper')
@mock.patch('refrakt_core.api.builders.model_builder.build_model')
def test_unit_inference_success(
    mock_build_model, mock_get_wrapper, mock_get_model, mock_load_model_checkpoint,
    mock_setup_artifact_dumper, mock_setup_data_loader_for_inference, mock_setup_logger, mock_load_config
):
    # Use a real DictConfig
    dummy_cfg = DictConfig({'model': {'name': 'dummy', 'params': {}}})
    mock_load_config.return_value = dummy_cfg
    mock_setup_logger.return_value = mock.Mock(log_config=mock.Mock(), info=mock.Mock())
    mock_setup_data_loader_for_inference.return_value = [torch.zeros(1, 3, 3)]
    mock_setup_artifact_dumper.return_value = mock.Mock()
    mock_load_model_checkpoint.return_value = 0
    mock_get_model.return_value = mock.Mock()
    mock_get_wrapper.return_value = mock.Mock()
    model_mock = mock.Mock()
    model_mock.__call__ = mock.Mock(return_value=torch.zeros(1, 3, 3))
    model_mock.eval = mock.Mock()
    mock_build_model.return_value = model_mock
    logger = mock.Mock()
    with pytest.raises(SystemExit):
        inference(cfg=dummy_cfg, model_path='/tmp/model.pth', logger=logger) 