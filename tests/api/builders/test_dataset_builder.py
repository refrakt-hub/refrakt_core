import pytest
from omegaconf import OmegaConf
from unittest.mock import patch, MagicMock
from refrakt_core.api.builders.dataset_builder import build_dataset

def test_build_dataset_smoke():
    cfg = OmegaConf.create({'name': 'dummy', 'params': {}})
    with patch('refrakt_core.api.builders.dataset_builder.get_dataset') as get_dataset:
        get_dataset.return_value = 'dataset_obj'
        result = build_dataset(cfg)
        assert result == 'dataset_obj'

def test_build_dataset_with_wrapper():
    cfg = OmegaConf.create({'name': 'dummy', 'params': {}, 'wrapper': 'wrap', 'transform': None})
    with patch('refrakt_core.api.builders.dataset_builder.get_dataset') as get_dataset, \
         patch('refrakt_core.api.builders.dataset_builder.DATASET_REGISTRY', {'wrap': MagicMock(return_value='wrapped')}) as reg:
        get_dataset.return_value = 'base_dataset'
        result = build_dataset(cfg)
        assert result == 'wrapped'

def test_build_dataset_bad_type():
    cfg = OmegaConf.create({'name': 123, 'params': {}})
    with pytest.raises(TypeError):
        build_dataset(cfg) 