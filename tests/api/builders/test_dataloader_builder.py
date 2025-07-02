import pytest
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from refrakt_core.api.builders.dataloader_builder import build_dataloader
from tests.helpers.fixtures import dummy_dataset

def test_build_dataloader_smoke(dummy_dataset):
    cfg = OmegaConf.create({
        'batch_size': 4,
        'shuffle': True,
        'num_workers': 0,
        'drop_last': False
    })
    loader = build_dataloader(dummy_dataset, cfg)
    assert isinstance(loader, DataLoader)
    batch = next(iter(loader))
    assert len(batch) == 2  # (data, target)
    assert batch[0].shape[0] == 4

def test_build_dataloader_missing_batch_size(dummy_dataset):
    cfg = OmegaConf.create({'shuffle': True})
    with pytest.raises(KeyError):
        build_dataloader(dummy_dataset, cfg)

def test_build_dataloader_bad_params(dummy_dataset):
    cfg = OmegaConf.create({'params': 'not_a_dict', 'batch_size': 4})
    with pytest.raises(TypeError):
        build_dataloader(dummy_dataset, cfg) 