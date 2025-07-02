import pytest
from omegaconf import OmegaConf
from unittest.mock import MagicMock
from refrakt_core.api.builders.trainer_builder import initialize_trainer

class DummyTrainer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

def dummy_get_trainer(name):
    return DummyTrainer

def test_initialize_trainer_smoke():
    cfg = OmegaConf.create({'trainer': {'name': 'supervised', 'params': {}}, 'optimizer': {'name': 'adam', 'params': {}}})
    modules = {'get_trainer': dummy_get_trainer, 'artifact_dumper': None}
    trainer = initialize_trainer(cfg, model='m', train_loader='tl', val_loader='vl', loss_fn='lf', optimizer='opt', scheduler=None, device='cpu', modules=modules, save_dir=None)
    assert isinstance(trainer, DummyTrainer)

def test_initialize_trainer_fallback():
    cfg = OmegaConf.create({'trainer': {'name': 'unknown', 'params': {}}, 'optimizer': {'name': 'adam', 'params': {}}})
    modules = {'get_trainer': dummy_get_trainer, 'artifact_dumper': None}
    trainer = initialize_trainer(cfg, model='m', train_loader='tl', val_loader='vl', loss_fn='lf', optimizer='opt', scheduler=None, device='cpu', modules=modules, save_dir=None)
    assert isinstance(trainer, DummyTrainer) 