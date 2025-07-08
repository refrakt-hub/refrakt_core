import pytest
from omegaconf import OmegaConf
from src.refrakt_core.api.builders.trainer_builder import initialize_trainer

class DummyTrainer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.initialized = True
    def train(self):
        return 'trained'
    def evaluate(self):
        return 'evaluated'

@pytest.fixture
def modules(monkeypatch):
    return {
        'get_trainer': lambda name: DummyTrainer,
        'artifact_dumper': None,
    }

@pytest.fixture
def base_cfg():
    return OmegaConf.create({'trainer': {'name': 'supervised', 'params': {}}, 'optimizer': {'name': 'adam', 'params': {}}})

@pytest.fixture
def model():
    class M:
        pass
    return M()

@pytest.fixture
def loader():
    return [1, 2, 3]

@pytest.fixture
def loss_fn():
    return lambda x, y: 0

@pytest.fixture
def optimizer():
    class O:
        def parameters(self):
            return []
    return O()

@pytest.fixture
def scheduler():
    return None

class TestTrainerBuilder:
    # Smoke Tests
    def test_initialize_trainer_smoke(self, base_cfg, model, loader, loss_fn, optimizer, scheduler, modules):
        trainer = initialize_trainer(base_cfg, model, loader, loader, loss_fn, optimizer, scheduler, 'cpu', modules, None)
        assert isinstance(trainer, DummyTrainer)
        assert trainer.initialized

    # Sanity Tests
    def test_initialize_trainer_sanity_params(self, base_cfg, model, loader, loss_fn, optimizer, scheduler, modules):
        base_cfg.trainer['params'] = {'foo': 42}
        trainer = initialize_trainer(base_cfg, model, loader, loader, loss_fn, optimizer, scheduler, 'cpu', modules, None)
        assert trainer.kwargs['foo'] == 42

    # Unit Tests
    def test_initialize_trainer_unit_missing_trainer_name(self, base_cfg, model, loader, loss_fn, optimizer, scheduler, modules):
        base_cfg.trainer['name'] = None
        with pytest.raises(TypeError):
            initialize_trainer(base_cfg, model, loader, loader, loss_fn, optimizer, scheduler, 'cpu', modules, None)

    def test_initialize_trainer_unit_params_not_dict(self, base_cfg, model, loader, loss_fn, optimizer, scheduler, modules):
        base_cfg.trainer['params'] = 123
        with pytest.raises(TypeError):
            initialize_trainer(base_cfg, model, loader, loader, loss_fn, optimizer, scheduler, 'cpu', modules, None)

    def test_initialize_trainer_unit_invalid_cfg_type(self, model, loader, loss_fn, optimizer, scheduler, modules):
        with pytest.raises(TypeError):
            initialize_trainer('not_a_cfg', model, loader, loader, loss_fn, optimizer, scheduler, 'cpu', modules, None) 