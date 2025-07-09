import importlib

import pytest
import torch
from omegaconf import OmegaConf

import src.refrakt_core.api.builders.utils.trainer_utils as trainer_utils


class DummyTrainer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.initialized = True

    def train(self):
        return "trained"

    def evaluate(self):
        return "evaluated"


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)


class DummyGAN:
    def __init__(self):
        self.generator = DummyModel()
        self.discriminator = DummyModel()


class TestTrainerUtils:
    # Smoke Tests
    def test_import_trainer_utils(self):
        importlib.reload(trainer_utils)

    def test_trainer_utils_has_any_symbol(self):
        symbols = [s for s in dir(trainer_utils) if not s.startswith("__")]
        assert symbols

    # Sanity Tests
    def test_validate_trainer_config_valid(self):
        cfg_dict = {"trainer": {"name": "supervised", "params": {"foo": 1}}}
        name, params = trainer_utils.validate_trainer_config(cfg_dict)
        assert name == "supervised"
        assert params["foo"] == 1

    # Unit Tests
    def test_validate_trainer_config_type_error(self):
        with pytest.raises(TypeError):
            trainer_utils.validate_trainer_config({"trainer": None})

    def test_setup_standard_trainer(self):
        trainer_cls = DummyTrainer
        model = DummyModel()
        train_loader = [1, 2, 3]
        val_loader = [4, 5, 6]
        loss_fn = lambda x, y: 0
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = None
        device = "cpu"
        artifact_dumper = None
        trainer_params = {"foo": 42}
        cfg_dict = {"optimizer": {"name": "adam", "params": {"lr": 0.01}}}
        trainer = trainer_utils.setup_standard_trainer(
            trainer_cls,
            model,
            train_loader,
            val_loader,
            loss_fn,
            optimizer,
            scheduler,
            device,
            artifact_dumper,
            trainer_params,
            cfg_dict,
        )
        assert isinstance(trainer, DummyTrainer)
        assert trainer.kwargs["foo"] == 42

    def test_setup_standard_trainer_type_error(self):
        trainer_cls = DummyTrainer
        model = DummyModel()
        train_loader = [1, 2, 3]
        val_loader = [4, 5, 6]
        loss_fn = lambda x, y: 0
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = None
        device = "cpu"
        artifact_dumper = None
        trainer_params = {"foo": 42}
        cfg_dict = {"optimizer": None}
        with pytest.raises(TypeError):
            trainer_utils.setup_standard_trainer(
                trainer_cls,
                model,
                train_loader,
                val_loader,
                loss_fn,
                optimizer,
                scheduler,
                device,
                artifact_dumper,
                trainer_params,
                cfg_dict,
            )
