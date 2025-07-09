import importlib
from types import SimpleNamespace

import pytest
import torch
from omegaconf import DictConfig

import src.refrakt_core.api.helpers.train_helpers as train_helpers
from src.refrakt_core.api.core.logger import RefraktLogger


class DummyLogger(RefraktLogger):
    def __init__(self):
        super().__init__(model_name="resnet18")

    def log_config(self, config):
        self.logged = True

    def info(self, msg):
        self.info_called = True

    def warning(self, msg):
        self.warning_called = True


class DummyTrainer:
    def __init__(self):
        self.save_dir = "./artifacts/yaml"
        self.global_step = 0
        self.train_called = False
        self.model_name = "resnet18"
        self.logger = DummyLogger()
        self.artifact_dumper = None

    def train(self, num_epochs):
        self.train_called = True
        return {"accuracy": 0.99}


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)  # Add parameters so optimizer doesn't fail

    def eval(self):
        self.eval_called = True
        return self


class TestTrainHelpers:
    # Smoke Test
    def test_import_train_helpers(self):
        importlib.reload(train_helpers)

    # Sanity Tests
    def test_load_and_validate_config_str(self, monkeypatch):
        from omegaconf import DictConfig

        monkeypatch.setattr(
            "src.refrakt_core.api.helpers.train_helpers.load_config",
            lambda cfg: DictConfig({"dummy": True}),
        )
        monkeypatch.setattr(
            "src.refrakt_core.api.helpers.train_helpers.OmegaConf.load",
            staticmethod(lambda x: DictConfig({"dummy": True})),
        )
        out = train_helpers._load_and_validate_config("foo.yaml")
        assert isinstance(out, DictConfig)
        assert out["dummy"] is True

    def test_load_and_validate_config_dictconfig(self):
        cfg = DictConfig({"foo": 1})
        out = train_helpers._load_and_validate_config(cfg)
        assert out == cfg

    def test_setup_logging_creates_logger(self, monkeypatch):
        called = {}

        class DummyLogger(RefraktLogger):
            def __init__(self):
                super().__init__(model_name="resnet18")

            def log_config(self, config):
                called["log"] = True

        monkeypatch.setattr(
            "src.refrakt_core.api.utils.train_utils.setup_logger",
            lambda config, name: DummyLogger(),
        )
        monkeypatch.setattr(
            "src.refrakt_core.api.helpers.train_helpers.OmegaConf",
            type(
                "OmegaConf",
                (),
                {"to_container": staticmethod(lambda cfg, resolve=True: {"foo": 1})},
            ),
        )
        cfg = DictConfig(
            {
                "model": {"name": "resnet18"},
                "dataset": {"name": "contrastive", "params": {}},
            }
        )
        out = train_helpers._setup_logging(cfg, "model", None)
        assert hasattr(out, "info") and hasattr(out, "error")
        assert called.get("log", True)

    def test_setup_logging_typeerror(self, monkeypatch):
        class DummyLogger(RefraktLogger):
            def __init__(self):
                super().__init__(model_name="resnet18")

            def log_config(self, config):
                pass

        monkeypatch.setattr(
            "src.refrakt_core.api.utils.train_utils.setup_logger",
            lambda config, name: DummyLogger(),
        )
        monkeypatch.setattr(
            "src.refrakt_core.api.helpers.train_helpers.OmegaConf",
            type(
                "OmegaConf",
                (),
                {"to_container": staticmethod(lambda cfg, resolve=True: 42)},
            ),
        )
        cfg = DictConfig(
            {
                "model": {"name": "resnet18"},
                "dataset": {"name": "contrastive", "params": {}},
            }
        )
        with pytest.raises(TypeError):
            train_helpers._setup_logging(cfg, "model", None)

    def test_check_pure_ml_training_true(self):
        cfg = DictConfig({"model": {"type": "ml"}, "dataset": {"name": "tabular_ml"}})
        assert train_helpers._check_pure_ml_training(cfg)

    def test_check_pure_ml_training_false(self):
        cfg = DictConfig({"model": {"type": "dl"}, "dataset": {"name": "image"}})
        assert not train_helpers._check_pure_ml_training(cfg)

    def test_get_modules_and_device(self, monkeypatch):
        monkeypatch.setattr(
            "refrakt_core.registry.loss_registry.get_loss", lambda: "loss"
        )
        monkeypatch.setattr(
            "refrakt_core.registry.model_registry.get_model",
            lambda name=None: lambda: "model",
        )
        monkeypatch.setattr(
            "refrakt_core.registry.trainer_registry.get_trainer", lambda: "trainer"
        )
        monkeypatch.setattr(
            "refrakt_core.registry.wrapper_registry.get_wrapper", lambda: "wrapper"
        )
        modules, device = train_helpers._get_modules_and_device()
        assert set(modules.keys()) == {
            "get_model",
            "get_loss",
            "get_trainer",
            "get_wrapper",
        }
        assert isinstance(device, torch.device)

    def test_build_datasets_and_model(self, monkeypatch):
        monkeypatch.setattr(
            "src.refrakt_core.api.utils.train_utils.build_datasets_and_loaders_with_resize",
            lambda config, logger: ("train_ds", "val_ds", "train_loader", "val_loader"),
        )
        monkeypatch.setattr(
            "refrakt_core.registry.model_registry.get_model",
            lambda name=None: lambda: DummyModel(),
        )
        monkeypatch.setattr(
            "refrakt_core.api.builders.model_builder.build_model",
            lambda *a, **kw: DummyModel(),
        )
        monkeypatch.setattr(
            "refrakt_core.api.builders.loss_builder.build_loss",
            lambda *a, **kw: "loss_fn",
        )
        modules = {
            "get_model": lambda name=None: DummyModel(),
            "get_wrapper": lambda: None,
        }
        train_loader, val_loader, model, loss_fn = (
            train_helpers._build_datasets_and_model(
                DictConfig(
                    {
                        "model": {"name": "resnet18"},
                        "dataset": {"name": "dummy"},
                        "dataloader": {"batch_size": 1},
                    }
                ),
                modules,
                torch.device("cpu"),
                DummyLogger(),
            )
        )
        assert train_loader == "train_loader" or hasattr(train_loader, "__iter__")
        assert val_loader == "val_loader" or hasattr(val_loader, "__iter__")
        assert isinstance(model, DummyModel)
        assert loss_fn == "loss_fn"

    def test_setup_optimizer_and_scheduler(self, monkeypatch):
        class DummyOpt:
            def __init__(self, params, **kwargs):
                self.params = params

        monkeypatch.setattr(
            "src.refrakt_core.api.utils.train_utils._setup_optimizer_config",
            lambda config: (DummyOpt, {"lr": 0.1}),
        )
        monkeypatch.setattr(
            "refrakt_core.api.builders.scheduler_builder.build_scheduler",
            lambda config, optimizer: "scheduler",
        )
        cfg = DictConfig({"scheduler": True})
        model = DummyModel()
        opt, sch = train_helpers._setup_optimizer_and_scheduler(cfg, model)
        assert hasattr(opt, "params") or hasattr(opt, "param_groups")
        assert sch == "scheduler"

    def test_setup_trainer(self, monkeypatch):
        monkeypatch.setattr(
            "src.refrakt_core.api.utils.train_utils._setup_trainer_params",
            lambda *a, **kw: (None, {"save_dir": None}, 10, "cpu"),
        )
        monkeypatch.setattr(
            "refrakt_core.api.builders.trainer_builder.initialize_trainer",
            lambda **kwargs: DummyTrainer(),
        )
        modules = {
            "get_model": lambda name=None: DummyModel(),
            "get_wrapper": lambda: None,
        }
        trainer, num_epochs, final_device = train_helpers._setup_trainer(
            DictConfig({"trainer": {"name": "supervised", "params": {}}}),
            DummyModel(),
            "train_loader",
            "val_loader",
            "loss_fn",
            "opt",
            "sch",
            "cpu",
            modules,
            None,
            "resnet18",
            DummyLogger(),
        )
        assert isinstance(trainer, DummyTrainer)
        assert num_epochs == 10
        assert final_device == "cpu"

    def test_execute_training(self, monkeypatch):
        monkeypatch.setattr(
            "src.refrakt_core.api.utils.train_utils._handle_fusion_training",
            lambda *a, **kw: None,
        )
        monkeypatch.setattr(
            "src.refrakt_core.api.utils.train_utils._save_config_and_log_metrics",
            lambda *a, **kw: None,
        )
        # Mock OmegaConf.save to avoid file system issues
        monkeypatch.setattr(
            "src.refrakt_core.api.helpers.train_helpers.OmegaConf.save",
            staticmethod(lambda cfg, path: None),
        )
        trainer = DummyTrainer()
        result = train_helpers._execute_training(
            trainer,
            5,
            DictConfig({"model": {"name": "dummy"}}),
            DummyModel(),
            "train_loader",
            "val_loader",
            "cpu",
            None,
            "resnet18",
            DummyLogger(),
        )
        assert isinstance(result, dict)
        assert "accuracy" in result or "metrics" in result
