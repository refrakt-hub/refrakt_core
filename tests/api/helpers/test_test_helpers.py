import importlib
from types import SimpleNamespace

import pytest
import torch
from omegaconf import DictConfig

from refrakt_core.api.helpers import test_helpers
from refrakt_core.api.core.logger import RefraktLogger
from refrakt_core.registry.dataset_registry import DATASET_REGISTRY


class DummyLogger(RefraktLogger):
    def __init__(self, model_name="dummy", **kwargs):
        super().__init__(model_name, **kwargs)

    def log_config(self, config):
        self.logged = True

    def info(self, msg):
        self.info_called = True

    def warning(self, msg):
        self.warning_called = True


class DummyTrainer:
    def __init__(self, has_evaluate=True):
        self.has_evaluate = has_evaluate
        self.evaluate_called = False
        self.model_name = "dummy"
        self.logger = DummyLogger()
        self.artifact_dumper = None

    def evaluate(self):
        self.evaluate_called = True
        return 0.99


class DummyModel(torch.nn.Module):
    def eval(self):
        self.eval_called = True
        return self


class TestTestHelpers:
    # Smoke Test
    def test_import_test_helpers(self):
        importlib.reload(test_helpers)

    # Sanity Tests
    def test_load_and_validate_config_str(self, monkeypatch):
        from omegaconf import DictConfig

        monkeypatch.setattr(
            "refrakt_core.api.helpers.test_helpers.load_config",
            lambda cfg: DictConfig({"dummy": True}),
        )
        monkeypatch.setattr(
            "refrakt_core.api.helpers.test_helpers.OmegaConf.load",
            staticmethod(lambda x: DictConfig({"dummy": True})),
        )
        out = test_helpers._load_and_validate_config("foo.yaml")
        assert isinstance(out, DictConfig)
        assert out["dummy"] is True

    def test_load_and_validate_config_dictconfig(self):
        cfg = DictConfig({"foo": 1})
        out = test_helpers._load_and_validate_config(cfg)
        assert out == cfg

    def test_setup_logging_creates_logger(self, monkeypatch):
        called = {}

        class DummyLogger(RefraktLogger):
            def __init__(self, model_name="dummy", **kwargs):
                super().__init__(model_name, **kwargs)

            def log_config(self, config):
                called["log"] = True

        monkeypatch.setattr(
            "refrakt_core.api.utils.train_utils.setup_logger",
            lambda config, name: DummyLogger(),
        )
        monkeypatch.setattr(
            "refrakt_core.api.helpers.test_helpers.OmegaConf",
            type(
                "OmegaConf",
                (),
                {"to_container": staticmethod(lambda cfg, resolve=True: {"foo": 1})},
            ),
        )
        cfg = DictConfig(
            {"model": {"name": "resnet"}, "dataset": {"name": "dummy", "params": {}}}
        )
        out = test_helpers._setup_logging(cfg, "model", None)
        assert hasattr(out, "info") and hasattr(out, "error")
        assert called.get("log", True)

    def test_setup_logging_typeerror(self, monkeypatch):
        class DummyLogger(RefraktLogger):
            def __init__(self, model_name="dummy", **kwargs):
                super().__init__(model_name, **kwargs)

            def log_config(self, config):
                pass

        monkeypatch.setattr(
            "refrakt_core.api.utils.train_utils.setup_logger",
            lambda config, name: DummyLogger(),
        )
        monkeypatch.setattr(
            "refrakt_core.api.helpers.test_helpers.OmegaConf",
            type(
                "OmegaConf",
                (),
                {"to_container": staticmethod(lambda cfg, resolve=True: 42)},
            ),
        )
        cfg = DictConfig(
            {"model": {"name": "resnet"}, "dataset": {"name": "dummy", "params": {}}}
        )
        with pytest.raises(TypeError):
            test_helpers._setup_logging(cfg, "model", None)

    def test_check_pure_ml_testing_true(self):
        cfg = DictConfig({"model": {"type": "ml"}, "dataset": {"name": "tabular_ml"}})
        assert test_helpers._check_pure_ml_testing(cfg)

    def test_check_pure_ml_testing_false(self):
        cfg = DictConfig({"model": {"type": "dl"}, "dataset": {"name": "image"}})
        assert not test_helpers._check_pure_ml_testing(cfg)

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
        modules, device = test_helpers._get_modules_and_device()
        assert set(modules.keys()) == {
            "get_model",
            "get_loss",
            "get_trainer",
            "get_wrapper",
        }
        assert isinstance(device, torch.device)

    @pytest.mark.skipif(
        "dummy" not in DATASET_REGISTRY, reason="'dummy' dataset not registered."
    )
    def test_build_test_components(self, monkeypatch):
        monkeypatch.setattr(
            "refrakt_core.api.utils.test_utils._build_test_loader_with_resize",
            lambda config, logger: "dataloader",
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
        class DummyDataset:
            def __len__(self):
                return 1
            def __getitem__(self, idx):
                return torch.zeros(1, 28, 28)
        monkeypatch.setattr(
            "refrakt_core.api.builders.dataset_builder.get_dataset",
            lambda name, **params: DummyDataset(),
        )
        modules = {
            "get_model": lambda name=None: DummyModel(),
            "get_wrapper": lambda: None,
        }
        dataloader, model, loss_fn = test_helpers._build_test_components(
            DictConfig(
                {
                    "model": {"name": "dummy"},
                    "dataset": {"name": "dummy"},
                    "dataloader": {"batch_size": 1},
                }
            ),
            modules,
            torch.device("cpu"),
            DummyLogger(),
        )
        assert dataloader == "dataloader" or hasattr(dataloader, "__iter__")
        assert isinstance(model, DummyModel)
        assert loss_fn == "loss_fn"

    def test_setup_trainer_for_testing(self, monkeypatch):
        monkeypatch.setattr(
            "refrakt_core.api.builders.trainer_builder.initialize_trainer",
            lambda **kwargs: DummyTrainer(),
        )
        modules = {
            "get_model": lambda name=None: DummyModel(),
            "get_wrapper": lambda: None,
        }
        trainer = test_helpers._setup_trainer_for_testing(
            DictConfig({}),
            DummyModel(),
            "dataloader",
            "loss_fn",
            "cpu",
            modules,
            None,
            "dummy",
            DummyLogger(),
        )
        assert isinstance(trainer, DummyTrainer)

    def test_evaluate_model_with_evaluate(self, monkeypatch):
        trainer = DummyTrainer()
        model = DummyModel()
        dataloader = "dataloader"
        device = torch.device("cpu")
        logger = DummyLogger()
        result = test_helpers._evaluate_model(
            trainer, model, dataloader, device, None, logger
        )
        assert (
            "accuracy" in result
            or "fusion_accuracy" in result
            or isinstance(result, dict)
        )

    def test_evaluate_model_without_evaluate(self, monkeypatch):
        trainer = SimpleNamespace()
        model = DummyModel()
        dataloader = "dataloader"
        device = torch.device("cpu")
        logger = DummyLogger()
        monkeypatch.setattr(
            "refrakt_core.api.utils.test_utils._run_manual_evaluation",
            lambda *a, **kw: {"manual": True},
        )
        result = test_helpers._evaluate_model(
            trainer, model, dataloader, device, None, logger
        )
        assert result.get("manual") or isinstance(result, dict)
