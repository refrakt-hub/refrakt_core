import argparse
import importlib

# --- Add module-level fixture to patch pipeline utils ---
import sys
from types import SimpleNamespace

import pytest
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

import refrakt_core.api.helpers.cli_helpers as cli_helpers
import refrakt_core.api.inference as inference_module
import refrakt_core.api.test as test_module
import refrakt_core.api.train as train_module
from refrakt_core.datasets import ContrastiveDataset


@pytest.fixture(autouse=True)
def patch_pipeline_utils(monkeypatch):
    called = {}
    # Patch pipeline utility functions only
    monkeypatch.setattr(
        "refrakt_core.api.utils.pipeline_utils.execute_training_pipeline",
        lambda *a, **kw: called.setdefault("train_util", True),
    )
    monkeypatch.setattr(
        "refrakt_core.api.utils.pipeline_utils.execute_testing_pipeline",
        lambda *a, **kw: called.setdefault("test_util", True),
    )
    monkeypatch.setattr(
        "refrakt_core.api.utils.pipeline_utils.execute_inference_pipeline",
        lambda *a, **kw: called.setdefault("inference_util", True),
    )
    monkeypatch.setattr(
        "refrakt_core.api.utils.pipeline_utils.execute_full_pipeline",
        lambda *a, **kw: called.setdefault("pipeline_util", True),
    )
    yield called


# --- End module-level fixture ---


# Add a simple base dataset for testing
class DummyBaseDataset(Dataset):
    def __init__(self, size=10, train=False):
        self.data = torch.randn(size, 3, 32, 32)
        self.labels = torch.randint(0, 10, (size,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Add fixture to patch dataset registry
@pytest.fixture(autouse=True)
def patch_dataset_registry(monkeypatch):
    def mock_get_dataset(name, **kwargs):
        if name == "dummy":
            return DummyBaseDataset(**kwargs)
        elif name == "contrastive":
            base_dataset = kwargs.get("base_dataset")
            transform = kwargs.get("transform")
            return ContrastiveDataset(base_dataset=base_dataset, transform=transform)
        raise ValueError(f"Unknown dataset: {name}")

    def mock_get_transform(name, **kwargs):
        if name == "contrastive":
            return lambda x: x  # Identity transform for testing
        raise ValueError(f"Unknown transform: {name}")

    def mock_os_path_exists(path):
        return True  # Mock all files as existing

    def mock_torch_load(*args, **kwargs):
        return {"model_state_dict": {}}  # Mock empty state dict

    monkeypatch.setattr(
        "refrakt_core.registry.dataset_registry.get_dataset", mock_get_dataset
    )
    monkeypatch.setattr(
        "refrakt_core.registry.dataset_registry.DATASET_REGISTRY",
        {"dummy": DummyBaseDataset, "contrastive": ContrastiveDataset},
    )
    monkeypatch.setattr(
        "refrakt_core.registry.transform_registry.get_transform", mock_get_transform
    )
    monkeypatch.setattr(
        "refrakt_core.registry.transform_registry.TRANSFORM_REGISTRY",
        {"contrastive": lambda x: x},
    )
    monkeypatch.setattr("os.path.exists", mock_os_path_exists)
    monkeypatch.setattr("torch.load", mock_torch_load)


# Add a simple logger for testing
class DummyLogger:
    def __init__(self):
        self.logs = []

    def info(self, msg):
        self.logs.append(("info", msg))

    def warning(self, msg):
        self.logs.append(("warning", msg))

    def error(self, msg):
        self.logs.append(("error", msg))

    def log_config(self, config):
        self.logs.append(("config", config))


class TestCLIHelpers:
    # Smoke Test
    def test_import_cli_helpers(self):
        importlib.reload(cli_helpers)

    # Sanity Test
    def test_argument_parser_returns_parser(self):
        parser = cli_helpers._setup_argument_parser()
        assert isinstance(parser, argparse.ArgumentParser)
        args = parser.parse_args(["--config", "foo.yaml"])
        assert hasattr(args, "config")

    # Unit Tests
    def test_extract_overrides_combines_flags_and_positional(self, monkeypatch):
        args = argparse.Namespace(override=["foo=1", "bar=2"])
        monkeypatch.setattr(
            "refrakt_core.hooks.hyperparameter_override.extract_overrides_from_args",
            lambda rem: (["baz=3"], []),
        )
        result = cli_helpers._extract_overrides(args, ["baz=3"])
        assert set(result) == {"foo=1", "bar=2", "baz=3"}

    def test_apply_config_overrides_applies_overrides(self, monkeypatch):
        class DummyOmegaConf:
            @staticmethod
            def to_container(cfg, resolve=True):
                return {"dataloader": {"params": {"batch_size": 32}}}

            @staticmethod
            def create(cfg):
                return cfg

        monkeypatch.setattr(cli_helpers, "OmegaConf", DummyOmegaConf)
        monkeypatch.setattr(
            "refrakt_core.hooks.hyperparameter_override.apply_overrides",
            lambda cfg, overrides: {"dataloader": {"params": {"batch_size": 64}}},
        )
        cfg = DictConfig({"dataloader": {"params": {"batch_size": 32}}})
        out = cli_helpers._apply_config_overrides(
            cfg, ["dataloader.params.batch_size=64"]
        )
        assert out["dataloader"]["params"]["batch_size"] == 64

    def test_extract_runtime_config_returns_dict(self, monkeypatch):
        class DummyOmegaConf:
            @staticmethod
            def to_container(cfg, resolve=True):
                return {"runtime": {"mode": "train"}}

        monkeypatch.setattr(cli_helpers, "OmegaConf", DummyOmegaConf)
        cfg = DictConfig({"runtime": {"mode": "train"}})
        out = cli_helpers._extract_runtime_config(cfg)
        assert out == {"mode": "train"}

    def test_extract_runtime_config_typeerror(self, monkeypatch):
        class DummyOmegaConf:
            @staticmethod
            def to_container(cfg, resolve=True):
                return 42

        monkeypatch.setattr(cli_helpers, "OmegaConf", DummyOmegaConf)
        cfg = DictConfig({})
        with pytest.raises(TypeError):
            cli_helpers._extract_runtime_config(cfg)

    def test_setup_logging_config_defaults(self):
        runtime_cfg = {}
        result = cli_helpers._setup_logging_config(runtime_cfg)
        assert result[0] == "train"
        assert result[1] == "./logs"
        assert isinstance(result[2], list)
        assert result[3] is True
        assert result[4] is None
        assert result[5] is False

    def test_setup_logging_config_with_args(self):
        runtime_cfg = {
            "mode": "test",
            "log_dir": "/tmp",
            "log_type": "file",
            "console": False,
            "model_path": "foo.pth",
            "debug": True,
        }
        result = cli_helpers._setup_logging_config(
            runtime_cfg, args_log_dir="/override"
        )
        assert result[0] == "test"
        assert result[1] == "/override"
        assert result[2] == ["file"]
        assert result[3] is False
        assert result[4] == "foo.pth"
        assert result[5] is True

    def test_execute_pipeline_mode_inference_missing_model_path(self):
        cfg = DictConfig(
            {
                "model": {"name": "resnet18"},
                "dataset": {
                    "name": "contrastive",
                    "params": {
                        "base_dataset": {"name": "dummy", "params": {"size": 10}}
                    },
                },
                "dataloader": {
                    "params": {"batch_size": 32, "shuffle": False, "num_workers": 0}
                },
                "loss": {"name": "nt_xent", "params": {"temperature": 0.5}},
                "trainer": {
                    "name": "contrastive",
                    "params": {
                        "save_dir": "./checkpoints",
                        "grad_log_interval": 100,
                        "param_log_interval": 500,
                    },
                },
                "optimizer": {
                    "name": "adam",
                    "params": {"lr": 1e-3, "weight_decay": 1e-5},
                },
            }
        )
        with pytest.raises(ValueError):
            cli_helpers._execute_pipeline_mode("inference", cfg, "", None)
