import importlib

import numpy as np
import pytest
import torch

from refrakt_core.api.core.utils import logger_helpers


class DummyLogger:
    pass


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)


class DummyTBWriter:
    def __init__(self):
        self.logged = []

    def add_scalar(self, k, v, step):
        self.logged.append((k, v, step))


class DummyWandbRun:
    def __init__(self):
        self.logged = []

    def log(self, data, step=None):
        self.logged.append((data, step))


class TestLoggerHelpers:
    # Smoke Tests
    def test_import_logger_helpers(self):
        importlib.reload(logger_helpers)

    def test_logger_helpers_has_any_function(self):
        funcs = [
            f
            for f in dir(logger_helpers)
            if callable(getattr(logger_helpers, f)) and not f.startswith("__")
        ]
        assert funcs

    # Sanity Tests
    def test_initialize_logged_metrics(self):
        logger = DummyLogger()
        s = logger_helpers._initialize_logged_metrics(logger)
        assert isinstance(s, set)
        # Should be idempotent
        s2 = logger_helpers._initialize_logged_metrics(logger)
        assert s is s2

    def test_create_metrics_to_log(self):
        metrics = {"acc": 0.9, "loss": 0.1}
        step = 1
        prefix = "train"
        logged_metrics = set()
        out = logger_helpers._create_metrics_to_log(
            metrics, step, prefix, logged_metrics
        )
        assert "acc" in out and "loss" in out
        # Duplicates should be filtered
        out2 = logger_helpers._create_metrics_to_log(
            metrics, step, prefix, logged_metrics
        )
        assert out2 == {}

    def test_log_to_tensorboard(self):
        tb_writer = DummyTBWriter()
        metrics = {"acc": 0.9}
        logger_helpers._log_to_tensorboard(tb_writer, metrics, 1, "train")
        assert tb_writer.logged[0][0] == "train/acc"

    def test_log_to_wandb(self):
        wandb_run = DummyWandbRun()
        metrics = {"acc": 0.9}
        logger_helpers._log_to_wandb(wandb_run, metrics, 1, "train")
        assert "train/acc" in wandb_run.logged[0][0]

    def test_prepare_input_tensor_for_graph_tensor(self):
        model = DummyModel()
        x = torch.ones(1, 2)
        out = logger_helpers._prepare_input_tensor_for_graph(model, x)
        assert torch.equal(out, x)

    def test_prepare_input_tensor_for_graph_dict(self):
        model = DummyModel()
        x = {"a": torch.ones(1, 2)}
        out = logger_helpers._prepare_input_tensor_for_graph(model, x)
        assert torch.equal(out["a"], x["a"])

    def test_should_skip_fusion_block_logging(self):
        class FusionBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()

        model = FusionBlock()
        assert logger_helpers._should_skip_fusion_block_logging(model)
        model2 = DummyModel()
        assert not logger_helpers._should_skip_fusion_block_logging(model2)
