import importlib

import numpy as np
import pytest
import torch

from refrakt_core.api.core.utils import logging_utils


class DummyOutput:
    def __init__(self, logits=None, reconstruction=None):
        self.logits = logits
        self.reconstruction = reconstruction
        self.foo = torch.tensor([42.0])


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)


class DummySummary:
    def summary(self):
        return {"a": 1, "b": 2}


class TestLoggingUtils:
    # Smoke Tests
    def test_import_logging_utils(self):
        importlib.reload(logging_utils)

    def test_logging_utils_has_any_function(self):
        funcs = [
            f
            for f in dir(logging_utils)
            if callable(getattr(logging_utils, f)) and not f.startswith("__")
        ]
        assert funcs

    # Sanity Tests
    def test_extract_tensor_from_model_output_tensor(self):
        t = torch.ones(2, 2)
        out = logging_utils.extract_tensor_from_model_output(t)
        assert out is not None and torch.equal(out, t)

    def test_extract_tensor_from_model_output_logits(self):
        t = torch.ones(2, 2)
        out = logging_utils.extract_tensor_from_model_output(DummyOutput(logits=t))
        assert out is not None and torch.equal(out, t)

    def test_extract_tensor_from_model_output_reconstruction(self):
        t = torch.ones(2, 2)
        out = logging_utils.extract_tensor_from_model_output(
            DummyOutput(reconstruction=t)
        )
        assert out is not None and torch.equal(out, t)

    def test_extract_tensor_from_model_output_any_tensor(self):
        t = torch.ones(2, 2)
        out = logging_utils.extract_tensor_from_model_output(DummyOutput())
        assert out is not None and torch.equal(out, torch.tensor([42.0]))

    def test_handle_scalar_value(self):
        assert logging_utils._handle_scalar_value("a", 1) == {"a": 1}
        assert logging_utils._handle_scalar_value("b", torch.tensor([2.0])) == {
            "b": 2.0
        }
        assert logging_utils._handle_scalar_value("c", [3]) == {"c": 3}
        assert logging_utils._handle_scalar_value("d", "foo") == {"d": "foo"}

    def test_handle_summary_object(self):
        obj = DummySummary()
        out = logging_utils._handle_summary_object("x", obj)
        assert "x/a" in out and "x/b" in out

    def test_create_scalar_config(self):
        config = {"a": 1, "b": torch.tensor([2.0]), "c": [3], "d": DummySummary()}
        out = logging_utils.create_scalar_config(config)
        # Only scalar values and tensors should be processed
        assert "a" in out and "b" in out
        # Lists and complex objects should be filtered out
        assert "c" not in out

    def test_create_tracing_model(self):
        model = DummyModel()
        tracing_model = logging_utils.create_tracing_model(model)
        x = torch.ones(1, 2)
        out = tracing_model(x)
        assert isinstance(out, torch.Tensor)

    def test_convert_to_wandb_image_tensor(self):
        t = torch.ones(3, 32, 32)
        img = logging_utils.convert_to_wandb_image(t)
        assert isinstance(img, np.ndarray)

    def test_convert_to_wandb_image_numpy(self):
        arr = np.ones((3, 32, 32))
        img = logging_utils.convert_to_wandb_image(arr)
        assert isinstance(img, np.ndarray)

    def test_convert_to_wandb_image_list(self):
        arr = [np.ones((3, 32, 32))]
        img = logging_utils.convert_to_wandb_image(arr)
        assert isinstance(img, np.ndarray)
