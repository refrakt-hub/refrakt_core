import importlib
from types import SimpleNamespace

import pytest
import torch
from omegaconf import DictConfig

import refrakt_core.api.utils.inference_utils as inference_utils
from refrakt_core.api.core.logger import RefraktLogger


class DummyLogger(RefraktLogger):
    def info(self, msg):
        self.info_called = True


class DummyModel(torch.nn.Module):
    def forward(self, x):
        return x + 1


class DummyDataLoader:
    def __iter__(self):
        for i in range(3):
            yield torch.ones(2, 2)


class TestInferenceUtils:
    # Smoke Test
    def test_import_inference_utils(self):
        importlib.reload(inference_utils)

    # Sanity Tests
    def test_resolve_model_name_for_inference_autoencoder(self):
        cfg = DictConfig(
            {"model": {"name": "autoencoder", "params": {"variant": "foo"}}}
        )
        name = inference_utils.resolve_model_name_for_inference(cfg)
        assert name == "autoencoder_foo"

    def test_resolve_model_name_for_inference_regular(self):
        cfg = DictConfig({"model": {"name": "resnet"}})
        name = inference_utils.resolve_model_name_for_inference(cfg)
        assert name == "resnet"

    def test_extract_inputs_from_batch_tensor(self):
        batch = torch.ones(2, 2)
        out = inference_utils.extract_inputs_from_batch(batch)
        assert torch.equal(out, batch)

    def test_extract_inputs_from_batch_dict(self):
        batch = {"input": torch.ones(2, 2)}
        out = inference_utils.extract_inputs_from_batch(batch)
        assert torch.equal(out, batch["input"])

    def test_extract_inputs_from_batch_dict_no_tensor(self):
        batch = {"foo": 1}
        out = inference_utils.extract_inputs_from_batch(batch)
        assert out is None

    def test_extract_inputs_from_batch_other(self):
        batch = 42
        out = inference_utils.extract_inputs_from_batch(batch)
        assert out is None

    def test_run_inference_loop(self):
        model = DummyModel()
        loader = DummyDataLoader()
        results = inference_utils.run_inference_loop(model, loader)
        assert isinstance(results, list)
        assert len(results) == 3
        assert torch.equal(results[0], torch.ones(2, 2) + 1)
