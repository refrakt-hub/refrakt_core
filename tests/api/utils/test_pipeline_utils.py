import importlib
import os
import zipfile
from contextlib import contextmanager

import pytest
from omegaconf import DictConfig

import src.refrakt_core.api.utils.pipeline_utils as pipeline_utils
from src.refrakt_core.api.core.logger import RefraktLogger


class DummyLogger(RefraktLogger):
    def info(self, msg):
        self.info_called = True

    def error(self, msg):
        self.error_called = True

    def warning(self, msg):
        self.warning_called = True

    def debug(self, msg):
        self.debug_called = True


class TestPipelineUtils:
    # Smoke Test
    def test_import_pipeline_utils(self):
        importlib.reload(pipeline_utils)

    # Sanity Tests
    def test_setup_logger_and_config_success(self):
        cfg = DictConfig({"model": {"name": "resnet"}})
        logger = pipeline_utils.setup_logger_and_config(
            cfg, "resnet", "./logs", ["file"], True, False, ["foo=1"]
        )
        assert type(logger).__name__ == "RefraktLogger"

    def test_setup_logger_and_config_typeerror(self):
        with pytest.raises(TypeError):
            pipeline_utils.setup_logger_and_config(
                42, "resnet", "./logs", ["file"], True, False, []
            )

    def test_setup_logger_and_config_valueerror(self):
        cfg = DictConfig({"model": {"name": "resnet"}})
        with pytest.raises(ValueError):
            pipeline_utils.setup_logger_and_config(
                cfg, "", "./logs", ["file"], True, False, []
            )
        with pytest.raises(ValueError):
            pipeline_utils.setup_logger_and_config(
                cfg, "resnet", "", ["file"], True, False, []
            )

    def test_execute_training_pipeline(self, monkeypatch, tmp_path):
        called = {}
        monkeypatch.setattr(
            pipeline_utils, "train", lambda *a, **kw: called.setdefault("train", True)
        )
        monkeypatch.setattr("pathlib.Path.exists", lambda self: True)
        monkeypatch.setattr("os.path.exists", lambda path: True)
        monkeypatch.setattr(os, "makedirs", lambda *a, **kw: None)

        @contextmanager
        def dummy_zipfile(*args, **kwargs):
            class DummyZip:
                def extractall(self, *a, **kw):
                    # Simulate extraction by doing nothing
                    pass

            yield DummyZip()

        monkeypatch.setattr(zipfile, "ZipFile", dummy_zipfile)
        # --- Patch directory and file structure ---
        extracted_dir = tmp_path / "refrakt_dataset_xxx"
        train_dir = extracted_dir / "train"
        val_dir = extracted_dir / "val"
        img1 = train_dir / "img1.png"
        img2 = train_dir / "img2.png"
        img3 = val_dir / "img3.png"
        fake_dirs = {
            str(extracted_dir): ["train", "val"],
            str(train_dir): ["img1.png", "img2.png"],
            str(val_dir): ["img3.png"],
        }
        fake_files = {
            str(img1),
            str(img2),
            str(img3),
        }
        monkeypatch.setattr(os, "listdir", lambda path: fake_dirs.get(path, []))
        monkeypatch.setattr(os.path, "isdir", lambda path: path in fake_dirs)
        monkeypatch.setattr(os.path, "isfile", lambda path: path in fake_files)
        # --- End patch ---
        logger = DummyLogger("resnet18", "./logs", ["file"], True, False)
        cfg = DictConfig(
            {
                "dataset": {
                    "name": "custom",
                    "params": {"zip_path": "dummy.zip"},
                    "wrapper": "contrastive",
                },
                "model": {"name": "resnet18"},
            }
        )
        pipeline_utils.execute_training_pipeline(cfg, "model.pth", logger)
        assert called["train"]

    def test_execute_testing_pipeline(self, monkeypatch, tmp_path):
        called = {}
        monkeypatch.setattr(
            pipeline_utils, "test", lambda *a, **kw: called.setdefault("test", True)
        )
        monkeypatch.setattr("pathlib.Path.exists", lambda self: True)
        monkeypatch.setattr("os.path.exists", lambda path: True)
        monkeypatch.setattr(os, "makedirs", lambda *a, **kw: None)

        @contextmanager
        def dummy_zipfile(*args, **kwargs):
            class DummyZip:
                def extractall(self, *a, **kw):
                    pass

            yield DummyZip()

        monkeypatch.setattr(zipfile, "ZipFile", dummy_zipfile)
        # --- Patch directory and file structure ---
        extracted_dir = tmp_path / "refrakt_dataset_xxx"
        train_dir = extracted_dir / "train"
        val_dir = extracted_dir / "val"
        img1 = train_dir / "img1.png"
        img2 = train_dir / "img2.png"
        img3 = val_dir / "img3.png"
        fake_dirs = {
            str(extracted_dir): ["train", "val"],
            str(train_dir): ["img1.png", "img2.png"],
            str(val_dir): ["img3.png"],
        }
        fake_files = {
            str(img1),
            str(img2),
            str(img3),
        }
        monkeypatch.setattr(os, "listdir", lambda path: fake_dirs.get(path, []))
        monkeypatch.setattr(os.path, "isdir", lambda path: path in fake_dirs)
        monkeypatch.setattr(os.path, "isfile", lambda path: path in fake_files)
        # --- End patch ---
        logger = DummyLogger("resnet18", "./logs", ["file"], True, False)
        cfg = DictConfig(
            {
                "dataset": {
                    "name": "custom",
                    "params": {"zip_path": "dummy.zip"},
                    "wrapper": "contrastive",
                },
                "model": {"name": "resnet18"},
            }
        )
        pipeline_utils.execute_testing_pipeline(cfg, "model.pth", logger)
        assert called["test"]

    def test_execute_inference_pipeline(self, monkeypatch, tmp_path):
        called = {}
        monkeypatch.setattr(
            pipeline_utils,
            "inference",
            lambda *a, **kw: called.setdefault("inference", True),
        )
        monkeypatch.setattr("pathlib.Path.exists", lambda self: True)
        monkeypatch.setattr("os.path.exists", lambda path: True)
        monkeypatch.setattr(os, "makedirs", lambda *a, **kw: None)

        @contextmanager
        def dummy_zipfile(*args, **kwargs):
            class DummyZip:
                def extractall(self, *a, **kw):
                    pass

            yield DummyZip()

        monkeypatch.setattr(zipfile, "ZipFile", dummy_zipfile)
        # --- Patch directory and file structure ---
        extracted_dir = tmp_path / "refrakt_dataset_xxx"
        train_dir = extracted_dir / "train"
        val_dir = extracted_dir / "val"
        img1 = train_dir / "img1.png"
        img2 = train_dir / "img2.png"
        img3 = val_dir / "img3.png"
        fake_dirs = {
            str(extracted_dir): ["train", "val"],
            str(train_dir): ["img1.png", "img2.png"],
            str(val_dir): ["img3.png"],
        }
        fake_files = {
            str(img1),
            str(img2),
            str(img3),
        }
        monkeypatch.setattr(os, "listdir", lambda path: fake_dirs.get(path, []))
        monkeypatch.setattr(os.path, "isdir", lambda path: path in fake_dirs)
        monkeypatch.setattr(os.path, "isfile", lambda path: path in fake_files)
        # --- End patch ---
        logger = DummyLogger("resnet18", "./logs", ["file"], True, False)
        cfg = DictConfig(
            {
                "dataset": {
                    "name": "custom",
                    "params": {"zip_path": "dummy.zip"},
                    "wrapper": "contrastive",
                },
                "model": {"name": "resnet18"},
            }
        )
        pipeline_utils.execute_inference_pipeline(cfg, "model.pth", logger)
        assert called["inference"]

    def test_execute_full_pipeline(self, monkeypatch, tmp_path):
        called = {"train": False, "test": False, "inference": False}
        monkeypatch.setattr(
            pipeline_utils, "train", lambda *a, **kw: called.update({"train": True})
        )
        monkeypatch.setattr(
            pipeline_utils, "test", lambda *a, **kw: called.update({"test": True})
        )
        monkeypatch.setattr(
            pipeline_utils,
            "inference",
            lambda *a, **kw: called.update({"inference": True}),
        )
        monkeypatch.setattr("pathlib.Path.exists", lambda self: True)
        monkeypatch.setattr("os.path.exists", lambda path: True)
        monkeypatch.setattr(os, "makedirs", lambda *a, **kw: None)

        @contextmanager
        def dummy_zipfile(*args, **kwargs):
            class DummyZip:
                def extractall(self, *a, **kw):
                    pass

            yield DummyZip()

        monkeypatch.setattr(zipfile, "ZipFile", dummy_zipfile)
        # --- Patch directory and file structure ---
        extracted_dir = tmp_path / "refrakt_dataset_xxx"
        train_dir = extracted_dir / "train"
        val_dir = extracted_dir / "val"
        img1 = train_dir / "img1.png"
        img2 = train_dir / "img2.png"
        img3 = val_dir / "img3.png"
        fake_dirs = {
            str(extracted_dir): ["train", "val"],
            str(train_dir): ["img1.png", "img2.png"],
            str(val_dir): ["img3.png"],
        }
        fake_files = {
            str(img1),
            str(img2),
            str(img3),
        }
        monkeypatch.setattr(os, "listdir", lambda path: fake_dirs.get(path, []))
        monkeypatch.setattr(os.path, "isdir", lambda path: path in fake_dirs)
        monkeypatch.setattr(os.path, "isfile", lambda path: path in fake_files)
        # --- End patch ---
        logger = DummyLogger("resnet18", "./logs", ["file"], True, False)
        cfg = DictConfig(
            {
                "dataset": {
                    "name": "custom",
                    "params": {"zip_path": "dummy.zip"},
                    "wrapper": "contrastive",
                },
                "model": {"name": "resnet18"},
                "trainer": {"params": {"save_dir": "./checkpoints"}},
            }
        )
        pipeline_utils.execute_full_pipeline(cfg, logger)
        assert all(called.values())

    def test_resolve_model_name_autoencoder(self):
        cfg = DictConfig(
            {
                "model": {"name": "autoencoder", "params": {"variant": "foo"}},
                "dataset": {"params": {}},
            }
        )
        name = pipeline_utils.resolve_model_name(cfg)
        assert name == "autoencoder_foo"

    def test_resolve_model_name_regular(self):
        cfg = DictConfig({"model": {"name": "resnet"}, "dataset": {"params": {}}})
        name = pipeline_utils.resolve_model_name(cfg)
        assert name == "resnet"

    def test_resolve_model_name_custom(self):
        cfg = DictConfig(
            {"model": {"name": "resnet"}, "dataset": {"params": {"path": "foo.zip"}}}
        )
        name = pipeline_utils.resolve_model_name(cfg)
        assert name == "resnet_custom"
