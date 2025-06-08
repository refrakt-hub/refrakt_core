# logger.py

import logging
import os
import sys
from datetime import datetime
from typing import List, Optional, Union

import numpy as np
import torch
from torch import Tensor

from refrakt_core.api.core.utils import flatten_and_filter_config

# logger.py (only __init__ shown here)


class RefraktLogger:
    def __init__(
        self,
        model_name: str,
        base_log_dir: str = "./logs",
        log_types: Optional[List[str]] = None,
        console: bool = False,
        debug: bool = False,
    ):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(base_log_dir, model_name)
        os.makedirs(log_dir, exist_ok=True)

        self.log_file = os.path.join(log_dir, f"{timestamp}.log")
        self.log_dir = log_dir
        self.log_types = log_types or []
        self.console = console
        self.wandb_run = None
        self.tb_writer = None
        self.debug_enabled = debug

        self.logger = logging.getLogger(f"refrakt:{timestamp}")  # Unique per run
        level = logging.DEBUG if debug else logging.INFO
        self.logger.setLevel(level)

        self._setup_handlers(level)
        self.logger.propagate = False

        if "wandb" in self.log_types:
            self._setup_wandb()
        if "tensorboard" in self.log_types:
            self._setup_tensorboard()
            
    def init_from_existing(
        self,
        existing_logger: logging.Logger,
        *,
        log_dir: str = "./logs",
        log_types: Optional[List[str]] = None,
        console: bool = True,
        debug: bool = False,
    ):
        self.logger = existing_logger
        self.debug_enabled = debug
        self.console = console
        self.log_types = log_types or []
        self.log_dir = log_dir
        self.log_file = None  # Not used in this mode
        self.wandb_run = None
        self.tb_writer = None

    def _setup_handlers(self, level):
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(file_handler)

        if self.console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(console_handler)

    def _setup_wandb(self):
        try:
            import wandb

            self.wandb_run = wandb.init(project="refrakt", dir=self.log_dir)
            self.info("Weights & Biases initialized")
        except ImportError:
            self.error("wandb not installed. Skipping WandB setup")
        except Exception as e:
            self.error(f"WandB initialization failed: {str(e)}")

    def _setup_tensorboard(self):
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_dir = os.path.join(self.log_dir, "tensorboard")
            os.makedirs(tb_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=tb_dir)
            self.info(f"TensorBoard initialized at {tb_dir}")
        except Exception as e:
            self.error(f"TensorBoard initialization failed: {str(e)}")

    def log_metrics(self, metrics: dict, step: int):
        if self.tb_writer:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, step)
        if self.wandb_run:
            self.wandb_run.log(metrics, step=step)

    def log_config(self, config: dict):
        if self.wandb_run:
            self.wandb_run.config.update(config)
        if self.tb_writer:
            from torch.utils.tensorboard.summary import hparams

            try:
                filtered_config = flatten_and_filter_config(config)
                exp, ssi, sei = hparams(filtered_config, {})
                self.tb_writer.file_writer.add_summary(exp)
                self.tb_writer.file_writer.add_summary(ssi)
                self.tb_writer.file_writer.add_summary(sei)
                self.info("Logged filtered config to TensorBoard hparams")
            except Exception as e:
                self.error(f"Failed to log hparams to TensorBoard: {str(e)}")

    def log_model_graph(self, model: torch.nn.Module, input_tensor: Tensor):
        if self.tb_writer:
            try:
                self.tb_writer.add_graph(model, input_tensor)
                self.info("Logged model graph to TensorBoard")
            except Exception as e:
                self.error(f"Failed to log model graph: {str(e)}")

    def log_images(
        self,
        tag: str,
        images: Union[Tensor, np.ndarray],
        step: int,
        dataformats: str = "NCHW",
    ):
        # Skip if not 4D
        if isinstance(images, torch.Tensor) and images.ndim != 4:
            self.warning(
                f"Skipping image log for tag '{tag}': expected 4D tensor, got shape {images.shape}"
            )
            return
        if isinstance(images, np.ndarray) and images.ndim != 4:
            self.warning(
                f"Skipping image log for tag '{tag}': expected 4D array, got shape {images.shape}"
            )
            return

        if self.tb_writer:
            try:
                self.tb_writer.add_images(tag, images, step, dataformats=dataformats)
            except Exception as e:
                self.error(f"TensorBoard image logging failed: {str(e)}")

        if self.wandb_run:
            try:
                import wandb

                if isinstance(images, Tensor):
                    images = images.detach().cpu().numpy()
                if dataformats == "NCHW":
                    images = np.transpose(images, (0, 2, 3, 1))
                wandb_images = [wandb.Image(img) for img in images]
                self.wandb_run.log({tag: wandb_images}, step=step)
            except Exception as e:
                self.error(f"WandB image logging failed: {str(e)}")

    def log_inference_results(
        self,
        inputs: Tensor,
        outputs: Tensor,
        targets: Optional[Tensor] = None,
        step: int = 0,
        max_images: int = 8,
    ):
        try:
            n = min(inputs.shape[0], max_images)
            inputs = inputs[:n].cpu()
            outputs = outputs[:n].cpu()
            targets = targets[:n].cpu() if targets is not None else None

            if inputs.ndim == 4:
                self.log_images("Input", inputs, step)
            if outputs.ndim == 4:
                self.log_images("Output", outputs, step)
            if targets is not None and targets.ndim == 4:
                self.log_images("Target", targets, step)

            if targets is not None and all(
                t.ndim == 4 for t in [inputs, outputs, targets]
            ):
                comparisons = torch.cat([inputs, outputs, targets], dim=0)
                self.log_images("Comparison", comparisons, step)
            elif all(t.ndim == 4 for t in [inputs, outputs]):
                comparisons = torch.cat([inputs, outputs], dim=0)
                self.log_images("Input_vs_Output", comparisons, step)

            self.info(f"Logged inference visualization for {n} samples")
        except Exception as e:
            self.error(f"Inference visualization failed: {str(e)}")

    def debug(self, msg: str, *args, **kwargs):
        if self.debug_enabled:
            self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def close(self):
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
        if self.tb_writer:
            self.tb_writer.close()
        if self.wandb_run:
            self.wandb_run.finish()
