"""Logger utility for Refrakt: supports logging via console, files, WandB, and TensorBoard."""

import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from refrakt_core.api.core.utils import flatten_and_filter_config


class RefraktLogger:
    """Logger class for handling console, file, WandB, and TensorBoard logging."""

    def __init__(
        self,
        model_name: str,
        log_dir: str = "./logs",
        log_types: Optional[List[str]] = None,
        console: bool = False,
        debug: bool = False,
    ) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(log_dir, model_name)
        os.makedirs(log_dir, exist_ok=True)

        self.log_file: str = os.path.join(log_dir, f"{timestamp}.log")
        self.log_dir: str = log_dir
        self.log_types: List[str] = log_types or []
        self.console: bool = console
        self.wandb_run: Optional[Any] = None
        self.tb_writer: Optional[Any] = None
        self.debug_enabled: bool = debug

        self.logger: logging.Logger = logging.getLogger(f"refrakt:{timestamp}")
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
    ) -> None:
        """Initialize this logger from an existing logging.Logger object."""
        self.logger = existing_logger
        self.debug_enabled = debug
        self.console = console
        self.log_types = log_types or []
        self.log_dir = log_dir
        self.log_file = ""
        self.wandb_run = None
        self.tb_writer = None

    def _setup_handlers(self, level: int) -> None:
        """Set up logging handlers."""
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

    def _setup_wandb(self) -> None:
        """Set up WandB logging."""
        try:
            import wandb
            self.wandb_run = wandb.init(project="refrakt", dir=self.log_dir)
            self.info("Weights & Biases initialized")
        except ImportError:
            self.error("wandb not installed. Skipping WandB setup")
        except (RuntimeError, ValueError) as err:
            self.error(f"WandB initialization failed: {str(err)}")

    def _setup_tensorboard(self) -> None:
        """Set up TensorBoard logging."""
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_dir = os.path.join(self.log_dir, "tensorboard")
            os.makedirs(tb_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=tb_dir)
            self.info(f"TensorBoard initialized at {tb_dir}")
        except (RuntimeError, ValueError) as err:
            self.error(f"TensorBoard initialization failed: {str(err)}")

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log scalar metrics to available loggers."""
        if self.tb_writer:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, step)
        if self.wandb_run:
            self.wandb_run.log(metrics, step=step)

    def log_config(self, config: Dict[str, Union[int, float, str, bool, Tensor]]) -> None:
        """Log model configuration."""
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
            except (RuntimeError, ValueError) as err:
                self.error(f"Failed to log hparams to TensorBoard: {str(err)}")

    def log_model_graph(self, model: Module, input_tensor: Tensor) -> None:
        """Log model graph."""
        if self.tb_writer:
            try:
                self.tb_writer.add_graph(model, input_tensor)
                self.info("Logged model graph to TensorBoard")
            except (RuntimeError, ValueError) as err:
                self.error(f"Failed to log model graph: {str(err)}")

    def log_images(
        self,
        tag: str,
        images: Union[Tensor, np.ndarray],
        step: int,
        dataformats: str = "NCHW",
    ) -> None:
        """Log image tensors."""
        if isinstance(images, (Tensor, np.ndarray)) and images.ndim != 4:
            self.warning(
                f"Skipping image log for tag '{tag}': expected 4D input, got shape {images.shape}"
            )
            return

        if self.tb_writer:
            try:
                self.tb_writer.add_images(tag, images, step, dataformats=dataformats)
            except (RuntimeError, ValueError) as err:
                self.error(f"TensorBoard image logging failed: {str(err)}")

        if self.wandb_run:
            try:
                import wandb

                if isinstance(images, Tensor):
                    images = images.detach().cpu().numpy()
                if dataformats == "NCHW":
                    images = np.transpose(images, (0, 2, 3, 1))
                wandb_images = [wandb.Image(img) for img in images]
                self.wandb_run.log({tag: wandb_images}, step=step)
            except (RuntimeError, ValueError) as err:
                self.error(f"WandB image logging failed: {str(err)}")

    def log_inference_results(
        self,
        inputs: Tensor,
        outputs: Tensor,
        targets: Optional[Tensor] = None,
        step: int = 0,
        max_images: int = 8,
    ) -> None:
        """Visualize inference results with inputs, outputs, and targets."""
        try:
            import torch.nn.functional as F

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

            # Upsample inputs to match output/target resolution for side-by-side comparison
            if outputs.ndim == 4:
                target_size = outputs.shape[-2:]
                inputs_up = F.interpolate(inputs, size=target_size, mode="bicubic", align_corners=False)

                if targets is not None and targets.ndim == 4:
                    comparisons = torch.cat([inputs_up, outputs, targets], dim=0)
                    self.log_images("Comparison", comparisons, step)
                else:
                    comparisons = torch.cat([inputs_up, outputs], dim=0)
                    self.log_images("Input_vs_Output", comparisons, step)

            self.info(f"Logged inference visualization for {n} samples")

        except (RuntimeError, ValueError) as err:
            self.error(f"Inference visualization failed: {str(err)}")


    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log debug message."""
        if self.debug_enabled:
            self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log info message."""
        self.logger.info(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log error message."""
        self.logger.error(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log warning message."""
        self.logger.warning(msg, *args, **kwargs)

    def close(self) -> None:
        """Close logging resources."""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
        if self.tb_writer:
            self.tb_writer.close()
        if self.wandb_run:
            self.wandb_run.finish()
