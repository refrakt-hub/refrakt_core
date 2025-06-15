"""Logger utility for Refrakt: console, file, WandB, TensorBoard logging."""

import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from refrakt_core.utils.methods import extract_visual_tensor
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

        self.log_file = os.path.join(log_dir, f"{timestamp}.log")
        self.log_dir = log_dir
        self.console = console
        self.debug_enabled = debug
        self.log_types = log_types or []

        self.logger = self._initialize_logger(timestamp)
        self.tb_writer = None
        self.wandb_run = None

        if "tensorboard" in self.log_types:
            self._init_tensorboard()
        if "wandb" in self.log_types:
            self._init_wandb()

    def _initialize_logger(self, timestamp: str) -> logging.Logger:
        logger = logging.getLogger(f"refrakt:{timestamp}")
        level = logging.DEBUG if self.debug_enabled else logging.INFO
        logger.setLevel(level)
        logger.propagate = False

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        if self.console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(console_handler)

        return logger

    def _init_wandb(self) -> None:
        try:
            import wandb
            self.wandb_run = wandb.init(project="refrakt", dir=self.log_dir)
            self.info("WandB initialized")
        except ImportError:
            self.error("WandB not installed.")
        except Exception as e:
            self.error(f"WandB init failed: {e}")

    def _init_tensorboard(self) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_path = os.path.join(self.log_dir, "tensorboard")
            os.makedirs(tb_path, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=tb_path)
            self.info(f"TensorBoard initialized at {tb_path}")
        except Exception as e:
            self.error(f"TensorBoard init failed: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        if self.tb_writer:
            for k, v in metrics.items():
                self.tb_writer.add_scalar(k, v, step)
        if self.wandb_run:
            self.wandb_run.log(metrics, step=step)

    def log_config(self, config: Dict[str, Union[int, float, str, bool, Tensor]]) -> None:
        if self.wandb_run:
            self.wandb_run.config.update(config)
        if self.tb_writer:
            try:
                from torch.utils.tensorboard.summary import hparams
                cfg = flatten_and_filter_config(config)
                exp, ssi, sei = hparams(cfg, {})
                self.tb_writer.file_writer.add_summary(exp)
                self.tb_writer.file_writer.add_summary(ssi)
                self.tb_writer.file_writer.add_summary(sei)
            except Exception as e:
                self.error(f"TensorBoard hparams failed: {e}")

    def log_model_graph(self, model: nn.Module, input_tensor: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> None:
        if self.tb_writer:
            try:
                # Automatically move tensors to model device
                device = next(model.parameters()).device

                if isinstance(input_tensor, dict):
                    input_tensor = {k: v.to(device) for k, v in input_tensor.items()}
                else:
                    input_tensor = input_tensor.to(device)

                self.tb_writer.add_graph(model, input_tensor)
            except Exception as e:
                self.error(f"Model graph logging failed: {e}")


    def log_images(self, tag: str, images: Union[Tensor, np.ndarray], step: int, dataformats: str = "NCHW") -> None:
        if isinstance(images, Tensor):
            images = images.detach().cpu().numpy()
        if images.ndim != 4:
            self.warning(f"Expected 4D image tensor, got {images.shape}")
            return

        if dataformats == "NCHW":
            images = np.transpose(images, (0, 2, 3, 1))

        if self.tb_writer:
            self.tb_writer.add_images(tag, torch.tensor(np.transpose(images, (0, 3, 1, 2))), step, dataformats="NCHW")
        if self.wandb_run:
            try:
                import wandb
                self.wandb_run.log({tag: [wandb.Image(img) for img in images]}, step=step)
            except Exception as e:
                self.error(f"WandB image logging failed: {e}")

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


    # Logging levels
    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        if self.debug_enabled:
            self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.error(msg, *args, **kwargs)

    def close(self) -> None:
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
        if self.tb_writer:
            self.tb_writer.close()
        if self.wandb_run:
            self.wandb_run.finish()
