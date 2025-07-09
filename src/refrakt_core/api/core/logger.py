"""Logger utility for Refrakt: console, file, WandB, TensorBoard logging."""

import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch import Tensor, nn


class RefraktLogger:
    """
    Logger class for handling console, file, WandB, and TensorBoard logging.

    Attributes:
        log_file (str): Path to the log file.
        log_dir (str): Directory for logs.
        console (bool): Whether to log to console.
        debug_enabled (bool): Whether debug logging is enabled.
        log_types (List[str]): Types of logging enabled (e.g., 'tensorboard', 'wandb').
        logger (logging.Logger): The Python logger instance.
        tb_writer (Optional[Any]): TensorBoard SummaryWriter instance.
        wandb_run (Optional[Any]): Weights & Biases run instance.
    """

    def __init__(
        self,
        model_name: str,
        log_dir: str = "./logs",
        log_types: Optional[List[str]] = None,
        console: bool = False,
        debug: bool = False,
    ) -> None:
        """
        Initialize the RefraktLogger.

        Args:
            model_name (str): Name of the model for logging context.
            log_dir (str, optional): Directory for logs. Defaults to './logs'.
            log_types (Optional[List[str]], optional): Types of logging to enable. \
                Defaults to None.
            console (bool, optional): Whether to log to console. Defaults to False.
            debug (bool, optional): Enable debug logging. Defaults to False.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(log_dir, model_name)
        os.makedirs(log_dir, exist_ok=True)

        self.log_file: str = os.path.join(log_dir, f"{timestamp}.log")
        self.log_dir: str = log_dir
        self.console: bool = console
        self.debug_enabled: bool = debug
        self.log_types: List[str] = log_types or []

        self.logger: logging.Logger = self._initialize_logger(timestamp)
        self.tb_writer: Optional[Any] = None
        self.wandb_run: Optional[Any] = None
        self._logged_metrics: set[tuple[str, int]] = set()

        if "tensorboard" in self.log_types:
            self._init_tensorboard()
        if "wandb" in self.log_types:
            self._init_wandb()

    def _initialize_logger(self, timestamp: str) -> logging.Logger:
        """
        Initialize the Python logger for file and console output.

        Args:
            timestamp (str): Timestamp for log file naming.

        Returns:
            logging.Logger: Configured logger instance.
        """
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
        """
        Initialize Weights & Biases (wandb) logging if available.
        """
        try:
            import wandb

            self.wandb_run = wandb.init(project="refrakt", dir=self.log_dir)
            self.info("WandB initialized")
        except ImportError:
            self.error("WandB not installed.")
        except Exception as e:
            self.error(f"WandB init failed: {e}")

    def _init_tensorboard(self) -> None:
        """Initialize TensorBoard writer."""
        try:
            from torch.utils.tensorboard.writer import SummaryWriter

            tb_path = os.path.join(
                self.log_dir, "tensorboard", datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            os.makedirs(tb_path, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=tb_path)  # type: ignore[no-untyped-call]
            self.info(f"TensorBoard initialized at {tb_path}")
        except Exception as e:
            self.error(f"TensorBoard init failed: {e}")

    def _extract_tensor_from_model_output(self, output: Any) -> Optional[Tensor]:
        """
        Extract a tensor from ModelOutput or raw output for logging compatibility.
        Prioritizes 'logits', 'reconstruction', or any available tensor field.

        Args:
            output (Any): Model output or tensor.

        Returns:
            Optional[Tensor]: Extracted tensor or None if not found.
        """
        from .utils.logging_utils import extract_tensor_from_model_output

        return extract_tensor_from_model_output(output)

    def log_metrics(
        self, metrics: Dict[str, float], step: int, prefix: Optional[str] = None
    ) -> None:
        """
        Log scalar metrics to TensorBoard and WandB, avoiding duplicates.

        Args:
            metrics (Dict[str, float]): Metrics to log.
            step (int): Training step or epoch.
            prefix (Optional[str], optional): Prefix for metric names. Defaults to None.
        """
        from .utils.logger_helpers import (
            _create_metrics_to_log,
            _initialize_logged_metrics,
            _log_to_tensorboard,
            _log_to_wandb,
        )

        logged_metrics = _initialize_logged_metrics(self)
        metrics_to_log = _create_metrics_to_log(metrics, step, prefix, logged_metrics)

        # Only log if we have metrics to log
        if not metrics_to_log:
            return

        # Log to TensorBoard and WandB
        _log_to_tensorboard(self.tb_writer, metrics_to_log, step, prefix)
        _log_to_wandb(self.wandb_run, metrics_to_log, step, prefix)

    def log_config(self, config: Dict[str, Any]) -> None:
        """
        Log configuration to WandB and TensorBoard, handling complex types.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
        """
        from .utils.logging_utils import create_scalar_config

        if self.wandb_run:
            self.wandb_run.config.update(config)

        if self.tb_writer:
            try:
                # Create a clean scalar-only config
                scalar_config = create_scalar_config(config)

                # Add placeholder metric for TensorBoard requirements
                metric_dict = {"placeholder": 0.0}

                # Create hparams summary
                self.tb_writer.add_hparams(scalar_config, metric_dict)

                # Add config as text for visibility
                config_text = "\n".join([f"{k}: {v}" for k, v in scalar_config.items()])
                self.tb_writer.add_text("config", config_text, 0)

                # Flush to ensure immediate write
                self.tb_writer.flush()
            except Exception as e:
                self.error(f"TensorBoard hparams logging failed: {e}")

    def log_model_graph(
        self,
        model: nn.Module,
        input_tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
        model_output: Optional[Any] = None,
    ) -> None:
        """
        Log the model graph to TensorBoard if possible.

        Args:
            model (nn.Module): The model to log.
            input_tensor (Union[torch.Tensor, Dict[str, torch.Tensor]]): Input for \
                tracing.
            model_output (Optional[Any], optional): Model output for graph extraction. \
                Defaults to None.
        """
        from .utils.logger_helpers import (
            _log_to_tensorboard_graph,
            _log_to_wandb_watch,
            _prepare_input_tensor_for_graph,
        )

        try:
            input_tensor = _prepare_input_tensor_for_graph(model, input_tensor)
            _log_to_tensorboard_graph(self.tb_writer, model, input_tensor, self)
            _log_to_wandb_watch(self.wandb_run, model, self)
        except Exception as e:
            self.error(f"Model graph logging failed: {e}")

    def _to_wandb_image(self, img: Any) -> Any:
        from .utils.logging_utils import convert_to_wandb_image

        return convert_to_wandb_image(img)

    def log_images(
        self,
        tag: str,
        images: Union[Tensor, np.ndarray[Any, Any]],
        step: int,
        dataformats: str = "NCHW",
    ) -> None:
        """
        Log images to TensorBoard and WandB.

        Args:
            tag (str): Tag for the images.
            images (Union[Tensor, np.ndarray[Any, Any]]): Images to log.
            step (int): Training step or epoch.
            dataformats (str, optional): Data format. Defaults to "NCHW".
        """
        from .utils.logger_helpers import (
            _log_images_to_tensorboard,
            _log_images_to_wandb,
            _prepare_images_for_logging,
        )

        images_seq = _prepare_images_for_logging(images)
        _log_images_to_tensorboard(self.tb_writer, tag, images_seq, step, dataformats)
        _log_images_to_wandb(
            self.wandb_run, tag, images_seq, step, self._to_wandb_image
        )

    def log_inference_results(
        self,
        inputs: Tensor,
        outputs: Tensor,
        targets: Optional[Tensor] = None,
        step: int = 0,
        max_images: int = 8,
    ) -> None:
        """
        Log inference results (inputs, outputs, targets) as images.

        Args:
            inputs (Tensor): Input images.
            outputs (Tensor): Output images.
            targets (Optional[Tensor], optional): Target images. Defaults to None.
            step (int, optional): Step for logging. Defaults to 0.
            max_images (int, optional): Maximum number of images to log. Defaults to 8.
        """
        try:

            # Convert tensors to numpy arrays and then to lists for \
            # Sequence compatibility
            in_imgs = inputs.detach().cpu().numpy()
            out_imgs = outputs.detach().cpu().numpy()
            in_imgs_seq = in_imgs.tolist()
            out_imgs_seq = out_imgs.tolist()
            if targets is not None:
                tgt_imgs = targets.detach().cpu().numpy()
                tgt_imgs_seq = tgt_imgs.tolist()
            else:
                tgt_imgs_seq = None
            # Only log up to max_images
            in_imgs_seq = in_imgs_seq[:max_images]
            out_imgs_seq = out_imgs_seq[:max_images]
            if tgt_imgs_seq is not None:
                tgt_imgs_seq = tgt_imgs_seq[:max_images]
            # Log images
            self.log_images("inference/inputs", np.array(in_imgs_seq), step)
            self.log_images("inference/outputs", np.array(out_imgs_seq), step)
            if tgt_imgs_seq is not None:
                self.log_images("inference/targets", np.array(tgt_imgs_seq), step)
            # Also log to wandb directly for Sequence compatibility
            if self.wandb_run:
                import wandb

                self.wandb_run.log(
                    {
                        "inference/inputs": [
                            wandb.Image(self._to_wandb_image(img))
                            for img in in_imgs_seq
                        ],
                        "inference/outputs": [
                            wandb.Image(self._to_wandb_image(img))
                            for img in out_imgs_seq
                        ],
                        **(
                            {
                                "inference/targets": [
                                    wandb.Image(self._to_wandb_image(img))
                                    for img in tgt_imgs_seq
                                ]
                            }
                            if tgt_imgs_seq is not None
                            else {}
                        ),
                    },
                    step=step,
                )
        except Exception as err:
            self.error(f"Inference visualization failed: {str(err)}")

    def log_parameters(self, model: nn.Module, step: int, prefix: str = "") -> None:
        """
        Log model parameters to TensorBoard and WandB.

        Args:
            model (nn.Module): Model whose parameters to log.
            step (int): Training step or epoch.
            prefix (str, optional): Prefix for parameter names. Defaults to "".
        """
        if self.tb_writer or self.wandb_run:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    full_name = f"{prefix}parameters/{name}"
                    param_data = param.data.cpu().detach()

                    # TensorBoard
                    if self.tb_writer:
                        # Log as histogram (requires flattened data)
                        self.tb_writer.add_histogram(
                            full_name, param_data.flatten(), step
                        )

                    # WandB
                    if self.wandb_run:
                        import wandb

                        wandb.log(
                            {
                                full_name: wandb.Histogram(
                                    param_data.cpu().numpy().flatten().tolist()
                                )
                            },
                            step=step,
                        )

    def log_gradients(self, model: nn.Module, step: int, prefix: str = "") -> None:
        """
        Log model gradients to TensorBoard and WandB.

        Args:
            model (nn.Module): Model whose gradients to log.
            step (int): Training step or epoch.
            prefix (str, optional): Prefix for gradient names. Defaults to "".
        """
        if self.tb_writer or self.wandb_run:
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    full_name = f"{prefix}gradients/{name}"
                    grad_data = param.grad.cpu().detach()

                    # TensorBoard
                    if self.tb_writer:
                        # Log as histogram (requires flattened data)
                        self.tb_writer.add_histogram(
                            full_name, grad_data.flatten(), step
                        )

                    # WandB
                    if self.wandb_run:
                        import wandb

                        wandb.log(
                            {
                                full_name: wandb.Histogram(
                                    grad_data.cpu().numpy().flatten().tolist()
                                )
                            },
                            step=step,
                        )

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a debug message."""
        if self.debug_enabled:
            self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an info message."""
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a warning message."""
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an error message."""
        self.logger.error(msg, *args, **kwargs)

    def close(self) -> None:
        """
        Close the logger and any open handlers (TensorBoard, WandB).
        """
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
        if self.tb_writer:
            self.tb_writer.close()
        if self.wandb_run:
            self.wandb_run.finish()
