"""Logger utility for Refrakt: console, file, WandB, TensorBoard logging."""

import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from refrakt_core.api.core.utils import flatten_and_filter_config
from refrakt_core.utils.methods import extract_visual_tensor
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
            log_types (Optional[List[str]], optional): Types of logging to enable. Defaults to None.
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
        """
        Initialize TensorBoard logging if available.
        """
        try:
            from torch.utils.tensorboard.writer import SummaryWriter

            tb_path = os.path.join(
                self.log_dir, "tensorboard", datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            os.makedirs(tb_path, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=tb_path)
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
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, "logits") and isinstance(output.logits, torch.Tensor):
            return output.logits
        if hasattr(output, "reconstruction") and isinstance(
            output.reconstruction, torch.Tensor
        ):
            return output.reconstruction
        for attr in dir(output):
            if not attr.startswith("_"):
                val = getattr(output, attr)
                if isinstance(val, torch.Tensor):
                    return val
        return None

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
        if not hasattr(self, "_logged_metrics"):
            self._logged_metrics = set()

        # Create metrics to log, checking for duplicates
        metrics_to_log = {}
        for metric_name, value in metrics.items():
            # Apply prefix only once
            full_metric_name = f"{prefix}/{metric_name}" if prefix else metric_name

            # Create unique ID for this metric at this step
            metric_id = (full_metric_name, step)

            if metric_id not in self._logged_metrics:
                self._logged_metrics.add(metric_id)
                metrics_to_log[metric_name] = value
            else:
                self.debug(
                    f"[RefraktLogger] Skipping duplicate metric '{full_metric_name}' at step {step}"
                )

        # Only log if we have metrics to log
        if not metrics_to_log:
            return

        # Log to TensorBoard
        if self.tb_writer:
            for k, v in metrics_to_log.items():
                full_k = f"{prefix}/{k}" if prefix else k
                self.tb_writer.add_scalar(full_k, v, step)

        # Log to WandB
        if self.wandb_run:
            log_data = {
                f"{prefix}/{k}" if prefix else k: v for k, v in metrics_to_log.items()
            }
            self.wandb_run.log(log_data, step=step)

    def log_config(self, config: Dict[str, Any]) -> None:
        """
        Log configuration to WandB and TensorBoard, handling complex types.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
        """
        if self.wandb_run:
            self.wandb_run.config.update(config)

        if self.tb_writer:
            try:
                # Create a clean scalar-only config by flattening and filtering
                scalar_config = {}
                for k, v in flatten_and_filter_config(config).items():
                    # Handle different value types
                    if isinstance(v, (int, float, str, bool)):
                        scalar_config[k] = v
                    elif torch.is_tensor(v) and v.numel() == 1:
                        scalar_config[k] = v.item()
                    elif isinstance(v, (list, tuple)) and len(v) == 1:
                        scalar_config[k] = v[0]
                    elif (
                        not isinstance(v, (torch.Tensor, list, tuple))
                        and hasattr(v, "summary")
                        and callable(getattr(v, "summary", None))
                    ):
                        summary = v.summary()
                        if isinstance(summary, dict):
                            for sk, sv in summary.items():
                                scalar_config[f"{k}/{sk}"] = sv

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
            input_tensor (Union[torch.Tensor, Dict[str, torch.Tensor]]): Input for tracing.
            model_output (Optional[Any], optional): Model output for graph extraction. Defaults to None.
        """
        if not isinstance(model, nn.Module):
            self.warning("Model graph logging skipped: model is not nn.Module.")
            return

        try:
            device = next(model.parameters()).device
            if isinstance(input_tensor, dict):
                input_tensor = {k: v.to(device) for k, v in input_tensor.items()}
            else:
                input_tensor = input_tensor.to(device)

            if self.tb_writer:
                try:
                    if (
                        hasattr(model, "__class__")
                        and "FusionBlock" in model.__class__.__name__
                    ):
                        self.info(
                            "Skipping TensorBoard graph logging for FusionBlock (complex model structure)"
                        )
                        return

                    # Create a tracing module that wraps the original model
                    class TracingModel(nn.Module):
                        def __init__(self, model: nn.Module) -> None:
                            super().__init__()
                            self.model = model

                        def forward(self, x: Any) -> torch.Tensor:
                            # Use forward_for_graph if available
                            if hasattr(self.model, "forward_for_graph"):
                                return self.model.forward_for_graph(x)
                            # Otherwise extract tensor from regular output
                            output = self.model(x)
                            return self._extract_tensor(output)

                        @staticmethod
                        def _extract_tensor(output: Any) -> torch.Tensor:
                            """Extract a tensor from ModelOutput or raw output"""
                            if isinstance(output, torch.Tensor):
                                return output
                            if hasattr(output, "logits") and isinstance(
                                output.logits, torch.Tensor
                            ):
                                return output.logits
                            if hasattr(output, "reconstruction") and isinstance(
                                output.reconstruction, torch.Tensor
                            ):
                                return output.reconstruction
                            # Try to find any tensor in output
                            for attr in dir(output):
                                if not attr.startswith("_") and isinstance(
                                    getattr(output, attr), torch.Tensor
                                ):
                                    return getattr(output, attr)
                            raise ValueError(
                                "No tensor found in model output for tracing"
                            )

                    tracing_model = TracingModel(model)
                    tracing_model.eval()
                    self.tb_writer.add_graph(tracing_model, input_tensor)
                    self.info("Logged model graph to TensorBoard.")
                except Exception as e:
                    self.warning(f"TensorBoard model graph logging failed: {e}")

            if self.wandb_run:
                try:
                    import wandb

                    self.wandb_run.watch(model, log="all", log_freq=100)
                    self.info("WandB is watching model and gradients.")
                except Exception as e:
                    self.error(f"WandB model watching failed: {e}")
        except Exception as e:
            self.error(f"Model graph logging failed: {e}")

    def _to_wandb_image(self, img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        if isinstance(img, list):
            img = np.array(img)
        if isinstance(img, np.ndarray):
            # If shape is (C, H, W), convert to (H, W, C)
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                img = np.transpose(img, (1, 2, 0))
        return img

    def log_images(
        self,
        tag: str,
        images: Union[Tensor, np.ndarray],
        step: int,
        dataformats: str = "NCHW",
    ) -> None:
        """
        Log images to TensorBoard and WandB.

        Args:
            tag (str): Tag for the images.
            images (Union[Tensor, np.ndarray]): Images to log.
            step (int): Training step or epoch.
            dataformats (str, optional): Data format. Defaults to "NCHW".
        """
        if isinstance(images, Tensor):
            images = images.detach().cpu().numpy()
        # Convert ndarray to list if needed for Sequence
        if isinstance(images, np.ndarray):
            images_seq = images.tolist()
        else:
            images_seq = images
        if self.tb_writer:
            self.tb_writer.add_images(
                tag, np.array(images_seq), step, dataformats=dataformats
            )
        if self.wandb_run:
            import wandb

            self.wandb_run.log(
                {tag: [wandb.Image(self._to_wandb_image(img)) for img in images_seq]},
                step=step,
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
            import torch.nn.functional as F

            # Convert tensors to numpy arrays and then to lists for Sequence compatibility
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
