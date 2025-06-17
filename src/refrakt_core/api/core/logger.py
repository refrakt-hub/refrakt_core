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
            from torch.utils.tensorboard.writer import SummaryWriter
            # Create unique directory using timestamp
            tb_path = os.path.join(self.log_dir, "tensorboard", 
                                datetime.now().strftime("%Y%m%d_%H%M%S"))
            os.makedirs(tb_path, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=tb_path)
            self.info(f"TensorBoard initialized at {tb_path}")
        except Exception as e:
            self.error(f"TensorBoard init failed: {e}")
    
    def _extract_tensor_from_model_output(self, output: Any) -> Optional[Tensor]:
        """
        Extracts a tensor from ModelOutput or raw output for logging compatibility.
        Prioritizes 'logits', 'reconstruction', or any available tensor field.
        """
        if isinstance(output, torch.Tensor):
            return output

        if hasattr(output, "logits") and isinstance(output.logits, torch.Tensor):
            return output.logits
        if hasattr(output, "reconstruction") and isinstance(output.reconstruction, torch.Tensor):
            return output.reconstruction

        # Check if any attribute is a tensor
        for attr in dir(output):
            if not attr.startswith("_"):
                val = getattr(output, attr)
                if isinstance(val, torch.Tensor):
                    return val
        return None
            
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: Optional[str] = None) -> None:
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
                self.debug(f"[RefraktLogger] Skipping duplicate metric '{full_metric_name}' at step {step}")

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
            log_data = {f"{prefix}/{k}" if prefix else k: v for k, v in metrics_to_log.items()}
            self.wandb_run.log(log_data, step=step)
            
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log configuration to WandB and TensorBoard, handling complex types."""
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
                    elif hasattr(v, 'summary') and callable(v.summary):
                        # Handle ModelOutput by using its summary
                        summary = v.summary()
                        for sk, sv in summary.items():
                            scalar_config[f"{k}/{sk}"] = sv
                    
                # Add placeholder metric for TensorBoard requirements
                metric_dict = {'placeholder': 0.0}
                
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
                    # Create a tracing module that wraps the original model
                    class TracingModel(nn.Module):
                        def __init__(self, model):
                            super().__init__()
                            self.model = model
                        
                        def forward(self, x):
                            # Use forward_for_graph if available
                            if hasattr(self.model, 'forward_for_graph'):
                                return self.model.forward_for_graph(x)
                            # Otherwise extract tensor from regular output
                            output = self.model(x)
                            return self._extract_tensor(output)
                        
                        @staticmethod
                        def _extract_tensor(output: Any) -> torch.Tensor:
                            """Extract a tensor from ModelOutput or raw output"""
                            if isinstance(output, torch.Tensor):
                                return output
                            if hasattr(output, "logits") and isinstance(output.logits, torch.Tensor):
                                return output.logits
                            if hasattr(output, "reconstruction") and isinstance(output.reconstruction, torch.Tensor):
                                return output.reconstruction
                            # Try to find any tensor in output
                            for attr in dir(output):
                                if not attr.startswith("_") and isinstance(getattr(output, attr), torch.Tensor):
                                    return getattr(output, attr)
                            raise ValueError("No tensor found in model output for tracing")

                    tracing_model = TracingModel(model)
                    tracing_model.eval()
                    self.tb_writer.add_graph(tracing_model, input_tensor)
                    self.info("Logged model graph to TensorBoard.")
                except Exception as e:
                    self.error(f"TensorBoard model graph logging failed: {e}")

            if self.wandb_run:
                try:
                    import wandb
                    self.wandb_run.watch(model, log="all", log_freq=100)
                    self.info("WandB is watching model and gradients.")
                except Exception as e:
                    self.error(f"WandB model watching failed: {e}")
        except Exception as e:
            self.error(f"Model graph logging failed: {e}")

    def log_images(self, tag: str, images: Union[Tensor, np.ndarray], step: int, dataformats: str = "NCHW") -> None:
        if isinstance(images, Tensor):
            images = images.detach().cpu().numpy()
        if images.ndim == 2:
            if isinstance(images, np.ndarray):
                images = images.reshape(-1, 1, 28, 28)
            else:
                images = images.view(-1, 1, 28, 28)
        elif images.ndim == 3:
            if isinstance(images, np.ndarray):
                images = np.expand_dims(images, axis=1)
            else:
                images = images.unsqueeze(1)
        elif images.ndim != 4:
            self.warning(f"Expected 2D–4D image tensor, got {images.shape}")
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

    def log_parameters(self, model: nn.Module, step: int, prefix: str = ""):
        """Log model parameters to TensorBoard and WandB"""
        if self.tb_writer or self.wandb_run:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    full_name = f"{prefix}parameters/{name}"
                    param_data = param.data.cpu().detach()
                    
                    # TensorBoard
                    if self.tb_writer:
                        # Log as histogram (requires flattened data)
                        self.tb_writer.add_histogram(full_name, param_data.flatten(), step)
                    
                    # WandB
                    if self.wandb_run:
                        import wandb
                        wandb.log({full_name: wandb.Histogram(param_data.numpy())}, step=step)

    def log_gradients(self, model: nn.Module, step: int, prefix: str = ""):
        """Log model gradients to TensorBoard and WandB"""
        if self.tb_writer or self.wandb_run:
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    full_name = f"{prefix}gradients/{name}"
                    grad_data = param.grad.cpu().detach()
                    
                    # TensorBoard
                    if self.tb_writer:
                        # Log as histogram (requires flattened data)
                        self.tb_writer.add_histogram(full_name, grad_data.flatten(), step)
                    
                    # WandB
                    if self.wandb_run:
                        import wandb
                        wandb.log({full_name: wandb.Histogram(grad_data.numpy())}, step=step)

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
