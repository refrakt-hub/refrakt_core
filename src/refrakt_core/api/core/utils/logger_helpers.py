"""
Logger helper functions for Refrakt.

This module contains internal helper functions used by the RefraktLogger class.
"""

import logging
import numpy as np
from typing import Any, Dict, Optional, Set, Tuple

import torch
from torch import Tensor, nn


def _initialize_logged_metrics(logger_instance) -> Set[Tuple[str, int]]:
    """Initialize the logged metrics set if it doesn't exist."""
    if not hasattr(logger_instance, "_logged_metrics"):
        logger_instance._logged_metrics = set()
    return logger_instance._logged_metrics


def _create_metrics_to_log(
    metrics: Dict[str, float], 
    step: int, 
    prefix: Optional[str], 
    logged_metrics: Set[Tuple[str, int]]
) -> Dict[str, float]:
    """Create metrics to log, checking for duplicates."""
    metrics_to_log = {}
    for metric_name, value in metrics.items():
        # Apply prefix only once
        full_metric_name = f"{prefix}/{metric_name}" if prefix else metric_name

        # Create unique ID for this metric at this step
        metric_id = (full_metric_name, step)

        if metric_id not in logged_metrics:
            logged_metrics.add(metric_id)
            metrics_to_log[metric_name] = value
        else:
            logging.debug(
                f"[RefraktLogger] Skipping duplicate metric '{full_metric_name}' at step {step}"
            )
    
    return metrics_to_log


def _log_to_tensorboard(tb_writer, metrics_to_log: Dict[str, float], step: int, prefix: Optional[str]) -> None:
    """Log metrics to TensorBoard."""
    if tb_writer:
        for k, v in metrics_to_log.items():
            full_k = f"{prefix}/{k}" if prefix else k
            tb_writer.add_scalar(full_k, v, step)


def _log_to_wandb(wandb_run, metrics_to_log: Dict[str, float], step: int, prefix: Optional[str]) -> None:
    """Log metrics to WandB."""
    if wandb_run:
        log_data = {
            f"{prefix}/{k}" if prefix else k: v for k, v in metrics_to_log.items()
        }
        wandb_run.log(log_data, step=step)


def _prepare_input_tensor_for_graph(
    model: nn.Module, 
    input_tensor: Any
) -> Any:
    """Prepare input tensor for model graph logging."""
    device = next(model.parameters()).device
    if isinstance(input_tensor, dict):
        input_tensor = {k: v.to(device) for k, v in input_tensor.items()}
    else:
        input_tensor = input_tensor.to(device)
    return input_tensor


def _should_skip_fusion_block_logging(model: nn.Module) -> bool:
    """Check if we should skip logging for FusionBlock models."""
    return (
        hasattr(model, "__class__")
        and "FusionBlock" in model.__class__.__name__
    )


def _log_to_tensorboard_graph(tb_writer, model: nn.Module, input_tensor: Any, logger_instance) -> None:
    """Log model graph to TensorBoard."""
    if tb_writer:
        try:
            if _should_skip_fusion_block_logging(model):
                logger_instance.info(
                    "Skipping TensorBoard graph logging for FusionBlock (complex model structure)"
                )
                return

            from .logging_utils import create_tracing_model
            tracing_model = create_tracing_model(model)
            tracing_model.eval()
            tb_writer.add_graph(tracing_model, input_tensor)
            logger_instance.info("Logged model graph to TensorBoard.")
        except Exception as e:
            logger_instance.warning(f"TensorBoard model graph logging failed: {e}")


def _log_to_wandb_watch(wandb_run, model: nn.Module, logger_instance) -> None:
    """Set up WandB model watching."""
    if wandb_run:
        try:
            import wandb
            wandb_run.watch(model, log="all", log_freq=100)
            logger_instance.info("WandB is watching model and gradients.")
        except Exception as e:
            logger_instance.error(f"WandB model watching failed: {e}")


def _prepare_images_for_logging(images: Any) -> Any:
    """Prepare images for logging by converting to appropriate format."""
    if isinstance(images, Tensor):
        images = images.detach().cpu().numpy()
    # Convert ndarray to list if needed for Sequence
    if isinstance(images, np.ndarray):
        images_seq = images.tolist()
    else:
        images_seq = images
    return images_seq


def _log_images_to_tensorboard(tb_writer, tag: str, images_seq: Any, step: int, dataformats: str) -> None:
    """Log images to TensorBoard."""
    if tb_writer:
        tb_writer.add_images(
            tag, np.array(images_seq), step, dataformats=dataformats
        )


def _log_images_to_wandb(wandb_run, tag: str, images_seq: Any, step: int, to_wandb_image_func) -> None:
    """Log images to WandB."""
    if wandb_run:
        import wandb
        wandb_run.log(
            {tag: [wandb.Image(to_wandb_image_func(img)) for img in images_seq]},
            step=step,
        ) 