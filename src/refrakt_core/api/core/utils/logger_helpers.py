"""
Logger helper functions for Refrakt.

This module contains internal helper functions used by the
RefraktLogger class and related logging utilities.
It provides helpers for logging metrics, images, and model
graphs to TensorBoard and WandB, as well as for managing
logged metrics and preparing data for visualization.

The module handles:
- Metric deduplication and management
- TensorBoard and WandB logging operations
- Model graph logging and visualization
- Image preparation and logging
- Input tensor preparation for model tracing
- Fusion block logging special cases

These utilities ensure robust and consistent logging across different backends
while handling edge cases and providing appropriate error handling.

Typical usage involves calling these helper functions from the main RefraktLogger
class to perform specific logging operations.
"""

import logging
from typing import Any, Dict, Optional, Set, Tuple, cast

import numpy as np
from torch import Tensor, nn


def _initialize_logged_metrics(logger_instance: Any) -> Set[Tuple[str, int]]:
    """
    Initialize the logged metrics set if it doesn't exist.

    This function ensures that the logger instance has a set to track
    logged metrics for deduplication purposes.

    Args:
        logger_instance: Logger instance that may or may not have _logged_metrics

    Returns:
        Set of tuples containing (metric_name, step) for deduplication tracking
    """
    if not hasattr(logger_instance, "_logged_metrics"):
        logger_instance._logged_metrics = set()
    return cast(Set[Tuple[str, int]], logger_instance._logged_metrics)


def _create_metrics_to_log(
    metrics: Dict[str, float],
    step: int,
    prefix: Optional[str],
    logged_metrics: Set[Tuple[str, int]],
) -> Dict[str, float]:
    """
    Create metrics to log, checking for duplicates.

    This function filters metrics to avoid logging duplicates at the same step.
    It applies prefixes and maintains a set of logged metrics for deduplication.

    Args:
        metrics: Dictionary of metrics to potentially log
        step: Current training step number
        prefix: Optional prefix to apply to metric names
        logged_metrics: Set of already logged metrics for deduplication

    Returns:
        Dictionary of metrics to log, excluding duplicates

    Note:
        This function prevents duplicate logging of the same metric at the same step,
        which can happen in complex training loops with multiple logging calls.
    """
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
                f"[RefraktLogger] Skipping duplicate metric '{full_metric_name}' \
                    at step {step}"
            )

    return metrics_to_log


def _log_to_tensorboard(
    tb_writer: Any, metrics_to_log: Dict[str, float], step: int, prefix: Optional[str]
) -> None:
    """
    Log metrics to TensorBoard.

    This function writes metrics to TensorBoard using the provided writer.
    It handles prefix application and ensures proper step tracking.

    Args:
        tb_writer: TensorBoard writer instance
        metrics_to_log: Dictionary of metrics to log
        step: Current training step number
        prefix: Optional prefix to apply to metric names
    """
    if tb_writer:
        for k, v in metrics_to_log.items():
            full_k = f"{prefix}/{k}" if prefix else k
            tb_writer.add_scalar(full_k, v, step)


def _log_to_wandb(
    wandb_run: Any, metrics_to_log: Dict[str, float], step: int, prefix: Optional[str]
) -> None:
    """
    Log metrics to WandB.

    This function logs metrics to WandB using the provided run instance.
    It handles prefix application and ensures proper step tracking.

    Args:
        wandb_run: WandB run instance
        metrics_to_log: Dictionary of metrics to log
        step: Current training step number
        prefix: Optional prefix to apply to metric names
    """
    if wandb_run:
        log_data = {
            f"{prefix}/{k}" if prefix else k: v for k, v in metrics_to_log.items()
        }
        wandb_run.log(log_data, step=step)


def _prepare_input_tensor_for_graph(model: nn.Module, input_tensor: Any) -> Any:
    """
    Prepare input tensor for model graph logging.

    This function ensures the input tensor is on the same device as the model
    parameters, which is required for proper graph logging.

    Args:
        model: The model whose parameters determine the target device
        input_tensor: The input tensor to prepare (can be dict or tensor)

    Returns:
        Input tensor moved to the model's device
    """
    device = next(model.parameters()).device
    if isinstance(input_tensor, dict):
        input_tensor = {k: v.to(device) for k, v in input_tensor.items()}
    else:
        input_tensor = input_tensor.to(device)
    return input_tensor


def _should_skip_fusion_block_logging(model: nn.Module) -> bool:
    """
    Check if we should skip logging for FusionBlock models.

    This function determines whether to skip graph logging for FusionBlock
    models, which can have complex structures that don't log well.

    Args:
        model: The model to check for FusionBlock type

    Returns:
        True if logging should be skipped for this model type
    """
    return hasattr(model, "__class__") and "FusionBlock" in model.__class__.__name__


def _log_to_tensorboard_graph(
    tb_writer: Any, model: nn.Module, input_tensor: Any, logger_instance: Any
) -> None:
    """
    Log model graph to TensorBoard.

    This function creates a tracing model and logs the computational graph
    to TensorBoard for visualization. It includes error handling and special
    cases for complex models.

    Args:
        tb_writer: TensorBoard writer instance
        model: The model to log the graph for
        input_tensor: The input tensor for graph tracing
        logger_instance: Logger instance for status messages

    Note:
        This function includes special handling for FusionBlock models which
        may have complex structures that don't log well to TensorBoard.
    """
    if tb_writer:
        try:
            if _should_skip_fusion_block_logging(model):
                logger_instance.info(
                    "Skipping TensorBoard graph logging for FusionBlock \
                        (complex model structure)"
                )
                return

            from .logging_utils import create_tracing_model

            tracing_model = create_tracing_model(model)
            tracing_model.eval()
            tb_writer.add_graph(tracing_model, input_tensor)
            logger_instance.info("Logged model graph to TensorBoard.")
        except Exception as e:
            logger_instance.warning(f"TensorBoard model graph logging failed: {e}")


def _log_to_wandb_watch(wandb_run: Any, model: nn.Module, logger_instance: Any) -> None:
    """
    Set up WandB model watching.

    This function configures WandB to watch the model's gradients and parameters
    for automatic logging during training.

    Args:
        wandb_run: WandB run instance
        model: The model to watch
        logger_instance: Logger instance for status messages

    Note:
        WandB watching provides automatic logging of gradients and parameters
        during training, which can be useful for debugging and analysis.
    """
    if wandb_run:
        try:

            wandb_run.watch(model, log="all", log_freq=100)
            logger_instance.info("WandB is watching model and gradients.")
        except Exception as e:
            logger_instance.error(f"WandB model watching failed: {e}")


def _prepare_images_for_logging(images: Any) -> Any:
    """
    Prepare images for logging by converting to appropriate format.

    This function ensures images are in the correct format for logging
    to TensorBoard or WandB, handling various input types.

    Args:
        images: The images to prepare (can be tensor, numpy array, or list)

    Returns:
        Images converted to the appropriate format for logging
    """
    if isinstance(images, Tensor):
        images = images.detach().cpu().numpy()
    # Convert ndarray to list if needed for Sequence
    if isinstance(images, np.ndarray):
        images_seq = images.tolist()
    else:
        images_seq = images
    return images_seq


def _log_images_to_tensorboard(
    tb_writer: Any, tag: str, images_seq: Any, step: int, dataformats: str
) -> None:
    """
    Log images to TensorBoard.

    Args:
        tb_writer (Any): TensorBoard writer instance.
        tag (str): The tag for the images.
        images_seq (Any): The sequence of images to log.
        step (int): Current training step.
        dataformats (str): The data format for TensorBoard.
    """
    if tb_writer:
        tb_writer.add_images(tag, np.array(images_seq), step, dataformats=dataformats)


def _log_images_to_wandb(
    wandb_run: Any, tag: str, images_seq: Any, step: int, to_wandb_image_func: Any
) -> None:
    """
    Log images to WandB.

    Args:
        wandb_run (Any): WandB run instance.
        tag (str): The tag for the images.
        images_seq (Any): The sequence of images to log.
        step (int): Current training step.
        to_wandb_image_func (Any): Function to convert image to WandB format.
    """
    if wandb_run:
        import wandb

        wandb_run.log(
            {tag: [wandb.Image(to_wandb_image_func(img)) for img in images_seq]},
            step=step,
        )
