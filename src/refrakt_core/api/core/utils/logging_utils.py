"""
Logging utilities for Refrakt.

This module provides utility functions for complex logging operations,
including extraction of tensors from model outputs, scalar config creation,
and image conversion for logging with TensorBoard and WandB.
These utilities are used throughout the logging and visualization pipeline
to ensure consistent and robust logging of metrics, images, and model graphs.

The module handles:
- Tensor extraction from various model output formats
- Configuration flattening and scalar value handling
- Image format conversion for different logging backends
- Model tracing for graph visualization
- Summary object handling for complex configurations

These utilities ensure that logging operations work consistently across different
model architectures and output formats, providing robust visualization capabilities.

Typical usage involves extracting tensors from model outputs and converting
configurations to scalar formats suitable for logging systems.
"""

from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
import torch
from torch import Tensor, nn

from refrakt_core.api.core.extras import flatten_and_filter_config


def extract_tensor_from_model_output(output: Any) -> Optional[Tensor]:
    """
    Extract a tensor from a model output for logging purposes.

    This function handles various model output formats and attempts to extract
    a suitable tensor for logging. It checks for common patterns like logits
    or reconstruction attributes, and falls back to searching for any tensor.

    Args:
        output: The model output, which may be a Tensor, an object with a 'logits'
                or 'reconstruction' attribute, or another type

    Returns:
        The extracted tensor, or None if no suitable tensor is found

    Note:
        This function is designed to handle various model output formats
        including wrapped outputs, named tuples, and custom objects.
    """
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "logits") and isinstance(output.logits, torch.Tensor):
        return output.logits
    if hasattr(output, "reconstruction") and isinstance(
        output.reconstruction, torch.Tensor
    ):
        return output.reconstruction
    # Try to find any tensor in output
    for attr in dir(output):
        if not attr.startswith("_"):
            value = getattr(output, attr)
            if isinstance(value, torch.Tensor):
                return value
    return None


def _handle_scalar_value(k: str, v: Any) -> Dict[str, Any]:
    """
    Handle scalar value types for config flattening.

    This function processes individual configuration values to extract scalar
    values suitable for logging. It handles various data types including
    tensors with single elements and single-item sequences.

    Args:
        k: Configuration key name
        v: Value to check and potentially convert to scalar

    Returns:
        Dictionary with scalar value if applicable, else empty dict
    """
    if isinstance(v, (int, float, str, bool)):
        return {k: v}
    elif torch.is_tensor(v) and v.numel() == 1:
        return {k: v.item()}
    elif isinstance(v, (list, tuple)) and len(v) == 1:
        return {k: v[0]}
    return {}


def _handle_summary_object(k: str, v: Any) -> Dict[str, Any]:
    """
    Handle objects with a summary method for config flattening.

    This function processes objects that have a summary method, extracting
    their summary values for logging. It's useful for complex objects that
    can provide meaningful scalar representations.

    Args:
        k: Configuration key name
        v: Value to check for summary method

    Returns:
        Dictionary with summary values if applicable, else empty dict
    """
    if (
        not isinstance(v, (torch.Tensor, list, tuple))
        and hasattr(v, "summary")
        and callable(getattr(v, "summary", None))
    ):
        summary = v.summary()
        if isinstance(summary, dict):
            return {f"{k}/{sk}": sv for sk, sv in summary.items()}
    return {}


def create_scalar_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a clean scalar-only config for logging.

    This function processes a configuration dictionary to extract only scalar
    values suitable for logging. It flattens nested structures and handles
    various data types to create a clean logging configuration.

    Args:
        config: Configuration dictionary to process

    Returns:
        Scalar configuration dictionary suitable for logging

    Note:
        This function is useful for logging configuration parameters that
        need to be displayed as scalar values in logging interfaces.
    """
    scalar_config = {}
    for k, v in flatten_and_filter_config(config).items():
        scalar_config.update(_handle_scalar_value(k, v))
        scalar_config.update(_handle_summary_object(k, v))
    return scalar_config


def create_tracing_model(model: nn.Module) -> nn.Module:
    """
    Create a tracing model for TensorBoard graph logging.

    This function creates a wrapper around the original model that ensures
    the output is suitable for TensorBoard graph visualization. It handles
    models with custom forward methods and extracts tensors appropriately.

    Args:
        model: Original model to wrap for tracing

    Returns:
        Tracing model wrapper suitable for TensorBoard graph logging

    Note:
        The tracing model ensures that the output is always a tensor,
        which is required for proper TensorBoard graph visualization.
    """

    class TracingModel(nn.Module):
        """
        Wrapper for a model to enable tracing for TensorBoard graph logging.
        """

        def __init__(self, model: nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(self, x: Any) -> torch.Tensor:
            # Use forward_for_graph if available
            if hasattr(self.model, "forward_for_graph"):
                return cast(torch.Tensor, self.model.forward_for_graph(x))
            # Otherwise extract tensor from regular output
            output = self.model(x)
            tensor = extract_tensor_from_model_output(output)
            if tensor is None:
                raise ValueError("No tensor found in model output for tracing")
            return tensor

    return TracingModel(model)


def convert_to_wandb_image(
    img: Union[torch.Tensor, np.ndarray[Any, Any], List[Any]],
) -> np.ndarray[Any, Any]:
    """
    Convert an image to a WandB-compatible numpy array format.

    This function handles various image input formats and converts them to
    the format expected by WandB for proper visualization.

    Args:
        img: Input image in various formats (tensor, numpy array, or list)

    Returns:
        WandB compatible image as numpy array

    Note:
        The function handles tensor-to-numpy conversion and channel ordering
        adjustments to ensure compatibility with WandB's image logging.
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if isinstance(img, list):
        img = np.array(img)
    if isinstance(img, np.ndarray):
        # If shape is (C, H, W), convert to (H, W, C)
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))
    return img
