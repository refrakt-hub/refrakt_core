"""
Logger utilities.

This module contains utility functions for logger implementations.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, Optional, Union

from refrakt_core.api.core.extras import flatten_and_filter_config


def extract_tensor_from_model_output(output: Any) -> Optional[torch.Tensor]:
    """Extract tensor from model output."""
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
    return None


def _handle_scalar_value(k: str, v: Any) -> Dict[str, Any]:
    """Handle scalar value types for config flattening."""
    if isinstance(v, (int, float, str, bool)):
        return {k: v}
    elif torch.is_tensor(v) and v.numel() == 1:
        return {k: v.item()}
    elif isinstance(v, (list, tuple)) and len(v) == 1:
        return {k: v[0]}
    return {}


def _handle_summary_object(k: str, v: Any) -> Dict[str, Any]:
    """Handle objects with summary method for config flattening."""
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
    """Create a clean scalar-only config by flattening and filtering."""
    scalar_config = {}
    for k, v in flatten_and_filter_config(config).items():
        # Handle different value types
        scalar_config.update(_handle_scalar_value(k, v))
        scalar_config.update(_handle_summary_object(k, v))
    return scalar_config


def create_tracing_model(model: nn.Module) -> nn.Module:
    """Create a tracing model that wraps the original model."""
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
            extracted = extract_tensor_from_model_output(output)
            if extracted is None:
                raise ValueError("No tensor found in model output for tracing")
            return extracted

    return TracingModel(model)


def convert_to_wandb_image(img: Union[torch.Tensor, np.ndarray, list]) -> np.ndarray:
    """Convert image to WandB format."""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if isinstance(img, list):
        img = np.array(img)
    if isinstance(img, np.ndarray):
        # If shape is (C, H, W), convert to (H, W, C)
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))
    return img 