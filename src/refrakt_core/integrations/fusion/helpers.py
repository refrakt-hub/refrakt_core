"""
Helper functions for fusion trainer and block decomposition.
"""

from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch

from refrakt_core.schema.model_output import ModelOutput


def unpack_batch(
    batch: Union[Tuple[Any, ...], List[Any], Dict[str, torch.Tensor]],
) -> Tuple[Any, Any]:
    """
    Unpack batch data in various formats.

    Args:
        batch: Batch data in tuple, list, or dict format

    Returns:
        Tuple of (input_data, target_data)

    Raises:
        TypeError: If batch format is unsupported
    """
    # Handle SimCLR-style batches: (img1, img2, label)
    if isinstance(batch, (tuple, list)):
        if len(batch) == 3:
            return batch[0], batch[2]  # img1, label
        return batch[0], batch[1]
    if isinstance(batch, dict):
        return batch["input"], batch["target"]
    raise TypeError("Unsupported batch format")


def extract_features_from_model(
    model: Any, x: Any, device: Any, teacher: bool = False, **kwargs: Any
) -> Tuple[Any, Any]:
    """
    Extract features from model with proper handling of different input types.

    Args:
        model: The model to extract features from
        x: Input tensor or dict
        device: Device to run on
        teacher: Whether to use teacher mode
        **kwargs: Additional arguments

    Returns:
        Tuple of (features, output)
    """
    # Handle dict input (for MSN)
    if isinstance(x, dict):
        base_output = model(x)
        feats = (
            base_output.embeddings
            if isinstance(base_output, ModelOutput)
            else base_output
        )
        # Only convert to numpy if feats is a tensor and not None
        if feats is not None and isinstance(feats, torch.Tensor):
            feats_np = feats.detach().cpu().numpy()
        else:
            feats_np = None
    else:
        # Standard tensor input
        if hasattr(model, "forward"):
            import inspect

            sig = inspect.signature(model.forward)
            if "teacher" in sig.parameters:
                base_output = model(x, teacher=teacher, **kwargs)
            else:
                base_output = model(x)
        else:
            base_output = model(x)
        feats = (
            base_output.embeddings
            if isinstance(base_output, ModelOutput)
            else base_output
        )
        if feats is not None and isinstance(feats, torch.Tensor):
            feats_np = feats.detach().cpu().numpy()
        else:
            feats_np = None

    return feats_np, base_output


def validate_model_output(output: Any, require_embeddings: bool = True) -> None:
    """
    Validate that model output has required components.

    Args:
        output: Model output to validate
        require_embeddings: Whether embeddings are required

    Raises:
        ValueError: If output doesn't meet requirements
    """
    if require_embeddings:
        if not isinstance(output, ModelOutput) or output.embeddings is None:
            raise ValueError(
                "Backbone must return `ModelOutput` with `embeddings` for fusion mode."
            )


def process_labels(y: torch.Tensor) -> np.ndarray[Any, np.dtype[Any]]:
    """
    Process labels to ensure correct format.

    Args:
        y: Input labels

    Returns:
        Processed labels as numpy array
    """
    y_cpu = y.detach().cpu()
    if y_cpu.ndim > 1:
        y_cpu = y_cpu.view(-1)
    return y_cpu.numpy()
