"""
Utility functions for default loss wrappers.
"""

from typing import Any, Callable, Dict, Optional, Union

import torch

from refrakt_core.schema.loss_output import LossOutput
from refrakt_core.schema.model_output import ModelOutput


def handle_mae_loss(
    loss_fn: Callable[[Union[torch.Tensor, ModelOutput, Dict[str, Any]]], LossOutput],
    output: Union[torch.Tensor, ModelOutput, Dict[str, Any]],
) -> LossOutput:
    """Handle MAE loss computation."""
    return loss_fn(output)


def handle_vae_loss(
    loss_fn: Callable[
        [Union[torch.Tensor, ModelOutput, Dict[str, Any]], Optional[torch.Tensor]],
        LossOutput,
    ],
    output: Union[torch.Tensor, ModelOutput, Dict[str, Any]],
    target: Optional[torch.Tensor],
) -> LossOutput:
    """Handle VAE loss computation."""
    if isinstance(output, ModelOutput) and hasattr(output, "reconstruction"):
        effective_target = target if target is not None else output.reconstruction
    else:
        effective_target = target
    return loss_fn(output, effective_target)


def extract_tensor_from_model_output(output: ModelOutput) -> torch.Tensor:
    """Extract tensor from ModelOutput based on available attributes."""
    if hasattr(output, "logits") and output.logits is not None:
        return output.logits  # type: ignore[no-any-return]
    elif hasattr(output, "reconstruction") and output.reconstruction is not None:
        return output.reconstruction  # type: ignore[no-any-return]
    elif hasattr(output, "embeddings") and output.embeddings is not None:
        return output.embeddings  # type: ignore[no-any-return]
    else:
        raise ValueError("Cannot extract tensor from ModelOutput")


def create_loss_output(result: Union[torch.Tensor, LossOutput]) -> LossOutput:
    """Create LossOutput from result."""
    if isinstance(result, LossOutput):
        return result
    elif isinstance(result, torch.Tensor):
        return LossOutput(total=result, components={"loss": result})
    else:
        raise TypeError(
            f"[DefaultLossWrapper] Unexpected loss_fn return type: {type(result)}"
        )
