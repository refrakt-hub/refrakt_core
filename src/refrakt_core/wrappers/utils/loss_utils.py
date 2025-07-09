"""
Utility functions for loss wrappers.
"""

import torch

from refrakt_core.schema.loss_output import LossOutput


def _convert_dict_to_loss_output(result: dict[str, torch.Tensor | float]) -> LossOutput:
    """Convert dictionary result to LossOutput."""
    if result:
        total = sum(result.values())
        if not isinstance(total, torch.Tensor):
            total = torch.tensor(total)
    else:
        total = torch.tensor(0.0)
    return LossOutput(total=total, components=result)


def _convert_tuple_to_loss_output(
    result: tuple[float | torch.Tensor, dict[str, torch.Tensor | float]],
) -> LossOutput:
    """Convert tuple result to LossOutput."""
    total, components = result
    if not isinstance(total, torch.Tensor):
        total = torch.tensor(total)
    return LossOutput(total=total, components=components)


def _convert_scalar_to_loss_output(result: float | torch.Tensor | None) -> LossOutput:
    """Convert scalar result to LossOutput."""
    if result is None:
        return LossOutput(total=torch.tensor(float("nan")))
    if not isinstance(result, torch.Tensor):
        result = torch.tensor(result)
    return LossOutput(total=result)


def convert_result_to_loss_output(result: object) -> LossOutput:
    """Convert various result types to LossOutput."""
    if isinstance(result, LossOutput):
        return result
    elif isinstance(result, dict):
        return _convert_dict_to_loss_output(result)
    elif isinstance(result, tuple) and len(result) == 2:
        return _convert_tuple_to_loss_output(result)
    else:
        # Use a type cast to satisfy mypy
        from typing import cast

        return _convert_scalar_to_loss_output(cast(float | torch.Tensor | None, result))
