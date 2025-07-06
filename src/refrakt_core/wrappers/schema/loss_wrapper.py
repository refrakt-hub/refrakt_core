# refrakt_core/losses/loss_wrapper.py

import torch
from inspect import signature
from typing import Optional

from refrakt_core.schema.loss_output import LossOutput
from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.wrappers.utils.loss_utils import convert_result_to_loss_output


def _handle_non_model_output(fn, output, target) -> LossOutput:
    """Handle case when output is not a ModelOutput."""
    result = fn(output, target)
    return convert_result_to_loss_output(result)


def _build_input_dict_with_field_map(fn, output: ModelOutput, target, field_map: dict) -> dict:
    """Build input dictionary using field mapping."""
    args = signature(fn).parameters.keys()
    input_dict = {}

    for k, v in field_map.items():
        if v is None:
            # Special case: use the target parameter
            input_dict[k] = target
        else:
            attr = getattr(output, v, None)
            if attr is not None:
                input_dict[k] = attr
    
    # Also add target if it's expected but not in field_map
    if "target" in args and "target" not in field_map:
        input_dict["target"] = target
    
    return input_dict


def _build_input_dict_auto(fn, output: ModelOutput, target) -> dict:
    """Build input dictionary automatically from ModelOutput attributes."""
    args = signature(fn).parameters.keys()
    input_dict = {}

    for arg in args:
        if arg == "target":
            input_dict["target"] = target
        elif hasattr(output, arg):
            val = getattr(output, arg)
            if val is not None:
                input_dict[arg] = val
        elif arg == "output":
            input_dict["output"] = output

    return input_dict


class LossWrapper:
    """
    Wraps any loss function or class and dispatches fields from ModelOutput accordingly.
    Returns a LossOutput with total + breakdowns.
    """

    def __init__(self, fn, field_map: Optional[dict] = None):
        """
        Args:
            fn: The actual loss function or class instance.
            field_map: Optional dict to map expected kwargs (like `student`, `reconstruction`)
                       to attributes from ModelOutput. Used when loss expects nonstandard names.
        """
        self.fn = fn
        self.field_map = field_map or {}

    def __call__(self, output: ModelOutput, target=None) -> LossOutput:
        if not isinstance(output, ModelOutput):
            return _handle_non_model_output(self.fn, output, target)

        # Build input dictionary based on field mapping
        if self.field_map:
            input_dict = _build_input_dict_with_field_map(self.fn, output, target, self.field_map)
        else:
            input_dict = _build_input_dict_auto(self.fn, output, target)

        result = self.fn(**input_dict)
        return convert_result_to_loss_output(result)
