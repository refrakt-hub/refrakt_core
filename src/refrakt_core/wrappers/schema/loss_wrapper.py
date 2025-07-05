# refrakt_core/losses/loss_wrapper.py

import torch
from inspect import signature
from typing import Optional

from refrakt_core.schema.loss_output import LossOutput
from refrakt_core.schema.model_output import ModelOutput


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
            result = self.fn(output, target)
            if isinstance(result, LossOutput):
                return result
            elif isinstance(result, dict):
                if result:
                    total = sum(result.values())
                    if not isinstance(total, torch.Tensor):
                        total = torch.tensor(total)
                else:
                    total = torch.tensor(0.0)
                return LossOutput(total=total, components=result)
            elif isinstance(result, tuple) and len(result) == 2:
                total, components = result
                if not isinstance(total, torch.Tensor):
                    total = torch.tensor(total)
                if not isinstance(components, dict):
                    components = {}
                return LossOutput(total=total, components=components)
            else:
                if result is None:
                    return LossOutput(total=torch.tensor(float('nan')))
                if not isinstance(result, torch.Tensor):
                    result = torch.tensor(result)
                return LossOutput(total=result)

        args = signature(self.fn).parameters.keys()
        input_dict = {}

        if self.field_map:
            for k, v in self.field_map.items():
                if v is None:
                    # Special case: use the target parameter
                    input_dict[k] = target
                else:
                    attr = getattr(output, v, None)
                    if attr is not None:
                        input_dict[k] = attr
            # Also add target if it's expected but not in field_map
            if "target" in args and "target" not in self.field_map:
                input_dict["target"] = target
        else:
            for arg in args:
                if arg == "target":
                    input_dict["target"] = target
                elif hasattr(output, arg):
                    val = getattr(output, arg)
                    if val is not None:
                        input_dict[arg] = val
                elif arg == "output":
                    input_dict["output"] = output

        result = self.fn(**input_dict)

        if isinstance(result, LossOutput):
            return result
        elif isinstance(result, dict):
            if result:
                total = sum(result.values())
                if not isinstance(total, torch.Tensor):
                    total = torch.tensor(total)
            else:
                total = torch.tensor(0.0)
            return LossOutput(total=total, components=result)
        elif isinstance(result, tuple) and len(result) == 2:
            total, components = result
            if not isinstance(total, torch.Tensor):
                total = torch.tensor(total)
            if not isinstance(components, dict):
                components = {}
            return LossOutput(total=total, components=components)
        else:
            if result is None:
                return LossOutput(total=torch.tensor(float('nan')))
            if not isinstance(result, torch.Tensor):
                result = torch.tensor(result)
            return LossOutput(total=result)
