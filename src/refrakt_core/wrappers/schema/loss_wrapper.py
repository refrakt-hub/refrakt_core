# refrakt_core/losses/loss_wrapper.py

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
            return LossOutput(total=result)

        args = signature(self.fn).parameters.keys()
        input_dict = {}

        if self.field_map:
            for k, v in self.field_map.items():
                attr = getattr(output, v, None)
                if attr is not None:
                    input_dict[k] = attr
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
            return LossOutput(total=sum(result.values()), components=result)
        elif isinstance(result, tuple):
            total, components = result
            return LossOutput(total=total, components=components)
        else:
            return LossOutput(total=result)
