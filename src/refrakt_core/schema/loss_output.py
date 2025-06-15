# refrakt_core/outputs/loss_output.py

from dataclasses import dataclass, field
from typing import Dict
import torch


@dataclass
class LossOutput:
    total: torch.Tensor
    components: Dict[str, torch.Tensor] = field(default_factory=dict)

    def item(self) -> float:
        """Returns the scalar float value of the total loss."""
        return self.total.item()

    def __repr__(self) -> str:
        keys = ", ".join(self.components.keys())
        return f"LossOutput(total={self.total.item():.4f}, components=[{keys}])"
