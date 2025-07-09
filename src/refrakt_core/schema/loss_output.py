from dataclasses import dataclass, field
from typing import Dict, Union

import torch


@dataclass
class LossOutput:
    total: torch.Tensor
    components: Dict[str, Union[torch.Tensor, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Ensure all components are torch.Tensor
        self.components = {
            k: (v if isinstance(v, torch.Tensor) else torch.tensor(v))
            for k, v in self.components.items()
        }

    def item(self) -> float:
        return self.total.item()

    def summary(self) -> Dict[str, float]:
        summary = {"loss/total": self.total.item()}
        for name, val in self.components.items():
            if isinstance(val, torch.Tensor):
                summary[f"loss/{name}"] = val.item()
            else:
                summary[f"loss/{name}"] = float(val)
        return summary

    def __repr__(self) -> str:
        keys = ", ".join(
            f"{k}={v.item():.4f}" if isinstance(v, torch.Tensor) else f"{k}={v:.4f}"
            for k, v in self.components.items()
        )
        return f"LossOutput(total={self.total.item():.4f}, components=[{keys}])"
