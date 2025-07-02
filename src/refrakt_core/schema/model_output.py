from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class ModelOutput:
    embeddings: Optional[Any] = None  # contrastive / latent features
    logits: Optional[Any] = None  # supervised output
    image: Optional[Any] = None  # GAN or output image
    reconstruction: Optional[Any] = None  # AE / VAE
    targets: Optional[Any] = None  # target values/labels
    attention_maps: Optional[Any] = None  # ViT, DINO
    loss_components: Dict[str, Any] = field(
        default_factory=dict
    )  # for contrastive/self-sup
    extra: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, float]:
        summary = {}

        if self.logits is not None and isinstance(self.logits, torch.Tensor):
            summary["logits/mean"] = self.logits.mean().item()
            summary["logits/std"] = self.logits.std().item()

        if self.embeddings is not None and isinstance(self.embeddings, torch.Tensor):
            summary["embeddings/norm_mean"] = self.embeddings.norm(dim=1).mean().item()
            summary["embeddings/std"] = self.embeddings.std().item()

        if self.reconstruction is not None and isinstance(
            self.reconstruction, torch.Tensor
        ):
            summary["reconstruction/mean"] = self.reconstruction.mean().item()

        if self.attention_maps is not None and isinstance(
            self.attention_maps, torch.Tensor
        ):
            summary["attention/mean"] = self.attention_maps.mean().item()
            summary["attention/std"] = self.attention_maps.std().item()

        # Loss components (e.g., contrastive / custom)
        for k, v in self.loss_components.items():
            if isinstance(v, torch.Tensor):
                summary[f"loss_component/{k}"] = v.item()

        # Extras (if scalar or single-tensor-like)
        for k, v in self.extra.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                summary[f"extra/{k}"] = v.item()

        return summary

    def to(self, device: str) -> "ModelOutput":
        def move(x):
            if isinstance(x, torch.Tensor):
                return x.to(device)
            elif isinstance(x, dict):
                return {k: move(v) for k, v in x.items()}
            elif isinstance(x, list):
                return [move(v) for v in x]
            elif isinstance(x, tuple):
                return tuple(move(v) for v in x)
            else:
                return x

        return ModelOutput(
            embeddings=move(self.embeddings),
            logits=move(self.logits),
            image=move(self.image),
            reconstruction=move(self.reconstruction),
            targets=move(self.targets),
            attention_maps=move(self.attention_maps),
            loss_components=move(self.loss_components),
            extra=move(self.extra),
        )
