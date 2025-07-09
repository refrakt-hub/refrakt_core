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

    def _add_tensor_stats(
        self, summary: Dict[str, float], tensor: Optional[Any], prefix: str
    ) -> None:
        """
        Helper method to add tensor statistics to summary.
        """
        if tensor is not None and isinstance(tensor, torch.Tensor):
            summary[f"{prefix}/mean"] = tensor.mean().item()
            if prefix != "reconstruction":  # Skip std for reconstruction
                summary[f"{prefix}/std"] = tensor.std().item()

    def _add_embeddings_stats(self, summary: Dict[str, float]) -> None:
        """
        Helper method to add embeddings statistics to summary.
        """
        if self.embeddings is not None and isinstance(self.embeddings, torch.Tensor):
            summary["embeddings/norm_mean"] = self.embeddings.norm(dim=1).mean().item()
            summary["embeddings/std"] = self.embeddings.std().item()

    def _add_loss_components(self, summary: Dict[str, float]) -> None:
        """
        Helper method to add loss components to summary.
        """
        for k, v in self.loss_components.items():
            if isinstance(v, torch.Tensor):
                summary[f"loss_component/{k}"] = v.item()

    def _add_extra_components(self, summary: Dict[str, float]) -> None:
        """
        Helper method to add extra components to summary.
        """
        for k, v in self.extra.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                summary[f"extra/{k}"] = v.item()

    def summary(self) -> Dict[str, float]:
        summary: Dict[str, float] = {}

        self._add_tensor_stats(summary, self.logits, "logits")
        self._add_embeddings_stats(summary)
        self._add_tensor_stats(summary, self.reconstruction, "reconstruction")
        self._add_tensor_stats(summary, self.attention_maps, "attention")
        self._add_loss_components(summary)
        self._add_extra_components(summary)

        return summary

    def to(self, device: Any) -> "ModelOutput":
        def move(x: Any) -> Any:
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
