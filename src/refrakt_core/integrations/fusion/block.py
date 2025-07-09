from typing import Any, Dict, Iterator, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

from refrakt_core.integrations.fusion.builder import build_fusion_head
from refrakt_core.integrations.fusion.helpers import (
    extract_features_from_model,
    process_labels,
    validate_model_output,
)
from refrakt_core.integrations.fusion.protocols import FusionHead
from refrakt_core.schema.model_output import ModelOutput


class FusionBlock(nn.Module):
    """
    Fusion block that wraps a DL backbone and an optional ML fusion head (e.g., sklearn).
    During training (or if not fitted), it returns embeddings for supervised training.
    During eval/inference, it uses the fusion head to predict from embeddings.
    """

    def __init__(self, backbone: nn.Module, fusion_cfg: Dict[str, Any]):
        super().__init__()
        # Check for generative models
        generative_types = ("mae", "autoencoder", "vae", "srgan")
        backbone_type = type(backbone).__name__.lower()
        if any(gen_type in backbone_type for gen_type in generative_types):
            raise NotImplementedError(
                "Fusion is not yet supported for generative models (MAE, AE, VAE, SRGAN)"
            )

        self.backbone = backbone
        self.fusion_head: FusionHead = build_fusion_head(fusion_cfg)
        self._trained = False
        self.wrapper_config = {"wrapper_type": "fusion"}

        # Register backbone as a submodule to ensure its parameters are tracked
        self.add_module("backbone", backbone)

    @property
    def device(self) -> torch.device:
        # Delegate to backbone if possible, else default to cpu
        if hasattr(self.backbone, "device"):
            dev = self.backbone.device
            if isinstance(dev, torch.device):
                return dev
            try:
                return torch.device(dev)
            except Exception:
                pass
        return torch.device("cpu")

    @device.setter
    def device(self, value: torch.device) -> None:
        # Set device on backbone if possible
        if hasattr(self.backbone, "to_device"):
            self.backbone.to_device(value)
        elif hasattr(self.backbone, "to"):
            self.backbone.to(value)
        # else: do nothing

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """
        Return an iterator over module parameters.
        This ensures the backbone's parameters are included in optimization.
        """
        return self.backbone.parameters(recurse=recurse)

    def _extract_features(
        self, x: torch.Tensor
    ) -> tuple[NDArray[np.float64], ModelOutput]:
        """
        Extract features from the backbone and return as numpy array and ModelOutput.
        """
        output: ModelOutput = self.backbone(x)
        validate_model_output(output)
        feats = output.embeddings
        if feats is not None:
            return feats.detach().cpu().numpy(), output
        raise ValueError("Backbone did not return embeddings in ModelOutput.")

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        feats, _ = self._extract_features(x)
        labels = process_labels(y)
        self.fusion_head.fit(feats, labels)
        self._trained = True

    def forward(
        self,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
        teacher: bool = False,
        **kwargs: Any,
    ) -> ModelOutput:
        feats_np, base_output = extract_features_from_model(
            self.backbone, x, self.device, teacher, **kwargs
        )

        if not self.training and self._trained and feats_np is not None:
            preds = self.fusion_head.predict(feats_np)
            proba = None
            try:
                proba = self.fusion_head.predict_proba(feats_np)
            except AttributeError:
                pass

            return ModelOutput(
                embeddings=(
                    base_output.embeddings
                    if isinstance(base_output, ModelOutput)
                    else None
                ),
                logits=torch.tensor(
                    preds,
                    device=x["anchor"].device if isinstance(x, dict) else x.device,
                ),
                extra={"fusion_preds": preds, "fusion_proba": proba},
            )

        # Propagate all ModelOutput fields if base_output is a ModelOutput
        if isinstance(base_output, ModelOutput):
            return ModelOutput(
                embeddings=base_output.embeddings,
                logits=base_output.logits,
                image=base_output.image,
                reconstruction=base_output.reconstruction,
                targets=base_output.targets,
                attention_maps=base_output.attention_maps,
                loss_components=base_output.loss_components,
                extra=base_output.extra,
            )
        else:
            return ModelOutput(
                embeddings=(
                    base_output.embeddings
                    if hasattr(base_output, "embeddings")
                    else None
                ),
                logits=getattr(base_output, "logits", None),
            )

    def forward_for_graph(self, x: torch.Tensor) -> torch.Tensor:
        """
        Traceable forward method for TensorBoard graph visualization.
        Returns the logits tensor directly without numpy conversions.
        """
        output: ModelOutput = self.backbone(x)
        if output.logits is not None:
            if isinstance(output.logits, torch.Tensor):
                return output.logits
            else:
                raise TypeError("output.logits must be a torch.Tensor")
        elif output.embeddings is not None:
            if isinstance(output.embeddings, torch.Tensor):
                return output.embeddings
            else:
                raise TypeError("output.embeddings must be a torch.Tensor")
        else:
            return torch.zeros(x.shape[0], 10, device=x.device)  # Assuming 10 classes

    def predict_proba(self, x: torch.Tensor) -> Optional[NDArray[np.float64]]:
        """
        Predict class probabilities using the fusion head if trained.
        """
        feats, _ = self._extract_features(x)
        return self.fusion_head.predict_proba(feats) if self._trained else None

    def update_teacher(self, *args: Any, **kwargs: Any) -> Any:
        """
        Delegate teacher update to the backbone if available.
        """
        if hasattr(self.backbone, "update_teacher"):
            return self.backbone.update_teacher(*args, **kwargs)
        raise AttributeError("Backbone does not support update_teacher()")

    def get_logits(self, output: Any) -> torch.Tensor:
        logits = getattr(output, "logits", None)
        if not isinstance(logits, torch.Tensor):
            raise TypeError("logits must be a torch.Tensor")
        return logits

    def get_embeddings(self, output: Any) -> torch.Tensor:
        embeddings = getattr(output, "embeddings", None)
        if not isinstance(embeddings, torch.Tensor):
            raise TypeError("embeddings must be a torch.Tensor")
        return embeddings
