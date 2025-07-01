import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Union, Iterator
from refrakt_core.integrations.fusion.builder import build_fusion_head
from refrakt_core.integrations.fusion.protocols import FusionHead
from refrakt_core.schema.model_output import ModelOutput


class FusionBlock(nn.Module):
    """
    Fusion block that wraps a DL backbone and an optional ML fusion head (e.g., sklearn).
    During training (or if not fitted), it returns embeddings for supervised training.
    During eval/inference, it uses the fusion head to predict from embeddings.
    """

    def __init__(self, backbone: nn.Module, fusion_cfg: Dict):
        super().__init__()
        self.backbone = backbone
        self.fusion_head: FusionHead = build_fusion_head(fusion_cfg)
        self._trained = False
        self.wrapper_config = {"wrapper_type": "fusion"}
        
        # Register backbone as a submodule to ensure its parameters are tracked
        self.add_module("backbone", backbone)

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """
        Return an iterator over module parameters.
        This ensures the backbone's parameters are included in optimization.
        """
        return self.backbone.parameters(recurse=recurse)

    def _extract_features(self, x: torch.Tensor) -> np.ndarray:
        output: ModelOutput = self.backbone(x)
        feats = output.embeddings
        if feats is None:
            raise ValueError("Backbone did not return embeddings in ModelOutput.")
        return feats.detach().cpu().numpy(), output

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        feats, _ = self._extract_features(x)
        labels = y.detach().cpu().numpy()
        self.fusion_head.fit(feats, labels)
        self._trained = True

    def forward(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]], teacher: bool = False, **kwargs) -> ModelOutput:
        # Handle dict input (for MSN)
        if isinstance(x, dict):
            base_output = self.backbone(x)
            feats = base_output.embeddings if isinstance(base_output, ModelOutput) else base_output
            # Only convert to numpy if feats is a tensor and not None
            if feats is not None and isinstance(feats, torch.Tensor):
                feats_np = feats.detach().cpu().numpy()
            else:
                feats_np = None
        else:
            # Standard tensor input
            if hasattr(self.backbone, 'forward'):
                import inspect
                sig = inspect.signature(self.backbone.forward)
                if 'teacher' in sig.parameters:
                    base_output = self.backbone(x, teacher=teacher, **kwargs)
                else:
                    base_output = self.backbone(x)
            else:
                base_output = self.backbone(x)
            feats = base_output.embeddings if isinstance(base_output, ModelOutput) else base_output
            if feats is not None and isinstance(feats, torch.Tensor):
                feats_np = feats.detach().cpu().numpy()
            else:
                feats_np = None

        if not self.training and self._trained and feats_np is not None:
            preds = self.fusion_head.predict(feats_np)
            proba = None
            try:
                proba = self.fusion_head.predict_proba(feats_np)
            except AttributeError:
                pass

            return ModelOutput(
                embeddings=feats,
                logits=torch.tensor(preds, device=x["anchor"].device if isinstance(x, dict) else x.device),
                extra={"fusion_preds": preds, "fusion_proba": proba}
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
                extra=base_output.extra
            )
        else:
            return ModelOutput(
                embeddings=feats,
                logits=getattr(base_output, 'logits', None)
            )

    def forward_for_graph(self, x: torch.Tensor) -> torch.Tensor:
        """
        Traceable forward method for TensorBoard graph visualization.
        Returns the logits tensor directly without numpy conversions.
        """

        output: ModelOutput = self.backbone(x)
        
        if output.logits is not None:
            return output.logits
        elif output.embeddings is not None:
            return output.embeddings
        else:
            return torch.zeros(x.shape[0], 10, device=x.device)  # Assuming 10 classes

    def predict_proba(self, x: torch.Tensor) -> Optional[np.ndarray]:
        feats, _ = self._extract_features(x)
        return self.fusion_head.predict_proba(feats) if self._trained else None

    def update_teacher(self, *args, **kwargs):
        """
        Delegate teacher update to the backbone if available.
        """
        if hasattr(self.backbone, "update_teacher"):
            return self.backbone.update_teacher(*args, **kwargs)
        raise AttributeError("Backbone does not support update_teacher()")
