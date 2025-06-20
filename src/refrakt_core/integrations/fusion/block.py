import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
from refrakt_core.integrations.fusion.builder import build_fusion_head
from refrakt_core.integrations.fusion.protocols import FusionHead
from refrakt_core.schema.model_output import ModelOutput


class FusionBlock(nn.Module):
    """
    Fusion block that combines a neural backbone with an ML head (sklearn/cuml).

    The backbone returns a ModelOutput with embeddings, which the ML head is trained on.
    """

    def __init__(self, backbone: nn.Module, fusion_cfg: Dict):
        super().__init__()
        self.backbone = backbone
        self.fusion_head: FusionHead = build_fusion_head(fusion_cfg)
        self._trained = False

    def _extract_features(self, x: torch.Tensor) -> np.ndarray:
        output: ModelOutput = self.backbone(x)
        feats = output.embeddings
        if feats is None:
            raise ValueError("Backbone did not return embeddings in ModelOutput.")
        return feats.detach().cpu().numpy()

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        feats = self._extract_features(x)
        labels = y.detach().cpu().numpy()
        self.fusion_head.fit(feats, labels)
        self._trained = True

    def forward(self, x: torch.Tensor) -> np.ndarray:
        feats = self._extract_features(x)
        return self.fusion_head.predict(feats)

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        feats = self._extract_features(x)
        return self.fusion_head.predict_proba(feats)
