"""
DINO: Self-Distillation with No Labels

Implements the DINO head and wrapper model using student-teacher architecture with
momentum update and normalized projection heads.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from refrakt_core.models.templates.base import BaseModel


class DINOHead(nn.Module):
    """
    Projection head used in DINO. Applies multiple linear layers followed by GELU,
    then a weight-normalized linear layer without bias and L2 normalization.

    Args:
        in_dim (int): Input feature dimension.
        out_dim (int): Output projection dimension (default: 65536).
        hidden_dim (int): Hidden dimension in the MLP (default: 2048).
        bottleneck_dim (int): Final dimension before output layer (default: 256).
        num_layers (int): Number of linear layers in MLP (default: 3).
    """

    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: int = 65536,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            elif i < num_layers - 1:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            if i < num_layers - 1:
                layers.append(nn.GELU())

        self.mlp: nn.Sequential = nn.Sequential(*layers)
        self.last_layer: nn.Module = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1.0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the DINO projection head.

        Args:
            x (Tensor): Input features of shape (B, D)

        Returns:
            Tensor: Normalized projected features of shape (B, out_dim)
        """
        x = self.mlp(x)
        x = F.normalize(self.last_layer(x), dim=-1)
        return x


class DINOModel(BaseModel):
    """
    DINO self-supervised model wrapper that manages student and teacher heads.

    Args:
        backbone (nn.Module): Feature extractor that outputs flat feature vectors.
        model_name (str): Model identifier.
        out_dim (int): Output dimension for the projection head.
    """

    def __init__(
        self,
        backbone: nn.Module,
        model_name: str = "dino",
        out_dim: int = 65536,
    ) -> None:
        super().__init__(model_name=model_name, model_type="contrastive")
        self.backbone: nn.Module = backbone
        self.student_head: DINOHead = DINOHead(in_dim=backbone.feature_dim, out_dim=out_dim)
        self.teacher_head: DINOHead = DINOHead(in_dim=backbone.feature_dim, out_dim=out_dim)
        self.teacher_head.load_state_dict(self.student_head.state_dict())

        for param in self.teacher_head.parameters():
            param.requires_grad = False

    def forward(self, x: Tensor, teacher: bool = False) -> Tensor:
        """
        Forward pass through student or teacher head.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).
            teacher (bool): If True, use teacher head.

        Returns:
            Tensor: Projected feature of shape (B, out_dim)
        """
        features = self.backbone(x)
        return self.teacher_head(features) if teacher else self.student_head(features)

    @torch.no_grad()
    def update_teacher(self, momentum: float = 0.996) -> None:
        """
        Exponential Moving Average (EMA) update of teacher parameters.

        Args:
            momentum (float): Momentum factor for EMA update.
        """
        for student_param, teacher_param in zip(
            self.student_head.parameters(), self.teacher_head.parameters()
        ):
            teacher_param.data = (
                momentum * teacher_param.data + (1.0 - momentum) * student_param.data
            )
