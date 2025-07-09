"""
DINO: Self-Distillation with No Labels

Implements the DINO head and wrapper model using student-teacher architecture with
momentum update and normalized projection heads.

Also contains a wrapper for ResNet backbones to integrate with DINO training.
"""

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from refrakt_core.models.resnet import ResNet18, ResNet50, ResNet101, ResNet152
from refrakt_core.models.templates.base import BaseModel
from refrakt_core.registry.model_registry import register_model


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
        self.student_head: DINOHead = DINOHead(
            in_dim=backbone.feature_dim, out_dim=out_dim
        )
        self.teacher_head: DINOHead = DINOHead(
            in_dim=backbone.feature_dim, out_dim=out_dim
        )
        self.teacher_head.load_state_dict(self.student_head.state_dict())

        for param in self.teacher_head.parameters():
            param.requires_grad = False

    def forward(self, x: Tensor, teacher: bool = False, **kwargs: Any) -> Tensor:
        """
        Forward pass through student or teacher head.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).
            teacher (bool): If True, use teacher head.

        Returns:
            Tensor: Projected feature of shape (B, out_dim)
        """
        features = self.backbone(x, return_features=True)
        return self.teacher_head(features) if teacher else self.student_head(features)  # type: ignore[no-any-return]

    def forward_for_graph(self, x: Tensor) -> Tensor:
        features = self.backbone(x, return_features=True)
        return self.student_head(features)  # type: ignore[no-any-return]

    @torch.no_grad()
    def update_teacher(self, momentum: float = 0.996) -> None:
        """
        Exponential Moving Average (EMA) update of teacher parameters.

        Args:
            momentum (float): Momentum factor for EMA update.
        """
        for student_param, teacher_param in zip(
            self.student_head.parameters(), self.teacher_head.parameters(), strict=False
        ):
            teacher_param.data = (
                momentum * teacher_param.data + (1.0 - momentum) * student_param.data
            )


class DINOBackboneWrapper(nn.Module):
    """
    Wraps a backbone model to expose features for DINO training.
    """

    def __init__(self, backbone: Any) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_dim = backbone.feature_dim

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Forward pass through the wrapped backbone to extract features.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Extracted features.
        """
        kwargs["return_features"] = True
        return self.backbone(x, **kwargs)  # type: ignore[no-any-return]


@register_model("dino")
class DINOModelWrapper(DINOModel):
    """
    DINO model wrapper that supports ResNet backbones.
    Instantiates and integrates with the DINO training setup.
    """

    def __init__(
        self, backbone: Any = "resnet18", out_dim: int = 2048, **kwargs: Any
    ) -> None:
        backbone_map = {
            "resnet18": ResNet18,
            "resnet50": ResNet50,
            "resnet101": ResNet101,
            "resnet152": ResNet152,
        }

        if isinstance(backbone, str):
            if backbone not in backbone_map:
                raise ValueError(f"Unsupported backbone '{backbone}' for DINO.")
            backbone_instance = backbone_map[backbone]()
        elif isinstance(backbone, nn.Module):
            backbone_instance = backbone  # type: ignore
        else:
            raise TypeError(
                f"Expected backbone to be str or nn.Module, got {type(backbone)}"
            )

        wrapped = DINOBackboneWrapper(backbone_instance)
        super().__init__(backbone=wrapped, model_name="dino", out_dim=out_dim)
        self._model_config = kwargs  # Store additional config

    def forward(
        self, x: torch.Tensor, teacher: bool = False, **kwargs: Any
    ) -> torch.Tensor:
        return super().forward(x, teacher=teacher, **kwargs)
