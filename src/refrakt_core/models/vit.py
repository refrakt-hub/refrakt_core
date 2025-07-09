"""
Vision Transformer (ViT) model implementation for image classification.

This module defines a ViT-based classifier registered as "vit" in the model registry.
"""

import torch
from torch import Tensor, nn

from refrakt_core.models.templates.models import BaseClassifier
from refrakt_core.registry.model_registry import register_model
from refrakt_core.utils.classes.resnet import ViTResidual
from refrakt_core.utils.methods import patchify, positional_embeddings


@register_model("vit")
class VisionTransformer(BaseClassifier):
    """
    Vision Transformer classifier for image classification.

    Args:
        image_size (int): Size of input images (assumes square images).
        patch_size (int): Size of each patch.
        num_classes (int): Number of output classes.
        dim (int): Embedding dimension.
        depth (int): Number of transformer blocks.
        heads (int): Number of attention heads.
        in_channels (int): Number of input image channels.
        model_name (str): Internal model name for logging or saving.
    """

    n_patches: int
    patch_size: int
    hidden_d: int
    input_d: int
    linear_mapper: nn.Linear
    v_class: nn.Parameter
    positional_embeddings: Tensor
    blocks: nn.ModuleList
    mlp_head: nn.Sequential

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_classes: int = 1000,
        dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        in_channels: int = 3,
        model_name: str = "vit_classifier",
    ) -> None:
        super().__init__(num_classes=num_classes, model_name=model_name)

        assert (
            image_size % patch_size == 0
        ), "Image size must be divisible by patch size"

        self.n_patches = image_size // patch_size
        self.patch_size = patch_size
        self.hidden_d = dim
        self.input_d = in_channels * patch_size * patch_size

        self.linear_mapper = nn.Linear(self.input_d, dim)
        self.v_class = nn.Parameter(torch.rand(1, dim))
        self.register_buffer(
            "positional_embeddings",
            positional_embeddings(self.n_patches**2 + 1, dim),
            persistent=False,
        )

        self.blocks = nn.ModuleList([ViTResidual(dim, heads) for _ in range(depth)])
        self.mlp_head = nn.Sequential(nn.Linear(dim, num_classes))

    def forward_features(self, images: Tensor) -> Tensor:
        """
        Extract features from images using ViT blocks.

        Args:
            images (Tensor): Input image batch of shape (B, C, H, W).

        Returns:
            Tensor: CLS token representation of shape (B, D).
        """
        n = images.shape[0]
        patches = patchify(images, self.n_patches).to(self.positional_embeddings.device)
        tokens = self.linear_mapper(patches)

        tokens = torch.cat([self.v_class.expand(n, 1, -1), tokens], dim=1)
        x = tokens + self.positional_embeddings.repeat(n, 1, 1)

        for block in self.blocks:
            x = block(x)

        return x[:, 0]  # CLS token

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the full ViT classifier.

        Args:
            x (Tensor): Input images.

        Returns:
            Tensor: Output logits for classification.
        """
        cls_token = self.forward_features(x)
        return self.mlp_head(cls_token)  # type: ignore[no-any-return]

    def features(self, x: Tensor) -> Tensor:
        """
        Return features extracted by ViT (before classification head).

        Args:
            x (Tensor): Input images.

        Returns:
            Tensor: CLS token embeddings.
        """
        return self.forward_features(x)
