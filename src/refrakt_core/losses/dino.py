"""
The DINO Loss implementation, which is used in self-supervised learning.
"""

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor

from refrakt_core.losses.templates.base import BaseLoss
from refrakt_core.registry.loss_registry import register_loss


@register_loss("dino")
class DINOLoss(BaseLoss):
    """
    DINO Loss for self-supervised learning.

    This loss computes the cross-entropy between temperature-scaled teacher and
    student outputs using softmax distributions, with a centering mechanism
    for teacher logits to stabilize training.

    Reference:
    Caron et al., DINO: Emerging Properties in Self-Supervised Vision Transformers, 2021.

    Args:
        out_dim (int): Dimensionality of output embeddings.
        teacher_temp (float): Temperature for teacher softmax.
        student_temp (float): Temperature for student softmax.
        center_momentum (float): Momentum for updating the teacher output center.
    """

    def __init__(
        self,
        out_dim: int = 65536,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__(name="DINOLoss")
        self.student_temp: float = student_temp
        self.teacher_temp: float = teacher_temp
        self.center_momentum: float = center_momentum
        self.center: Tensor
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output: Tensor, teacher_output: Tensor) -> Tensor:
        """
        Compute the DINO loss.

        Args:
            student_output (Tensor): Student network outputs of shape (B, num_views, D).
            teacher_output (Tensor): Teacher network outputs of shape (B, 1, D).

        Returns:
            Tensor: Scalar loss value.

        Raises:
            ValueError: If tensor shapes are invalid.
        """
        if student_output.ndim != 3 or teacher_output.ndim != 3:
            raise ValueError("Expected 3D tensors for student and teacher outputs.")
        if student_output.shape[0] != teacher_output.shape[0]:
            raise ValueError("Batch size mismatch.")
        if student_output.shape[2] != teacher_output.shape[2]:
            raise ValueError("Feature dimension mismatch.")
        if teacher_output.shape[1] != 1:
            raise ValueError("Teacher output must have shape (B, 1, D).")

        device = student_output.device
        student_output = student_output.float().to(device)
        teacher_output = teacher_output.float().detach().to(device)

        n_views: int = student_output.shape[1]
        total_loss: Tensor = torch.tensor(0.0, device=device)
        n_loss_terms: int = 0

        teacher_probs: Tensor = F.softmax(
            (teacher_output - self.center) / self.teacher_temp, dim=-1
        )

        for v in range(n_views):
            student_probs: Tensor = F.log_softmax(
                student_output[:, v, :] / self.student_temp, dim=-1
            )
            loss: Tensor = torch.sum(-teacher_probs * student_probs, dim=-1).mean()
            total_loss += loss
            n_loss_terms += 1

        self.update_center(teacher_output)
        return total_loss / n_loss_terms

    @torch.no_grad()
    def update_center(self, teacher_output: Tensor) -> None:
        """
        Momentum update for the teacher's output center.

        Args:
            teacher_output (Tensor): Tensor of shape (B, 1, D).
        """
        batch_center: Tensor = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (
            1.0 - self.center_momentum
        )

    def get_config(self) -> Dict[str, float]:
        """
        Return configuration of the loss function.

        Returns:
            dict: Configuration dictionary.
        """
        return {
            **super().get_config(),
            "out_dim": int(self.center.shape[1]),
            "teacher_temp": self.teacher_temp,
            "student_temp": self.student_temp,
            "center_momentum": self.center_momentum,
        }

    def extra_repr(self) -> str:
        return (
            f"name={self.name}, "
            f"teacher_temp={self.teacher_temp}, "
            f"student_temp={self.student_temp}, "
            f"center_momentum={self.center_momentum}"
        )
