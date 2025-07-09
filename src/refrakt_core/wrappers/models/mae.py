from typing import Any, Dict

from torch import Tensor, nn

from refrakt_core.registry.wrapper_registry import register_wrapper
from refrakt_core.schema.model_output import ModelOutput


@register_wrapper("mae")
class MAEWrapper(nn.Module):
    """
    Wrapper for MAE model that converts output to standardized ModelOutput format.

    Args:
        model (nn.Module): The already-initialized MAE model
    """

    def __init__(self, model: nn.Module, **kwargs: Any) -> None:
        super().__init__()
        self.model = model
        self.expected_input_dim = getattr(
            model, "img_size", (3, 224, 224)
        )  # fallback if not present

    def _unpatchify(self, patches: Tensor, target: Tensor) -> Tensor:
        """
        Convert (B, N, patch_dim) back to (B, C, H, W) using shape info from target.
        """
        B, N, patch_dim = patches.shape
        _, _, target_dim = target.shape
        patch_size = int((target_dim // 3) ** 0.5)
        H = W = int(N**0.5)

        return (
            patches.reshape(B, H, W, patch_size, patch_size, 3)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(B, 3, H * patch_size, W * patch_size)
        )

    def forward(self, x: Tensor) -> ModelOutput:
        model_output: Dict[str, Tensor] = self.model(x)
        recon = model_output["recon"]
        patches = model_output["original_patches"]

        if recon.ndim == 3:
            recon = self._unpatchify(recon, target=patches)

        return ModelOutput(
            reconstruction=recon,
            extra={"mask": model_output["mask"], "original_patches": patches},
        )
