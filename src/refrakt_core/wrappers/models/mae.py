# Add to mae.py (the file containing MAELossWrapper)
from torch import nn, Tensor
from refrakt_core.registry.wrapper_registry import register_wrapper
from refrakt_core.schema.model_output import ModelOutput

@register_wrapper("mae")
class MAEWrapper(nn.Module):
    """
    Wrapper for MAE model that converts output to standardized ModelOutput format.
    
    Args:
        model_name (str): Registered name of the base model ('mae')
        model_params (dict): Parameters for the base MAE model
    """

    def __init__(self, model_name: str, model_params: dict):
        self.expected_input_dim = (3, model_params['img_size'], model_params['img_size'])

        super().__init__()
        if model_name != "mae":
            raise ValueError(f"MAEModelWrapper requires model_name='mae', got '{model_name}'")
        
        # Get base model from registry
        from refrakt_core.registry.model_registry import MODEL_REGISTRY
        self.model = MODEL_REGISTRY[model_name](**model_params)
        
    def _unpatchify(self, patches: Tensor, target: Tensor) -> Tensor:
        """
        Convert (B, N, patch_dim) back to (B, C, H, W) using shape info from target.
        """
        B, N, patch_dim = patches.shape
        _, _, target_dim = target.shape
        patch_size = int((target_dim // 3) ** 0.5)
        H = W = int(N ** 0.5)

        return patches.reshape(B, H, W, patch_size, patch_size, 3).permute(0, 5, 1, 3, 2, 4).reshape(B, 3, H * patch_size, W * patch_size)


    def forward(self, x: Tensor) -> ModelOutput:
        model_output = self.model(x)
        recon = model_output["recon"]
        patches = model_output["original_patches"]

        if recon.ndim == 3:
            recon = self._unpatchify(recon, target=patches)

        return ModelOutput(
            reconstruction=recon,
            extra={
                "mask": model_output["mask"],
                "original_patches": patches
            }
        )
