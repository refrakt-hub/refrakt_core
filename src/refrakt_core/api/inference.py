"""The inference code for Refrakt (cleaned version)."""

import os
import sys
import traceback
import glob
from typing import Any, Dict, Optional, Union

import torch
from PIL import Image
from omegaconf import OmegaConf

from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.api.builders.dataloader_builder import build_dataloader
from refrakt_core.api.builders.dataset_builder import build_dataset
from refrakt_core.api.builders.model_builder import build_model
from refrakt_core.api.core.logger import RefraktLogger
from refrakt_core.api.core.utils import import_modules
from refrakt_core.api.builders.transform_builder import build_transform


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None, expected_channels: int = 3):
        self.image_paths = image_paths
        self.transform = transform
        self.expected_channels = expected_channels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])
        img = img.convert("L") if self.expected_channels == 1 else img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def inference(
    cfg: Union[str, OmegaConf],
    model_path: str,
    data: Any = None,
    logger: Optional[RefraktLogger] = None,
) -> Dict[str, Any]:
    """Run inference with a trained model and return raw outputs."""

    if logger is None:
        logger = RefraktLogger("./logs", console=True)

    try:
        config = OmegaConf.load(cfg) if isinstance(cfg, str) else cfg
        modules = import_modules()

        device = (
            config.trainer.params.device
            if hasattr(config, "trainer") and hasattr(config.trainer, "params") and hasattr(config.trainer.params, "device")
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Using device: {device}")

        model = build_model(config, modules, device)

        if not os.path.exists(model_path):
            logger.warning(f"Model checkpoint not found at: {model_path}")
            base_name = os.path.splitext(os.path.basename(model_path))[0]
            search_pattern = os.path.join(os.path.dirname(model_path), f"{base_name}_*.pth")
            candidates = sorted(glob.glob(search_pattern))
            if candidates:
                model_path = max(candidates, key=os.path.getmtime)
                logger.warning(f"⚠️ Falling back to available checkpoint: {model_path}")
            else:
                raise FileNotFoundError(f"❌ No matching checkpoint found for pattern: {search_pattern}")

        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        model.eval()

        # Dataloader setup
        custom_data = config.get("custom_data")
        if custom_data:
            logger.info("Using custom data for inference...")
            if custom_data.get("image_path"):
                image_paths = [custom_data.image_path]
            elif custom_data.get("image_dir"):
                image_dir = custom_data.image_dir
                image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + \
                              glob.glob(os.path.join(image_dir, "*.png")) + \
                              glob.glob(os.path.join(image_dir, "*.jpeg"))
            else:
                raise ValueError("custom_data must contain either image_path or image_dir")

            transform_config = custom_data.get("transform", [])
            transform = build_transform(transform_config)
            expected_channels = config.model.params.get("in_channels", 3)
            custom_dataset = CustomImageDataset(image_paths, transform, expected_channels)
            data_loader = torch.utils.data.DataLoader(
                custom_dataset,
                batch_size=config.dataloader.params.get("batch_size", 1),
                shuffle=False,
                num_workers=config.dataloader.params.get("num_workers", 0)
            )

        elif data is None:
            logger.info("No data provided, using test dataset...")
            test_cfg = OmegaConf.merge(config.dataset, OmegaConf.create({"params": {"train": False}}))
            test_dataset = build_dataset(test_cfg)
            data_loader = build_dataloader(test_dataset, config.dataloader)
        else:
            data_loader = data

        logger.info("Running inference...")
        results = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if isinstance(batch, torch.Tensor) or (isinstance(batch, list) and len(batch) == 1):
                    inputs = batch[0] if isinstance(batch, list) else batch
                    inputs = inputs.to(device)
                elif isinstance(batch, dict):
                    inputs = next((batch[k].to(device) for k in ["image", "lr", "input"] if k in batch), None)
                    if inputs is None:
                        raise KeyError("Expected one of ['image', 'lr', 'input'] in batch dict.")
                elif isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(device)
                else:
                    inputs = batch.to(device)

                if inputs.dim() != 4:
                    raise ValueError(f"Expected 4D image tensor (B, C, H, W), got {inputs.shape}")

                if hasattr(model, 'generator'):
                    outputs = model.generator(inputs)
                else:
                    outputs = model(inputs)

                if isinstance(outputs, dict):
                    result = {k: v.cpu() for k, v in outputs.items()}
                elif isinstance(outputs, torch.Tensor):
                    result = {"logits": outputs.cpu()}
                elif isinstance(outputs, ModelOutput):
                    result = outputs.__dict__  # or serialize relevant fields only
                    result = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in result.items()}
                else:
                    raise ValueError(f"Unexpected model output type: {type(outputs)}")

                results.append(result)


        logger.info("✅ Inference completed successfully.")
        return {"model": model, "results": results, "config": config}

    except Exception as e:
        logger.error(f"\n❌ Inference failed: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
