"""The inference code for Refrakt."""
import os
import sys
import traceback
import glob
from typing import Any, Dict, Optional, Union

import torch
from PIL import Image
from omegaconf import OmegaConf

import matplotlib
matplotlib.use("Agg")


from refrakt_core.api.builders.dataloader_builder import build_dataloader
from refrakt_core.api.builders.dataset_builder import build_dataset
from refrakt_core.api.builders.model_builder import build_model
from refrakt_core.api.core.logger import RefraktLogger
from refrakt_core.api.core.utils import import_modules
from refrakt_core.utils.methods import extract_visual_tensor
from refrakt_core.utils.visualize import visualize_latent_space_tensorboard, visualize_latent_space_wandb
# Ensure the refrakt_core package is in the Python path
# Import your existing transform builder
from refrakt_core.api.builders.transform_builder import build_transform

import torchvision.transforms.functional as TF
from PIL import ImageDraw


class CustomImageDataset(torch.utils.data.Dataset):
    """Flexible dataset for RGB or grayscale images based on model input channels."""
    def __init__(self, image_paths, transform=None, expected_channels: int = 3):
        self.image_paths = image_paths
        self.transform = transform
        self.expected_channels = expected_channels  # from config.model.params.in_channels
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])

        # Convert based on expected input channels
        if self.expected_channels == 1:
            img = img.convert("L")  # grayscale
        else:
            img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img

def reshape_if_flattened(tensor: torch.Tensor, input_shape: tuple) -> torch.Tensor:
    """Reshape flattened tensors to image format if possible"""
    if tensor.ndim == 2 and tensor.shape[1] == 784:  # MNIST flattened
        return tensor.view(-1, 1, 28, 28)
    elif tensor.ndim == 2 and tensor.shape[1] == 3072:  # CIFAR-10 flattened
        return tensor.view(-1, 3, 32, 32)
    elif input_shape is not None:
        return tensor.view(-1, *input_shape)
    return tensor


def overlay_label_on_images(images: torch.Tensor, preds: torch.Tensor, class_names=None) -> torch.Tensor:
    """
    Overlay predicted labels on image tensors.

    Args:
        images (torch.Tensor): Batch of images (C,H,W).
        preds (torch.Tensor): Predicted class indices.
        class_names (list or None): Optional list of class names.

    Returns:
        torch.Tensor: Images with labels overlaid.
    """
    imgs = []
    for img, pred in zip(images, preds):
        img_pil = TF.to_pil_image(img)

        # Convert to RGB if grayscale
        if img_pil.mode != "RGB":
            img_pil = img_pil.convert("RGB")

        draw = ImageDraw.Draw(img_pil)
        label = str(pred.item()) if class_names is None else class_names[pred.item()]
        draw.text((5, 5), label, fill=(255, 0, 0))  # Red text
        imgs.append(TF.to_tensor(img_pil))
    return torch.stack(imgs)


def inference(
    cfg: Union[str, OmegaConf],
    model_path: str,
    data: Any = None,
    logger: Optional[RefraktLogger] = None,
) -> Dict[str, Any]:
    """
    Run inference with a trained model.

    Args:
        cfg: Either a path to a config file or an OmegaConf object
        model_path: Path to a saved model checkpoint
        data: Optional data for inference. If None, uses test dataset

    Returns:
        Dict containing inference results
    """

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
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
        model.eval()

        # Handle custom image input
        custom_data = config.get("custom_data")
        input_shape = config.model.params.get("input_shape", None)
        if custom_data:
            logger.info("Using custom data for inference...")
            
            # Get image paths
            if custom_data.get("image_path"):
                image_paths = [custom_data.image_path]
            elif custom_data.get("image_dir"):
                image_dir = custom_data.image_dir
                image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + \
                             glob.glob(os.path.join(image_dir, "*.png")) + \
                             glob.glob(os.path.join(image_dir, "*.jpeg"))
            else:
                raise ValueError("custom_data must contain either image_path or image_dir")
            
            # Build transform using existing function
            transform_config = custom_data.get("transform", [])
            transform = build_transform(transform_config)

            # Get expected input channels from model config
            expected_channels = config.model.params.get("in_channels", 3)

            # Create dataset and dataloader with channel-awareness
            custom_dataset = CustomImageDataset(
                image_paths,
                transform=transform,
                expected_channels=expected_channels
            )
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

        logger.info("\nRunning inference...")
        results, vis_inputs, vis_outputs, vis_targets = [], [], [], []
        max_visualization = 8

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                # Handle custom image dataset (no targets)
                if isinstance(batch, torch.Tensor) or (isinstance(batch, list) and len(batch) == 1):
                    inputs = batch[0] if isinstance(batch, list) else batch
                    inputs = inputs.to(device)
                    targets = None
                elif isinstance(batch, dict):
                    inputs = next((batch[k].to(device) for k in ["image", "lr", "input"] if k in batch), None)
                    if inputs is None:
                        raise KeyError("Expected one of ['image', 'lr', 'input'] in batch dict.")
                    targets = next((batch[k].to(device) for k in ["label", "target", "hr"] if k in batch), None)
                elif isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0].to(device), batch[1].to(device) if len(batch) > 1 else None
                else:
                    inputs, targets = batch.to(device), None

                outputs = model(inputs)

                batch_results = (
                    {k: v.cpu() for k, v in outputs.items()} if isinstance(outputs, dict) else outputs.cpu()
                )
                results.append(batch_results)

                if len(vis_inputs) < max_visualization:
                    vis_inputs.append(inputs.cpu())
                    try:
                        vis_outputs.append(extract_visual_tensor(outputs).cpu())
                    except Exception as e:
                        logger.error(f"Skipping sample due to visualization extraction error: {str(e)}")

                    if targets is not None:
                        vis_targets.append(targets.cpu())

                if batch_idx % 100 == 0:
                    logger.info(f"Processed batch {batch_idx + 1}/{len(data_loader)}")

        # Visualization
        try:
            if vis_inputs:
                inputs_vis = torch.cat(vis_inputs)[:max_visualization]
                outputs_vis = torch.cat(vis_outputs)[:max_visualization]
                targets_vis = torch.cat(vis_targets)[:max_visualization] if vis_targets else None

                # Reshape flattened inputs/outputs
                input_shape = config.model.params.get("input_shape", None)
                inputs_vis = reshape_if_flattened(inputs_vis, input_shape)
                outputs_vis = reshape_if_flattened(outputs_vis, input_shape)
                
                if outputs_vis.ndim == 4:
                    logger.log_inference_results(inputs_vis, outputs_vis, targets_vis, step=0)
                elif outputs_vis.ndim == 2:  # Classification output
                    preds = torch.argmax(outputs_vis, dim=1)

                    class_names = None
                    try:
                        class_names = config.dataset.params.class_names
                        if not class_names:
                            class_names = None
                    except Exception:
                        class_names = None

                    vis_with_labels = overlay_label_on_images(inputs_vis[:preds.size(0)], preds, class_names=class_names)
                    logger.log_images("Predictions", vis_with_labels, step=0)
                    logger.info(f"Logged classification predictions for {preds.size(0)} samples")

                    # Log readable predictions textually
                    if class_names:
                        preds_text = [class_names[pred.item()] for pred in preds]
                    else:
                        preds_text = [str(pred.item()) for pred in preds]
                    logger.info(f"Predicted classes for {preds.size(0)} samples: {preds_text}")

                else:
                    logger.info("Skipping visual logging: unsupported output shape")

        except Exception as e:
            logger.error(f"Inference visualization failed: {str(e)}")

        logger.info("\nInference completed successfully!")
        
        # Optional Latent Visualization (classification or autoencoder)
        try:
            visualize_latent = (
                config.get("runtime", {}).get("visualize_latent", False)
                or config.get("trainer", {}).get("params", {}).get("visualize_latent", False)
            )

            if visualize_latent:
                log_targets = config.runtime.get("log_type", [])
                logger.info(f"Generating latent space projection (targets: {log_targets})...")

                if "tensorboard" in log_targets:
                    from torch.utils.tensorboard import SummaryWriter
                    writer = SummaryWriter(log_dir=config.runtime.get("log_dir", "runs/inference"))
                    visualize_latent_space_tensorboard(model, data_loader, device, writer, step=0, logger=logger)

                if "wandb" in log_targets:
                    import wandb
                    wandb.init(project="refrakt", config=config)
                    visualize_latent_space_wandb(model, data_loader, device, logger=logger)

        except Exception as e:
            logger.warn(f"Latent visualization failed: {str(e)}")
            
        return {"model": model, "results": results, "config": config}

    except Exception as e:
        logger.error(f"\n❌ Inference failed: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)