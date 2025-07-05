import gc
import glob
import os
import sys
import traceback
from typing import Any, Dict, Optional, Tuple, Union, cast

import torch
import torchvision.transforms
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from refrakt_core.api.builders.dataloader_builder import build_dataloader
from refrakt_core.api.builders.dataset_builder import build_dataset
from refrakt_core.api.builders.loss_builder import build_loss
from refrakt_core.api.builders.model_builder import build_model
from refrakt_core.api.builders.scheduler_builder import build_scheduler
from refrakt_core.api.core.logger import RefraktLogger
from refrakt_core.registry.model_registry import get_model
from refrakt_core.registry.wrapper_registry import get_wrapper
from refrakt_core.schema.artifact import ArtifactDumper
from torch.utils.data import Dataset

from refrakt_core.transforms.image_resizer import SmartImageResizer, ImageSizeConfig
from refrakt_core.transforms.standard_transforms import StandardImageTransform

"""
Utility functions for safe model wrapping in Refrakt.
"""


def get_safe_wrapper(
    wrapper_name: str,
    raw_model: object,
    model_params: dict,
    modules: dict,
    device: object,
) -> object:
    """
    Safely wrap a model using the specified wrapper and parameters.

    Args:
        wrapper_name (str): Name of the wrapper to use.
        raw_model (object): The model to wrap.
        model_params (dict): Parameters for the wrapper.
        modules (dict): Module registry with 'get_wrapper'.
        device (object): Device to move the wrapped model to.

    Returns:
        object: The wrapped model on the specified device.
    """
    import inspect

    wrapper_cls = modules["get_wrapper"](wrapper_name)
    sig = inspect.signature(wrapper_cls.__init__)
    valid_args = set(sig.parameters.keys()) - {"self", "model"}
    wrapper_args = {k: v for k, v in model_params.items() if k in valid_args}
    return wrapper_cls(model=raw_model, **wrapper_args).to(device)


def load_config(cfg: Union[str, DictConfig]) -> DictConfig:
    """
    Load an OmegaConf config from a file path or return as-is if already a DictConfig.
    """
    loaded = OmegaConf.load(cfg) if isinstance(cfg, str) else cfg
    if not isinstance(loaded, DictConfig):
        raise TypeError(f"Config must be a DictConfig, got {type(loaded)}")
    return loaded


def setup_logger(cfg: DictConfig, model_name: str) -> RefraktLogger:
    """
    Set up a RefraktLogger from config and model name.
    """
    runtime_cfg = cfg.get("runtime", {})
    log_types = runtime_cfg.get("log_type", [])
    log_dir = runtime_cfg.get("log_dir", "./logs")
    console = runtime_cfg.get("console", True)
    debug = runtime_cfg.get("debug", False)
    return RefraktLogger(
        model_name=model_name,
        log_dir=log_dir,
        log_types=log_types,
        console=console,
        debug=debug,
    )


def analyze_and_resize_dataset_images(
    dataset: Any,
    logger: RefraktLogger,
    max_size: Tuple[int, int] = (448, 448),
    min_size: Tuple[int, int] = (32, 32),
    target_size: Tuple[int, int] = (224, 224)
) -> Tuple[bool, Any]:
    """
    Analyze dataset image sizes and resize if needed.

    Args:
        dataset: The dataset to analyze
        logger: Logger instance for logging resize operations
        max_size: Maximum allowed image size
        min_size: Minimum allowed image size
        target_size: Target size for resizing

    Returns:
        Tuple of (needs_resize, modified_dataset)
    """
    logger.info("🔍 Analyzing dataset image sizes...")

    # Initialize size validator and resizer
    size_config = ImageSizeConfig(
        standard_size=target_size,
        max_size=max_size,
        min_size=min_size
    )
    resizer = SmartImageResizer(size_config)

    # Sample images to analyze sizes
    sample_count = min(100, len(dataset))  # Sample up to 100 images
    sample_indices = list(range(0, len(dataset), max(1, len(dataset) // sample_count)))[:sample_count]

    sizes = []
    needs_resize = False
    oversized_count = 0
    undersized_count = 0

    logger.info(f"📊 Sampling {len(sample_indices)} images for size analysis...")

    for idx in sample_indices:
        try:
            # Get image from dataset
            sample = dataset[idx]
            if isinstance(sample, (tuple, list)):
                # Handle (image, label) format
                image = sample[0]
            elif isinstance(sample, dict):
                # Handle dict format (e.g., {'lr': tensor, 'hr': tensor})
                image = list(sample.values())[0]
            else:
                image = sample

            # Convert tensor to PIL if needed
            if isinstance(image, torch.Tensor):
                if image.dim() == 3:  # (C, H, W)
                    size = (image.size(2), image.size(1))  # (W, H)
                else:  # (H, W)
                    size = (image.size(1), image.size(0))  # (W, H)
            else:
                size = image.size  # PIL Image

            sizes.append(size)

            # Check if size is outside acceptable range
            width, height = size
            if width > max_size[0] or height > max_size[1]:
                oversized_count += 1
                needs_resize = True
            elif width < min_size[0] or height < min_size[1]:
                undersized_count += 1
                needs_resize = True

        except Exception as e:
            logger.warning(f"⚠️ Could not analyze image at index {idx}: {e}")
            continue

    if not sizes:
        logger.warning("⚠️ Could not analyze any images in dataset")
        return False, dataset

    # Calculate statistics
    avg_width = sum(s[0] for s in sizes) / len(sizes)
    avg_height = sum(s[1] for s in sizes) / len(sizes)
    max_width = max(s[0] for s in sizes)
    max_height = max(s[1] for s in sizes)
    min_width = min(s[0] for s in sizes)
    min_height = min(s[1] for s in sizes)

    logger.info(f"📈 Image size statistics:")
    logger.info(f"   Average: {avg_width:.1f}x{avg_height:.1f}")
    logger.info(f"   Range: {min_width}x{min_height} to {max_width}x{max_height}")
    logger.info(f"   Oversized images: {oversized_count}")
    logger.info(f"   Undersized images: {undersized_count}")

    if needs_resize:
        logger.info("🔄 Dataset contains images outside acceptable size range (32x32 to 448x448)")
        logger.info(f"📏 Resizing images to {target_size[0]}x{target_size[1]}...")

        # Create a wrapper dataset that resizes images on-the-fly
        class ResizedDataset(Dataset):
            def __init__(self, original_dataset, resizer, target_size):
                self.original_dataset = original_dataset
                self.resizer = resizer
                self.target_size = target_size

            def __len__(self):
                return len(self.original_dataset)

            def __getitem__(self, idx):
                sample = self.original_dataset[idx]

                if isinstance(sample, (tuple, list)):
                    # Handle (image, label) format
                    image, *rest = sample
                    resized_image = self._resize_image(image)
                    return (resized_image, *rest)
                elif isinstance(sample, dict):
                    # Handle dict format
                    resized_sample = {}
                    for key, value in sample.items():
                        if isinstance(value, torch.Tensor) and value.dim() >= 2:
                            resized_sample[key] = self._resize_image(value)
                        else:
                            resized_sample[key] = value
                    return resized_sample
                else:
                    # Handle single image
                    return self._resize_image(sample)

            def _resize_image(self, image):
                """Resize image using SmartImageResizer"""
                if isinstance(image, torch.Tensor):
                    # Convert tensor to PIL for resizing
                    if image.dim() == 3:  # (C, H, W)
                        pil_image = torchvision.transforms.ToPILImage()(image)
                    else:  # (H, W)
                        pil_image = torchvision.transforms.ToPILImage()(image.unsqueeze(0))

                    # Resize using SmartImageResizer's internal method to bypass validation
                    resized_pil = self.resizer._resize_maintain_aspect(
                        pil_image,
                        self.target_size
                    )

                    # Convert back to tensor
                    return torchvision.transforms.ToTensor()(resized_pil)
                else:
                    # PIL Image - use internal method to bypass validation
                    return self.resizer._resize_maintain_aspect(
                        image,
                        self.target_size
                    )

        # Create resized dataset
        resized_dataset = ResizedDataset(dataset, resizer, target_size)
        logger.info("✅ Dataset resizing complete!")

        return True, resized_dataset
    else:
        logger.info("✅ All images are within acceptable size range (32x32 to 448x448)")
        return False, dataset


def build_datasets_and_loaders_with_resize(
    cfg: DictConfig,
    logger: RefraktLogger
) -> Tuple[Any, Any, Any, Any]:
    """
    Build train/val datasets and dataloaders from config with automatic image resizing.
    """
    if not isinstance(cfg.dataset, DictConfig):
        raise TypeError("cfg.dataset must be a DictConfig")

    # Build original datasets
    train_dataset = build_dataset(cfg.dataset)
    val_cfg = OmegaConf.merge(
        cfg.dataset, OmegaConf.create({"params": {"train": False}})
    )
    if not isinstance(val_cfg, DictConfig):
        raise TypeError("val_cfg must be a DictConfig")
    val_dataset = build_dataset(val_cfg)

    # Analyze and resize if needed
    train_resized, train_dataset = analyze_and_resize_dataset_images(
        train_dataset, logger
    )
    val_resized, val_dataset = analyze_and_resize_dataset_images(
        val_dataset, logger
    )

    if train_resized or val_resized:
        logger.info("🔄 Using resized datasets for training")

    # Build dataloaders
    train_loader = build_dataloader(train_dataset, cfg.dataloader)
    val_loader = build_dataloader(val_dataset, cfg.dataloader)

    return train_dataset, val_dataset, train_loader, val_loader


def build_datasets_and_loaders(cfg: DictConfig) -> Tuple[Any, Any, Any, Any]:
    """
    Build train/val datasets and dataloaders from config.
    """
    if not isinstance(cfg.dataset, DictConfig):
        raise TypeError("cfg.dataset must be a DictConfig")
    train_dataset = build_dataset(cfg.dataset)
    val_cfg = OmegaConf.merge(
        cfg.dataset, OmegaConf.create({"params": {"train": False}})
    )
    if not isinstance(val_cfg, DictConfig):
        raise TypeError("val_cfg must be a DictConfig")
    val_dataset = build_dataset(val_cfg)
    train_loader = build_dataloader(train_dataset, cfg.dataloader)
    val_loader = build_dataloader(val_dataset, cfg.dataloader)
    return train_dataset, val_dataset, train_loader, val_loader


def build_model_and_log_graph(
    cfg: DictConfig,
    modules: Dict[str, Any],
    device: str,
    train_loader: Any,
    logger: RefraktLogger,
) -> Any:
    """
    Build the model and log its graph using a sample batch.
    """
    model_cls = modules["get_model"](cfg.model.name)
    model = build_model(
        cast(OmegaConf, cfg),
        modules={
            "get_model": modules["get_model"],
            "get_wrapper": modules["get_wrapper"],
            "model": model_cls,
        },
        device=device,
    )
    # Log model graph
    try:
        sample_batch = next(iter(train_loader))
        sample_input = (
            sample_batch[0] if isinstance(sample_batch, (tuple, list)) else sample_batch
        )
        if isinstance(sample_input, dict):
            sample_input = {k: v.to(device) for k, v in sample_input.items()}
        else:
            sample_input = sample_input.to(device)
        logger.log_model_graph(model, sample_input)
    except Exception as e:
        logger.error(f"Model graph logging failed: {str(e)}")
    return model


def build_optimizer_and_scheduler(cfg: DictConfig, model: Any) -> Tuple[Any, Any]:
    """
    Build optimizer and scheduler from config and model.
    """
    from refrakt_core.api.builders.optimizer_builder import build_optimizer

    optimizer = build_optimizer(cfg, model)
    scheduler = (
        build_scheduler(cast(OmegaConf, cfg), optimizer)
        if cfg.get("scheduler")
        else None
    )
    return optimizer, scheduler


def setup_artifact_dumper(
    cfg: DictConfig, model_name: str, logger: RefraktLogger
) -> ArtifactDumper:
    """
    Set up an ArtifactDumper from config, model name, and logger.
    """
    artifact_log_every = cfg.get("artifacts", {}).get("log_every", 1)
    artifact_enabled = cfg.get("artifacts", {}).get("enabled", True)
    return ArtifactDumper(
        enabled=artifact_enabled,
        base_path="./artifacts",
        model_name=model_name,
        log_every=artifact_log_every,
        logger=logger,
    )


def load_checkpoint(
    model: torch.nn.Module, model_path: Optional[str], device: torch.device, logger: Any
) -> int:
    """
    Load a model checkpoint, with fallback logic for missing files and variants.
    """
    if model_path is None:
        logger.warning("No model checkpoint provided — using random init weights")
        return 0
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        logger.info(f"Loaded model from {model_path}")
        return checkpoint.get("global_step", 0)
    base_dir = os.path.dirname(model_path)
    base_name = os.path.splitext(os.path.basename(model_path))[0]
    exact_match = os.path.join(base_dir, f"{base_name}.pth")
    if os.path.exists(exact_match):
        checkpoint = torch.load(exact_match, map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        logger.warning(f"⚠️ Falling back to exact checkpoint: {exact_match}")
        return checkpoint.get("global_step", 0)
    pattern = os.path.join(base_dir, f"{base_name}_*.pth")
    candidates = glob.glob(pattern)
    if candidates:
        preferred = [c for c in candidates if "latest" in c or "final" in c]
        fallback_path = max(preferred or candidates, key=os.path.getmtime)
        checkpoint = torch.load(fallback_path, map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        logger.warning(f"⚠️ Falling back to available checkpoint: {fallback_path}")
        return checkpoint.get("global_step", 0)
    logger.error(f"Model path does not exist: {model_path}")
    raise FileNotFoundError(model_path)


def load_fusion_head(path: str) -> Any:
    """
    Load a fusion head from a joblib file.
    """
    import joblib

    return joblib.load(path)


class CustomImageDataset(Dataset):
    """
    A PyTorch Dataset for loading images from a list of file paths, with optional transforms and channel selection.

    Args:
        image_paths (list[str]): List of image file paths.
        transform (callable, optional): Transform to apply to each image.
        expected_channels (int, optional): Number of channels (1 for grayscale, 3 for RGB). Defaults to 3.
    """

    def __init__(
        self,
        image_paths: list[str],
        transform: Optional[Any] = None,
        expected_channels: int = 3,
    ) -> None:
        self.image_paths = image_paths
        self.transform = transform
        self.expected_channels = expected_channels

    def __len__(self) -> int:
        """Return the number of images."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Any:
        """Load and return an image, applying transform and channel conversion if needed."""
        img = Image.open(self.image_paths[idx])
        img = img.convert("L") if self.expected_channels == 1 else img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def setup_data_loader_for_inference_with_resize(config: DictConfig, data: Any = None, logger: Optional[RefraktLogger] = None) -> Any:
    """
    Set up a data loader for inference with automatic image resizing, supporting custom data or test dataset.
    """
    if data is not None:
        return data
    custom_data = config.get("custom_data")
    if custom_data:
        if custom_data.get("image_path"):
            image_paths = [custom_data.image_path]
        elif custom_data.get("image_dir"):
            image_dir = custom_data.image_dir
            image_paths = (
                glob.glob(os.path.join(image_dir, "*.jpg"))
                + glob.glob(os.path.join(image_dir, "*.png"))
                + glob.glob(os.path.join(image_dir, "*.jpeg"))
            )
        else:
            raise ValueError("custom_data must contain either image_path or image_dir")
        from refrakt_core.api.builders.transform_builder import build_transform

        transform = build_transform(custom_data.get("transform", []))
        expected_channels = config.model.params.get("in_channels", 3)
        dataset = CustomImageDataset(image_paths, transform, expected_channels)
        
        # Apply resizing if logger is provided
        if logger is not None:
            inference_resized, dataset = analyze_and_resize_dataset_images(dataset, logger)
            if inference_resized:
                logger.info("🔄 Using resized inference dataset")
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=config.dataloader.params.get("batch_size", 1),
            shuffle=False,
            num_workers=config.dataloader.params.get("num_workers", 0),
        )
    test_cfg = OmegaConf.merge(
        config.dataset, OmegaConf.create({"params": {"train": False}})
    )
    if not isinstance(test_cfg, DictConfig):
        raise TypeError("test_cfg must be a DictConfig")
    test_dataset = build_dataset(test_cfg)
    
    # Apply resizing if logger is provided
    if logger is not None:
        inference_resized, test_dataset = analyze_and_resize_dataset_images(test_dataset, logger)
        if inference_resized:
            logger.info("🔄 Using resized inference dataset")
    
    return build_dataloader(test_dataset, config.dataloader)


def setup_data_loader_for_inference(config: DictConfig, data: Any = None) -> Any:
    """
    Set up a data loader for inference, supporting custom data or test dataset.
    """
    if data is not None:
        return data
    custom_data = config.get("custom_data")
    if custom_data:
        if custom_data.get("image_path"):
            image_paths = [custom_data.image_path]
        elif custom_data.get("image_dir"):
            image_dir = custom_data.image_dir
            image_paths = (
                glob.glob(os.path.join(image_dir, "*.jpg"))
                + glob.glob(os.path.join(image_dir, "*.png"))
                + glob.glob(os.path.join(image_dir, "*.jpeg"))
            )
        else:
            raise ValueError("custom_data must contain either image_path or image_dir")
        from refrakt_core.api.builders.transform_builder import build_transform

        transform = build_transform(custom_data.get("transform", []))
        expected_channels = config.model.params.get("in_channels", 3)
        dataset = CustomImageDataset(image_paths, transform, expected_channels)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=config.dataloader.params.get("batch_size", 1),
            shuffle=False,
            num_workers=config.dataloader.params.get("num_workers", 0),
        )
    test_cfg = OmegaConf.merge(
        config.dataset, OmegaConf.create({"params": {"train": False}})
    )
    if not isinstance(test_cfg, DictConfig):
        raise TypeError("test_cfg must be a DictConfig")
    test_dataset = build_dataset(test_cfg)
    return build_dataloader(test_dataset, config.dataloader)


def build_ml_numpy_splits(cfg: DictConfig):
    """
    Build X, y numpy arrays for train/val from config for ML pipelines.
    Assumes dataset.name == 'tabular_ml'.
    """
    from refrakt_core.api.builders.dataset_builder import build_dataset
    from omegaconf import DictConfig
    train_cfg = DictConfig(cfg.dataset)
    val_cfg = DictConfig(OmegaConf.merge(cfg.dataset, OmegaConf.create({"params": {"train": False}})))
    train_dataset = build_dataset(train_cfg)
    val_dataset = build_dataset(val_cfg)
    X_train, y_train = train_dataset.get_numpy()
    X_val, y_val = val_dataset.get_numpy()
    return X_train, y_train, X_val, y_val
