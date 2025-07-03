import glob
import os
from typing import Any, Dict, Optional, Tuple, Union, cast

import torch
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
