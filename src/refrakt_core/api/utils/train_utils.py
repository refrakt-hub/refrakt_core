import glob
import os
from typing import Any, Dict, Optional, Tuple, Union, cast

import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import Dataset

from refrakt_core.api.builders.dataloader_builder import build_dataloader
from refrakt_core.api.builders.dataset_builder import build_dataset
from refrakt_core.api.builders.model_builder import build_model
from refrakt_core.api.builders.scheduler_builder import build_scheduler
from refrakt_core.api.core.logger import RefraktLogger
from refrakt_core.integrations.fusion.builder import build_fusion_head
from refrakt_core.integrations.fusion.trainer import FusionTrainer
from refrakt_core.schema.artifact import ArtifactDumper

"""
Utility functions for safe model wrapping in Refrakt.

This module provides comprehensive utility functions for training pipeline operations,
including model building, dataset handling, optimizer setup, and training execution.

The module handles:
- Safe model wrapper creation and configuration
- Configuration loading and validation
- Logger setup and configuration
- Dataset analysis and automatic image resizing
- Model building with graph logging
- Optimizer and scheduler configuration
- Artifact dumping and checkpoint management
- Fusion training and ML pipeline handling
- Training execution and metrics logging

These utilities ensure robust training pipeline operations with proper error handling,
automatic dataset optimization, and comprehensive logging capabilities.

Typical usage involves calling these utility functions to set up and execute
complete training pipelines with automatic optimization and logging.
"""


def get_safe_wrapper(
    wrapper_name: str,
    raw_model: Any,
    model_params: Dict[str, Any],
    modules: Dict[str, Any],
    device: Any,
) -> Any:
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
    min_size: Tuple[int, int] = (28, 28),
    target_size: Tuple[int, int] = (224, 224),
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
    from .image_analysis_utils import (
        analyze_image_sizes,
        calculate_size_statistics,
        create_resized_dataset,
    )

    logger.info("🔍 Analyzing dataset image sizes...")

    # Analyze image sizes
    sizes, needs_resize, oversized_count, undersized_count = analyze_image_sizes(
        dataset, max_size, min_size
    )

    if not sizes:
        logger.warning("⚠️ Could not analyze any images in dataset")
        return False, dataset

    # Calculate and log statistics
    avg_width, avg_height, max_width, max_height, min_width, min_height = (
        calculate_size_statistics(sizes)
    )

    logger.info("📈 Image size statistics:")
    logger.info(f"   Average: {avg_width:.1f}x{avg_height:.1f}")
    logger.info(f"   Range: {min_width}x{min_height} to {max_width}x{max_height}")
    logger.info(f"   Oversized images: {oversized_count}")
    logger.info(f"   Undersized images: {undersized_count}")

    if needs_resize:
        logger.info(
            "🔄 Dataset contains images outside acceptable size range \
                (28x28 to 448x448)"
        )
        logger.info(f"📏 Resizing images to {target_size[0]}x{target_size[1]}...")

        # Create resized dataset
        resized_dataset = create_resized_dataset(dataset, target_size)
        logger.info("✅ Dataset resizing complete!")

        return True, resized_dataset
    else:
        logger.info("✅ All images are within acceptable size range (28x28 to 448x448)")
        return False, dataset


def build_datasets_and_loaders_with_resize(
    cfg: DictConfig, logger: RefraktLogger
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
    val_resized, val_dataset = analyze_and_resize_dataset_images(val_dataset, logger)

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
    Load a model checkpoint from a file path.

    Args:
        model: The model to load state dict into
        model_path: Path to the checkpoint file
        device: Device to load the checkpoint on
        logger: Logger instance

    Returns:
        Global step from checkpoint
    """
    if model_path is None:
        logger.warning("No model checkpoint provided — using random init weights")
        return 0
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        logger.info(f"Loaded model from {model_path}")
        return cast(int, checkpoint.get("global_step", 0))
    base_dir = os.path.dirname(model_path)
    base_name = os.path.splitext(os.path.basename(model_path))[0]
    exact_match = os.path.join(base_dir, f"{base_name}.pth")
    if os.path.exists(exact_match):
        checkpoint = torch.load(exact_match, map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        logger.warning(f"⚠️ Falling back to exact checkpoint: {exact_match}")
        return cast(int, checkpoint.get("global_step", 0))
    pattern = os.path.join(base_dir, f"{base_name}_*.pth")
    candidates = glob.glob(pattern)
    if candidates:
        preferred = [c for c in candidates if "latest" in c or "final" in c]
        fallback_path = max(preferred or candidates, key=os.path.getmtime)
        checkpoint = torch.load(fallback_path, map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        logger.warning(f"⚠️ Falling back to available checkpoint: {fallback_path}")
        return cast(int, checkpoint.get("global_step", 0))
    logger.error(f"Model path does not exist: {model_path}")
    raise FileNotFoundError(model_path)


def load_fusion_head(path: str) -> Any:
    """
    Load a fusion head from a joblib file.
    """
    import joblib  # type: ignore[import-untyped]

    return joblib.load(path)


class CustomImageDataset(Dataset[Any]):
    """
    A PyTorch Dataset for loading images from a list of file paths,
    with optional transforms and channel selection.

    Args:
        image_paths (list[str]): List of image file paths.
        transform (callable, optional): Transform to apply to each image.

        expected_channels (int, optional):
        Number of channels (1 for grayscale, 3 for RGB).
        Defaults to 3.
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
        """Load and return an image, applying transform and
        channel conversion if needed."""
        img = Image.open(self.image_paths[idx])
        img = img.convert("L") if self.expected_channels == 1 else img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def setup_data_loader_for_inference_with_resize(
    config: DictConfig, data: Any = None, logger: Optional[RefraktLogger] = None
) -> Any:
    """
    Set up a data loader for inference with automatic image resizing,
    supporting custom data or test dataset.
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
            inference_resized, dataset = analyze_and_resize_dataset_images(
                dataset, logger
            )
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
        inference_resized, test_dataset = analyze_and_resize_dataset_images(
            test_dataset, logger
        )
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


def build_ml_numpy_splits(cfg: DictConfig) -> Tuple[Any, Any, Any, Any]:
    """
    Build X, y numpy arrays for train/val from config for ML pipelines.
    Assumes dataset.name == 'tabular_ml'.
    """
    from omegaconf import DictConfig

    from refrakt_core.api.builders.dataset_builder import build_dataset

    train_cfg = DictConfig(cfg.dataset)
    val_cfg = DictConfig(
        OmegaConf.merge(cfg.dataset, OmegaConf.create({"params": {"train": False}}))
    )
    train_dataset = build_dataset(train_cfg)
    val_dataset = build_dataset(val_cfg)
    X_train, y_train = train_dataset.get_numpy()
    X_val, y_val = val_dataset.get_numpy()
    return X_train, y_train, X_val, y_val


def _resolve_model_name_train(cfg: DictConfig) -> str:
    """Resolve the model name based on configuration for training."""
    if cfg.model.name == "autoencoder":
        variant = cfg.model.params.get("variant", "simple")
        resolved_model_name = f"autoencoder_{variant}"
    else:
        resolved_model_name = cfg.model.name

    # Check if using custom dataset and append _custom suffix
    dataset_params = (
        cfg.dataset.params
        if hasattr(cfg, "dataset") and hasattr(cfg.dataset, "params")
        else {}
    )
    dataset_path = dataset_params.get("path", "") or dataset_params.get("zip_path", "")
    if dataset_path and str(dataset_path).endswith(".zip"):
        resolved_model_name = f"{resolved_model_name}_custom"

    return resolved_model_name


def _handle_pure_ml_training(
    cfg: DictConfig, resolved_model_name: str, logger: RefraktLogger
) -> dict:
    """Handle pure ML pipeline training."""
    from refrakt_core.api.utils.train_utils import build_ml_numpy_splits
    from refrakt_core.integrations.ml.ml_builder import build_ml_pipeline
    from refrakt_core.integrations.ml.trainer import MLTrainer

    # Build ML pipeline
    X_train, y_train, X_val, y_val = build_ml_numpy_splits(cfg)
    feature_pipeline, ml_model, _, _, _, _ = build_ml_pipeline(
        {
            "feature_engineering": getattr(cfg, "feature_engineering", []),
            "model": OmegaConf.to_container(cfg.model, resolve=True),
        },
        X_train,
        y_train,
        X_val,
        y_val,
    )

    # Setup artifact dumper
    artifact_dumper = setup_artifact_dumper(cfg, resolved_model_name, logger)

    # Train ML model
    trainer = MLTrainer(
        feature_pipeline, ml_model, X_train, y_train, X_val, y_val, artifact_dumper
    )  # type: ignore[no-untyped-call]
    metrics = trainer.train()  # type: ignore[no-untyped-call]

    logger.info(f"[ML] Training complete. Metrics: {metrics}")
    print("\nTraining Results:", metrics)

    # Save model and pipeline
    import joblib

    save_dir = (
        cfg.trainer.params.save_dir
        if hasattr(cfg.trainer, "params") and hasattr(cfg.trainer.params, "save_dir")
        else "./checkpoints"
    )
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(
        {"feature_pipeline": feature_pipeline, "model": ml_model},
        os.path.join(save_dir, f"{resolved_model_name}_ml.joblib"),
    )
    logger.info(
        f"[ML] Saved ML pipeline to "
        f"{os.path.join(save_dir, f'{resolved_model_name}_ml.joblib')}"
    )
    return {"status": "completed", "type": "ml"}


def _setup_optimizer_config(cfg: DictConfig) -> Tuple[Any, Dict[str, Any]]:
    """Setup optimizer configuration."""
    from torch import optim

    opt_map = {
        "adam": optim.Adam,
        "sgd": optim.SGD,
        "adamw": optim.AdamW,
        "rmsprop": optim.RMSprop,
    }
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise TypeError(
            f"Config must be a dict after OmegaConf.to_container, got {type(cfg_dict)}"
        )
    optimizer_cfg = cfg_dict.get("optimizer", {})
    opt_name = optimizer_cfg.get("name", "adamw")
    opt_cls = opt_map[opt_name.lower()]
    optimizer_args = optimizer_cfg.get("params", {}) or {}
    return opt_cls, optimizer_args


def _setup_trainer_params(
    cfg: DictConfig,
    device: str,
    logger: RefraktLogger,
    artifact_dumper: Any,
    resolved_model_name: str,
) -> Tuple[Any, Dict[str, Any], int, str]:
    """Setup trainer parameters."""
    from refrakt_core.registry.trainer_registry import get_trainer

    trainer_cls = get_trainer(cfg.trainer.name)
    trainer_params = (
        OmegaConf.to_container(cfg.trainer.params, resolve=True)
        if cfg.trainer.params
        else {}
    )
    if not isinstance(trainer_params, dict):
        trainer_params = {}
    trainer_params = cast(Dict[str, Any], trainer_params)
    num_epochs = trainer_params.pop("num_epochs", 10)  # Changed default from 1 to 10
    device_param = trainer_params.pop("device", device)
    final_device = device_param or device
    trainer_params["logger"] = logger
    trainer_params["artifact_dumper"] = artifact_dumper
    trainer_params["model_name"] = resolved_model_name

    return trainer_cls, trainer_params, num_epochs, final_device


def _handle_fusion_training(
    cfg: DictConfig,
    model: torch.nn.Module,
    train_loader: Any,
    val_loader: Any,
    device: str,
    artifact_dumper: Any,
    trainer: Any,
    logger: RefraktLogger,
) -> None:
    """Handle fusion head training if configured."""
    if not hasattr(cfg.model, "fusion"):
        return

    logger.info(
        "\n[FUSION] Fusion head config detected. Starting fusion head training..."
    )
    fusion_cfg = cfg.model.fusion
    fusion_head = build_fusion_head(fusion_cfg)
    fusion_trainer = FusionTrainer(
        model=model,
        fusion_head=fusion_head,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        artifact_dumper=artifact_dumper,
        model_name=cfg.trainer.params.model_name,
    )
    fusion_metrics = fusion_trainer.train()
    fusion_save_path = os.path.join(
        cfg.trainer.params.save_dir,
        f"{cfg.trainer.params.model_name}_fusion.joblib",
    )
    save_method = getattr(fusion_head, "save", None)
    if callable(save_method):
        save_method(fusion_save_path)
        logger.info(f"[FUSION] Fusion head saved to {fusion_save_path}")
    if logger:
        logger.log_metrics(fusion_metrics, step=trainer.global_step, prefix="fusion")


def _save_config_and_log_metrics(
    cfg: DictConfig,
    trainer: Any,
    resolved_model_name: str,
    final_metrics: Dict[str, Any],
    logger: RefraktLogger,
) -> None:
    """Save configuration and log final metrics."""
    # Save Config
    config_save_path = os.path.join(
        getattr(trainer, "save_dir", os.path.join("./artifacts", "yaml")),
        f"{resolved_model_name}.yaml",
    )
    OmegaConf.save(cfg, config_save_path)
    logger.info(f"Saved config to {config_save_path}")

    # Log Final Metrics
    print("\nFinal Metrics:", final_metrics)
    if logger:
        logger.log_metrics(final_metrics, step=trainer.global_step, prefix="final")
    logger.info("\n✅ Training completed successfully!")
