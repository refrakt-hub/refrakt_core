"""The inference code for Refrakt (with optional fusion head support)."""

import glob
import os
import re
import sys
import traceback
import warnings
from datetime import datetime
from typing import Any, Dict, Optional, Union

import torch
from omegaconf import OmegaConf
from PIL import Image

from refrakt_core.api.builders.dataloader_builder import build_dataloader
from refrakt_core.api.builders.dataset_builder import build_dataset
from refrakt_core.api.builders.model_builder import build_model
from refrakt_core.api.builders.transform_builder import build_transform
from refrakt_core.api.core.logger import RefraktLogger
from refrakt_core.api.core.utils import import_modules
from refrakt_core.global_logging import get_global_logger
from refrakt_core.integrations.sklearn.wrapper import SklearnWrapper
from refrakt_core.registry.model_registry import get_model
from refrakt_core.registry.wrapper_registry import get_wrapper
from refrakt_core.schema.model_output import ModelOutput

warnings.filterwarnings("ignore")


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


def load_fusion_head(path: str) -> Any:
    import joblib

    return joblib.load(path)


def inference(
    cfg: Union[str, OmegaConf],
    model_path: str,
    fusion_head_path: Optional[str] = None,
    data: Any = None,
    logger: Optional[RefraktLogger] = None,
) -> Dict[str, Any]:
    """Run inference with a trained model and optionally a fusion head."""

    try:
        config = OmegaConf.load(cfg) if isinstance(cfg, str) else cfg
        runtime_cfg = config.get("runtime", {})
        log_types = runtime_cfg.get("log_type", [])
        log_dir = runtime_cfg.get("log_dir", "./logs")
        mode = runtime_cfg.get("mode", "inference")
        console = runtime_cfg.get("console", True)
        debug = runtime_cfg.get("debug", False)

        if config.model.name == "autoencoder":
            variant = config.model.params.get("variant", "simple")
            if variant not in {"simple", "vae"}:
                raise ValueError(f"Unsupported autoencoder variant: {variant!r}")

            resolved_model_name = f"autoencoder_{variant}"
            print(f"[Resolved] Using model checkpoint name: {resolved_model_name}")
        else:
            resolved_model_name = config.model.name

        logger = logger or RefraktLogger(
            model_name=resolved_model_name,
            log_dir=log_dir,
            log_types=log_types,
            console=console,
            debug=debug,
        )
        logger.log_config(OmegaConf.to_container(config, resolve=True))

        modules = import_modules()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        model_cls = get_model(config.model.name)
        model = build_model(
            config,
            modules={
                "get_model": get_model,
                "get_wrapper": get_wrapper,
                "model": model_cls,
            },
            device=device,
        )

        # Log model structure for debugging
        # print("\nModel state dict keys:")
        # model_state = model.state_dict()
        # for k in sorted(model_state.keys()):
        #     print(f"  {k}: {model_state[k].shape}")

        if not os.path.exists(model_path):
            base_path = os.path.splitext(model_path)[0]
            candidates = glob.glob(f"{base_path}_*.pth")
            if candidates:
                model_path = max(candidates, key=os.path.getmtime)
                logger.warning(f"Using available checkpoint: {model_path}")
            else:
                raise FileNotFoundError(f"No model found at {model_path}")

        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Load state dict directly without transformation
        try:
            model.load_state_dict(state_dict)
            logger.info("Successfully loaded state dict")
        except RuntimeError as e:
            # Silently try loading with strict=False without logging warnings
            try:
                model.load_state_dict(state_dict, strict=False)
                logger.info("Successfully loaded state dict (some keys ignored)")
            except RuntimeError as e2:
                logger.error(f"All attempts to load state dict failed: {str(e2)}")
                raise

        model.eval()

        # Load fusion head if path provided and exists
        fusion_head = None
        if fusion_head_path:
            if os.path.exists(fusion_head_path):
                fusion_head = load_fusion_head(fusion_head_path)
                logger.info(f"Loaded fusion head from {fusion_head_path}")
            else:
                logger.warning(f"Fusion head path does not exist: {fusion_head_path}")

        # Data Loader Setup
        custom_data = config.get("custom_data")
        if custom_data:
            logger.info("Using custom data for inference...")
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
                raise ValueError(
                    "custom_data must contain either image_path or image_dir"
                )

            transform = build_transform(custom_data.get("transform", []))
            expected_channels = config.model.params.get("in_channels", 3)
            dataset = CustomImageDataset(image_paths, transform, expected_channels)
            data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=config.dataloader.params.get("batch_size", 1),
                shuffle=False,
                num_workers=config.dataloader.params.get("num_workers", 0),
            )
        elif data is None:
            logger.info("No data provided, using test dataset...")
            test_cfg = OmegaConf.merge(
                config.dataset, OmegaConf.create({"params": {"train": False}})
            )
            test_dataset = build_dataset(test_cfg)
            data_loader = build_dataloader(test_dataset, config.dataloader)
        else:
            data_loader = data

        from refrakt_core.schema.artifact import ArtifactDumper

        artifact_dumper = ArtifactDumper(
            enabled=True,
            base_path="./artifacts",
            model_name=resolved_model_name,
            log_every=1,
            logger=logger,
        )

        def ensure_4d_tensor(tensor):
            if tensor.dim() == 2:  # Flattened [B, 784]
                return tensor.view(-1, 1, 28, 28)
            elif tensor.dim() == 4:
                return tensor
            raise ValueError(f"Expected 2D or 4D tensor, got {tensor.shape}")

        results = []

        if hasattr(logger, "wandb") and hasattr(logger, "step"):
            logger.wandb.step = 0

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if isinstance(batch, torch.Tensor):
                    inputs = batch
                elif isinstance(batch, dict):
                    inputs = batch.get("input") or batch.get("image") or batch.get("lr")
                    if inputs is None:
                        raise ValueError(
                            f"Inference input could not be resolved from batch keys: {list(batch.keys())}"
                        )
                elif isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    raise TypeError(f"Unsupported batch type: {type(batch)}")

                inputs = inputs.to(device)
                inputs = ensure_4d_tensor(inputs)

                output = model(inputs)  # Should be ModelOutput

                fusion_cfg = config.model.get("fusion")
                if (
                    fusion_cfg
                    and isinstance(output, ModelOutput)
                    and output.embeddings is not None
                ):
                    fusion_model_key = fusion_cfg.model
                    fusion_type = fusion_cfg.type
                    fusion_path = os.path.join(
                        config.trainer.params.save_dir,
                        f"{config.model.name}_fusion.joblib",
                    )

                    if os.path.exists(fusion_path):
                        # logger.info(f"[FUSION] Loading fusion head from {fusion_path}")

                        if fusion_type == "sklearn":
                            fusion_head = SklearnWrapper.load(
                                fusion_model_key, fusion_path
                            )
                        elif fusion_type == "cuml":
                            from refrakt_core.integrations.cuml.wrapper import CuMLWrapper
                            fusion_head = CuMLWrapper.load(
                                fusion_model_key, fusion_path
                            )
                        else:
                            raise ValueError(
                                f"[FUSION] Unsupported fusion type: {fusion_type}"
                            )

                        # Run predictions using ML head
                        fusion_preds = fusion_head.predict(
                            output.embeddings.cpu().numpy()
                        )

                        # Store predictions inside output.extra
                        output.extra["fusion_preds"] = fusion_preds
                        # logger.info("[FUSION] Fusion predictions stored in output.extra['fusion_preds']")
                    else:
                        logger.warning(
                            f"[FUSION] Skipping — fusion head not found at {fusion_path}"
                        )

                # If fusion head is loaded, run fusion prediction on embeddings
                if fusion_head is not None:
                    if not isinstance(output, ModelOutput):
                        raise TypeError(
                            "Model output must be ModelOutput for fusion inference"
                        )
                    if output.embeddings is None:
                        raise ValueError(
                            "Backbone output missing embeddings for fusion inference"
                        )
                    embeddings_np = output.embeddings.detach().cpu().numpy()
                    fusion_preds = fusion_head.predict(embeddings_np)
                    output.extra["fusion_preds"] = fusion_preds

                # Adjust reconstruction shape if present
                if isinstance(output, ModelOutput):
                    if (
                        hasattr(output, "reconstruction")
                        and output.reconstruction is not None
                    ):
                        output.reconstruction = ensure_4d_tensor(output.reconstruction)
                    artifact_dumper.log_output(output, batch_id=i, targets=inputs)
                    results.append(output)
                elif isinstance(output, dict):
                    if "recon" in output:
                        output["recon"] = ensure_4d_tensor(output["recon"])
                    results.append(ModelOutput(**output))
                elif isinstance(output, torch.Tensor):
                    output = ensure_4d_tensor(output)
                    results.append(ModelOutput(reconstruction=output))
                else:
                    raise TypeError(f"Unknown output type: {type(output)}")

        artifact_path = os.path.join(
            artifact_dumper.base_path,
            f"{config.model.name}",
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_inference.pt",
        )
        artifact_dumper.save(artifact_path)

        logger.info("✅ Inference completed successfully.")
        return {
            "model": model,
            "results": results,
            "config": config,
            "artifacts_path": artifact_path,
        }

    except Exception as e:
        logger = logger or get_global_logger()
        logger.error(f"\n❌ Inference failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise
