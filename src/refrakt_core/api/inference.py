"""The inference code for Refrakt."""

import os
import sys
import traceback
from typing import Any, Dict, Optional, Union

import torch
from omegaconf import OmegaConf

from refrakt_core.api.builders.dataloader_builder import build_dataloader
from refrakt_core.api.builders.dataset_builder import build_dataset
from refrakt_core.api.builders.model_builder import build_model
from refrakt_core.api.core.logger import RefraktLogger
from refrakt_core.api.core.utils import import_modules


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
        # Load configuration
        if isinstance(cfg, str):
            config = OmegaConf.load(cfg)
        else:
            config = cfg

        modules = import_modules()

        # Get device from config, fallback to auto-detection
        if (
            hasattr(config, "trainer")
            and hasattr(config.trainer, "params")
            and hasattr(config.trainer.params, "device")
        ):
            device = config.trainer.params.device
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Using device: {device}")

        # Build model
        model = build_model(config, modules, device)

        # Load model checkpoint
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # Handle both checkpoint formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.eval()

        # Prepare data
        if data is None:
            # Use test dataset if no data provided
            logger.info("No data provided, using test dataset...")
            test_cfg = OmegaConf.merge(
                config.dataset, OmegaConf.create({"params": {"train": False}})
            )
            test_dataset = build_dataset(test_cfg)
            data_loader = build_dataloader(test_dataset, config.dataloader)
        else:
            # Assume data is already a DataLoader or convert it
            data_loader = data

        # Run inference
        logger.info("\nRunning inference...")
        results = []

        # ====== NEW: Collect inputs and outputs for visualization ======
        vis_inputs = []
        vis_targets = []
        vis_outputs = []
        max_visualization = 8
        # ====== END NEW ======

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if isinstance(batch, dict):
                    # Try the most common keys first
                    for key in ["image", "lr", "input"]:
                        if key in batch:
                            inputs = batch[key].to(device)
                            break
                    else:
                        raise KeyError("Expected one of ['image', 'lr', 'input'] in batch dict.")

                    # Optional: handle target if present
                    for tgt_key in ["label", "target", "hr"]:
                        if tgt_key in batch:
                            targets = batch[tgt_key].to(device)
                            break
                    else:
                        targets = None

                elif isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(device)
                    targets = batch[1].to(device) if len(batch) > 1 else None

                else:
                    inputs = batch.to(device)
                    targets = None


                outputs = model(inputs)

                # Store results (convert to CPU for memory efficiency)
                if isinstance(outputs, dict):
                    batch_results = {k: v.cpu() for k, v in outputs.items()}
                else:
                    batch_results = outputs.cpu()

                results.append(batch_results)

                if len(vis_inputs) < max_visualization:
                    vis_inputs.append(inputs.cpu())
                    if isinstance(outputs, dict):
                        vis_outputs.append(next(iter(outputs.values())).cpu())
                    else:
                        vis_outputs.append(outputs.cpu())

                    if targets is not None:
                        vis_targets.append(targets.cpu())  # ✅ Add this line

                if batch_idx % 100 == 0:
                    logger.info(f"Processed batch {batch_idx + 1}/{len(data_loader)}")

        # ====== NEW: Log inference visualization ======
        try:
            if vis_inputs:
                inputs_vis = torch.cat([t.cpu() for t in vis_inputs])[
                    :max_visualization
                ]
                outputs_vis = torch.cat([t.cpu() for t in vis_outputs])[
                    :max_visualization
                ]
                targets_vis = None
                if vis_targets:
                    targets_vis = torch.cat([t.cpu() for t in vis_targets])[
                        :max_visualization
                    ]

                if outputs_vis.ndim == 4:
                    logger.log_inference_results(
                        inputs=inputs_vis,
                        outputs=outputs_vis,
                        targets=targets_vis,
                        step=0,
                    )
                else:
                    logger.info(
                        "Skipping visual logging: output is not a 4D image tensor"
                    )
        except Exception as e:
            logger.error(f"Inference visualization failed: {str(e)}")
        # ====== END NEW ======

        logger.info("\nInference completed successfully!")

        return {"model": model, "results": results, "config": config}

    except Exception as e:
        logger.error(f"\n❌ Training failed: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
