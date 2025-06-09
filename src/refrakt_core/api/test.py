"""The test module for Refrakt."""
import os
import sys
import traceback
from typing import Any, Dict, Optional, Union

import torch
from omegaconf import OmegaConf

# Add direct imports for dataset and dataloader builders
from refrakt_core.api.builders.dataloader_builder import build_dataloader
from refrakt_core.api.builders.dataset_builder import build_dataset
from refrakt_core.api.builders.trainer_builder import initialize_trainer
from refrakt_core.api.core.logger import RefraktLogger
from refrakt_core.api.core.utils import build_model_components, import_modules
from refrakt_core.logging import get_global_logger


def test(
    cfg: Union[str, OmegaConf],
    model_path: Optional[str] = None,
    logger: Optional[RefraktLogger] = None,
) -> Dict[str, Any]:
    """
    Test/evaluate a model based on the provided configuration.

    Args:
        cfg: Either a path to a config file or an OmegaConf object
        model_path: Optional path to a saved model checkpoint

    Returns:
        Dict containing evaluation results
    """

    if logger is None:
        logger = get_global_logger()

    try:
        # Load configuration
        if isinstance(cfg, str):
            config = OmegaConf.load(cfg)
        else:
            config = cfg

        logger.log_config(OmegaConf.to_container(config, resolve=True))

        modules = import_modules()

        # === Build Dataset & DataLoader ===
        logger.info("Building test datasets...")
        test_cfg = OmegaConf.merge(
            config.dataset, OmegaConf.create({"params": {"train": False}})
        )
        test_dataset = build_dataset(test_cfg)
        test_loader = build_dataloader(test_dataset, config.dataloader)
        logger.info(f"Test batches: {len(test_loader)}")

        # Build model components
        components = build_model_components(config)

        # Load model checkpoint if provided
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path, map_location=components.device)
            components.model.load_state_dict(
                checkpoint.get("model_state_dict", checkpoint)
            )

        # Initialize trainer for evaluation
        trainer = initialize_trainer(
            config,
            components.model,
            test_loader,
            test_loader,  # Use test_loader for both
            components.loss_fn,
            components.optimizer,
            components.scheduler,
            components.device,
            modules,save_dir=None,  # No save_dir needed for evaluation
        )

        trainer.logger = logger
        logger.info("\nRunning evaluation...")
        eval_results = trainer.evaluate()

        # Run evaluation
        # ====== NEW: Visualize test results ======
        try:
            # Get sample batch for visualization
            sample_batch = next(iter(test_loader))
            if isinstance(sample_batch, (tuple, list)):
                inputs = sample_batch[0].to(components.device)
                targets = sample_batch[1] if len(sample_batch) > 1 else None
            else:
                inputs = sample_batch.to(components.device)
                targets = None

            # Run model
            with torch.no_grad():
                outputs = components.model(inputs)

            # Log visualization
            logger.log_inference_results(
                inputs=inputs,
                outputs=outputs,
                targets=targets,
                step=0,  # Use step 0 for test
            )
        except Exception as e:
            logger.error(f"Test visualization failed: {str(e)}")
        # ====== END NEW ======

        logger.info("\nEvaluation completed successfully!")

        return {
            "model": components.model,
            "evaluation_results": eval_results,
            "config": config,
        }

    except Exception as e:
        logger.error(f"\n❌ Training failed: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
