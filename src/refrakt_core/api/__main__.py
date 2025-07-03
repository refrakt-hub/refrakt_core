"""
Refrakt CLI entry point for training, testing, and inference.

This module parses command-line arguments and dispatches to the appropriate pipeline stage.
"""

import argparse
import gc
import os
import sys
from typing import cast

import torch


def main() -> None:
    """
    Main entry point for the Refrakt CLI.

    Parses command-line arguments, sets up logging, and dispatches to train, test, or inference.
    """
    try:
        print("==> Refrakt CLI launched")

        parser = argparse.ArgumentParser(description="Refrakt Core Pipeline")
        parser.add_argument(
            "--config", required=True, help="Path to configuration file"
        )
        parser.add_argument("--log_dir", help="Override log directory path")
        parser.add_argument("--debug", action="store_true", help="Enable debug logging")
        args = parser.parse_args()

        # Delay ALL imports until after logger configuration
        from omegaconf import OmegaConf
        from refrakt_core.api.core.logger import RefraktLogger
        from refrakt_core.global_logging import set_global_logger

        cfg = OmegaConf.load(args.config)

        # Extract runtime parameters from YAML config
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(cfg_dict, dict):
            raise TypeError(
                "Config must be a dict after OmegaConf.to_container, got type: {}".format(
                    type(cfg_dict)
                )
            )
        cfg_dict = cast(dict, cfg_dict)
        runtime_cfg = cfg_dict.get("runtime", {})

        mode = runtime_cfg.get("mode", "train")
        log_dir = args.log_dir or runtime_cfg.get("log_dir", "./logs")

        # Handle log_types - accept list or single string
        log_types = runtime_cfg.get("log_type", [])
        if isinstance(log_types, str):
            log_types = [log_types]  # Convert single string to list
        elif log_types is None:
            log_types = []  # Convert None to empty list

        console = runtime_cfg.get("console", True)
        model_path = runtime_cfg.get("model_path", None)
        debug = args.debug or runtime_cfg.get("debug", False)

        model_name = cfg.model.name

        logger = RefraktLogger(
            model_name=model_name,
            log_dir=log_dir,
            log_types=log_types,
            console=console,
            debug=debug,
        )

        logger.info(f"Logging initialized. Log file: {logger.log_file}")
        set_global_logger(logger)

        # Now import pipeline components
        from refrakt_core.api.inference import inference
        from refrakt_core.api.test import test
        from refrakt_core.api.train import train

        try:
            if mode == "train":
                logger.info(f"Starting training with config: {args.config}")
                train(args.config, model_path=model_path, logger=logger)

            elif mode == "test":
                logger.info(f"Starting testing with config: {args.config}")
                test(args.config, model_path=model_path, logger=logger)

            elif mode == "inference":
                if not model_path:
                    raise ValueError(
                        "model_path must be provided in runtime config for inference mode"
                    )
                logger.info(f"Starting inference with config: {args.config}")
                inference(args.config, model_path=model_path, logger=logger)

            elif mode == "pipeline":
                logger.info("🔁 Starting full pipeline (train → test → inference)")
                save_dir = cfg.trainer.params.save_dir
                model_name = cfg.trainer.params.model_name
                model_path = os.path.join(save_dir, f"{model_name}.pth")

                logger.info("🚀 Training phase started")
                train(args.config, logger=logger)

                logger.info("🧪 Testing phase started")
                test(args.config, model_path=model_path, logger=logger)

                logger.info("🔮 Inference phase started")
                inference(args.config, model_path=model_path, logger=logger)

        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
        finally:
            logger.info("Finalizing and saving logs...")
            logger.close()

    finally:
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
