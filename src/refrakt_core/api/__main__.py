"""
Refrakt CLI entry point for training, testing, and inference.

This module parses command-line arguments and dispatches to the appropriate pipeline stage.
"""

import gc
from typing import cast

import torch
from omegaconf import DictConfig, OmegaConf

from refrakt_core.api.utils.pipeline_utils import (
    setup_logger_and_config,
)
from refrakt_core.api.helpers.cli_helpers import (
    _setup_argument_parser,
    _extract_overrides,
    _apply_config_overrides,
    _extract_runtime_config,
    _setup_logging_config,
    _execute_pipeline_mode,
)


def main() -> None:
    """
    Main entry point for the Refrakt CLI.

    Parses command-line arguments, sets up logging, and dispatches to train, test, or inference.
    """
    try:
        print("==> Refrakt CLI launched")

        parser = _setup_argument_parser()
        args, remaining = parser.parse_known_args()

        # Extract overrides from remaining arguments
        all_overrides = _extract_overrides(args, remaining)

        # Load and apply configuration
        cfg = OmegaConf.load(args.config)
        cfg = _apply_config_overrides(cfg, all_overrides)

        # Extract runtime parameters from YAML config
        runtime_cfg = _extract_runtime_config(cfg)
        mode, log_dir, log_types, console, model_path, debug = _setup_logging_config(runtime_cfg, args.log_dir)

        # Override debug flag if provided in args
        debug = args.debug or debug

        model_name = cfg.model.name

        # Setup logger
        logger = setup_logger_and_config(
            cfg, model_name, log_dir, log_types, console, debug, all_overrides
        )

        try:
            _execute_pipeline_mode(mode, cfg, model_path, logger)

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
