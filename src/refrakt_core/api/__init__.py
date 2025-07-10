"""
Refrakt API Module

This module provides the main entry points for the Refrakt framework, \
    including training, testing, and inference pipelines. It serves as the \
    primary interface for users to interassct with the Refrakt system.

The module includes:
- Main CLI entry point for different pipeline modes
- Configuration management and validation
- Integration with PyTorch and CUDA memory management
- Error handling and logging setup
"""

import gc
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from . import inference, test, train

__all__ = ["main", "train", "test", "inference"]

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))
gc.collect()
torch.cuda.empty_cache()


def main(config_path: str, mode: str = "train") -> Optional[Dict[str, Any]]:
    """
    Main function for CLI usage of the Refrakt framework.

    This function serves as the primary entry point for running different pipeline modes
    including training, testing, and inference. It handles configuration loading,
    mode validation, and dispatches to the appropriate pipeline function.

    Args:
        config_path: Path to the configuration YAML file containing all pipeline
            parameters
        mode: Pipeline mode to execute. Must be one of 'train', 'test', or 'inference'

    Returns:
        Dictionary containing training results and metrics if mode is 'train',
        None for 'test' and 'inference' modes

    Raises:
        ValueError: If an invalid mode is provided or if inference mode is used
            without proper model_path parameter
    """
    if mode == "train":
        return train.train(config_path)
    elif mode == "test":
        test.test(config_path)
        return None
    elif mode == "inference":
        raise ValueError(
            "Inference mode requires model_path parameter. Use inference() function "
            "directly."
        )
    else:
        raise ValueError(
            f"Invalid mode: {mode}. Must be one of 'train', 'test', 'inference'"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Refrakt CLI - Training, Testing, and Inference Framework"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test", "inference"],
        help="Mode to run: train, test, or inference",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model checkpoint (for test/inference)",
    )
    args = parser.parse_args()

    if args.mode == "inference" and not args.model_path:
        raise ValueError("--model-path is required for inference mode")

    if args.mode == "train":
        main(args.config, "train")
    elif args.mode == "test":
        test.test(args.config, args.model_path)
    elif args.mode == "inference":
        inference.inference(args.config, args.model_path)
