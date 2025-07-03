import gc
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from omegaconf import OmegaConf
from refrakt_core.api.inference import inference
from refrakt_core.api.test import test
from refrakt_core.api.train import train

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))
gc.collect()
torch.cuda.empty_cache()


def main(config_path: str, mode: str = "train"):
    """
    Main function for CLI usage

    Args:
        config_path: Path to configuration file
        mode: One of 'train', 'test', 'inference'
    """
    if mode == "train":
        return train(config_path)
    elif mode == "test":
        return test(config_path)
    elif mode == "inference":
        raise ValueError(
            "Inference mode requires model_path parameter. Use inference() function directly."
        )
    else:
        raise ValueError(
            f"Invalid mode: {mode}. Must be one of 'train', 'test', 'inference'"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
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
        test(args.config, args.model_path)
    elif args.mode == "inference":
        inference(args.config, args.model_path)
