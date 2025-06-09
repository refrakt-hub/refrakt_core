"""# dataset_builder.py"""
from typing import Any

from omegaconf import DictConfig, OmegaConf

from refrakt_core.api.builders.transform_builder import build_transform
from refrakt_core.registry.dataset_registry import (DATASET_REGISTRY,
                                                    get_dataset)


def build_dataset(cfg: DictConfig) -> Any:
    """Build dataset from configuration"""
    # Convert to native Python types for compatibility
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Extract dataset parameters
    dataset_params = cfg_dict.get("params", {}).copy()
    dataset_name = cfg_dict["name"]
    wrapper_name = cfg_dict.get("wrapper", None)

    # Handle transform separately
    transform_cfg = cfg_dict.get("transform", None)
    transform_fn = build_transform(transform_cfg) if transform_cfg else None

    # Handle wrapped datasets (e.g., contrastive)
    if wrapper_name:
        # Create base dataset without transform
        base_dataset = get_dataset(dataset_name, **dataset_params)

        # Get wrapper class from registry
        if wrapper_name not in DATASET_REGISTRY:
            raise ValueError(f"Wrapper dataset '{wrapper_name}' not found in registry")

        wrapper_cls = DATASET_REGISTRY[wrapper_name]

        # Pass base_dataset to wrapper
        return wrapper_cls(base_dataset, transform=transform_fn)
    else:
        # For non-wrapped datasets, apply transform directly
        if transform_fn:
            dataset_params["transform"] = transform_fn
        return get_dataset(dataset_name, **dataset_params)