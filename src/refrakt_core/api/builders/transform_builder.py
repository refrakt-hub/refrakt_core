"""# transform_builder.py"""
from omegaconf import ListConfig
from torchvision import transforms
from typing import Callable

from refrakt_core.registry.transform_registry import get_transform


def build_transform(cfg) -> Callable:
    """Build transform pipeline from config using transform registry"""
    transform_list = []

     # Resolve the actual transform sequence depending on style
    if isinstance(cfg, (list, ListConfig)):
        transform_sequence = cfg
    elif isinstance(cfg, dict):
        if "views" in cfg:
            transform_sequence = cfg["views"][0]
        elif "components" in cfg:
            transform_sequence = cfg["components"]
        else:
            raise ValueError(f"Unsupported transform configuration format: {cfg}")
    else:
        raise TypeError(f"Invalid transform config type: {type(cfg)}")

    transform_list = []
    for t in transform_sequence:
        name = t["name"]
        params = t.get("params", {})

        # Handle nested transform blocks like RandomApply
        if name.lower() == "randomapply":
            nested_cfgs = params.get("transforms", [])
            nested_transforms = build_transform(nested_cfgs)

            if isinstance(nested_transforms, transforms.Compose):
                nested_transforms = nested_transforms.transforms

            transform = get_transform("RandomApply", nested_transforms, p=params.get("p", 0.5))
        else:
            transform = get_transform(name, **params)

        transform_list.append(transform)

    # Fix for PairedTransform
    if len(transform_list) == 1:
        fn = transform_list[0]
        if hasattr(fn, "__call__") and fn.__call__.__code__.co_argcount >= 2:
            return fn

    return transforms.Compose(transform_list)
