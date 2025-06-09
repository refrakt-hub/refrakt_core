"""# transform_builder.py"""
from omegaconf import ListConfig
from torchvision import transforms

from refrakt_core.registry.transform_registry import get_transform


def build_transform(cfg) -> transforms.Compose:
    """Build transform pipeline from config using transform registry"""
    transform_list = []

    # Handle different transform configuration styles
    if isinstance(cfg, (list, ListConfig)):
        transform_sequence = cfg
    elif isinstance(cfg, dict) and "views" in cfg:
        transform_sequence = cfg["views"][0]
    else:
        raise ValueError(f"Unsupported transform configuration: {type(cfg)}")

    for t in transform_sequence:
        name = t["name"]
        params = t.get("params", {})

        # Special handling for transforms with nested transforms
        if name == "RandomApply":
            # Recursively build nested transforms
            nested_transforms_cfg = params.get("transforms", [])
            nested_transforms = build_transform(nested_transforms_cfg).transforms
            transform_list.append(
                get_transform(
                    "RandomApply", transforms=nested_transforms, p=params.get("p", 0.5)
                )
            )
        else:
            # Handle all other transforms through the registry
            transform = get_transform(name, **params)
            transform_list.append(transform)

    return transforms.Compose(transform_list)
