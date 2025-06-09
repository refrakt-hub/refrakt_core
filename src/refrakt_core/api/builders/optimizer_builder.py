"""# optimizer_builder.py"""
from typing import Any, Dict, Union

import torch
from omegaconf import OmegaConf, DictConfig


def build_optimizer(cfg: DictConfig, model: Any) -> Union[Any, Dict[str, Any]]:
    """Build optimizer(s)"""
    print("Building optimizer...")
    opt_map = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "adamw": torch.optim.AdamW,
        "rmsprop": torch.optim.RMSprop,
    }

    if cfg.optimizer.get("generator") or cfg.optimizer.get("discriminator"):
        # Handle GAN-style optimizer without explicit 'components' key
        optimizer = {}
        for comp_name in ["generator", "discriminator"]:
            comp_cfg = cfg.optimizer.get(comp_name)
            if comp_cfg:
                opt_name = comp_cfg["name"]
                opt_cls = opt_map.get(opt_name.lower())
                if not opt_cls:
                    raise ValueError(
                        f"Unsupported optimizer for {comp_name}: {opt_name}"
                    )

                opt_params = OmegaConf.to_container(comp_cfg.get("params", {}), resolve=True)

                # Get parameters for specific component
                if comp_name == "generator":
                    parameters = model.generator.parameters()
                elif comp_name == "discriminator":
                    parameters = model.discriminator.parameters()
                else:
                    raise ValueError(f"Unknown optimizer component: {comp_name}")

                optimizer[comp_name] = opt_cls(parameters, **opt_params)
                print(f"Optimizer ({comp_name}): {opt_name} with params: {opt_params}")

    elif cfg.optimizer.get("components"):
        # Handle multi-component optimizer (GAN)
        optimizer = {}
        for comp_name, comp_cfg in cfg.optimizer.components.items():
            opt_name = comp_cfg["name"]
            opt_cls = opt_map.get(opt_name.lower())
            if not opt_cls:
                raise ValueError(f"Unsupported optimizer for {comp_name}: {opt_name}")

            opt_params = OmegaConf.to_container(comp_cfg.get("params", {}), resolve=True)

            # Get parameters for specific component
            if comp_name == "generator":
                parameters = model.generator.parameters()
            elif comp_name == "discriminator":
                parameters = model.discriminator.parameters()
            else:
                raise ValueError(f"Unknown optimizer component: {comp_name}")

            optimizer[comp_name] = opt_cls(parameters, **opt_params)
            print(f"Optimizer ({comp_name}): {opt_name} with params: {opt_params}")
    else:
        # Standard single optimizer (VAE, AE, etc.)
        opt_cls = opt_map.get(cfg.optimizer.name.lower())
        if not opt_cls:
            raise ValueError(f"Unsupported optimizer: {cfg.optimizer.name}")

        optimizer_params = OmegaConf.to_container(cfg.optimizer.params or {}, resolve=True)
        optimizer = opt_cls(model.parameters(), **optimizer_params)
        print(f"Optimizer: {cfg.optimizer.name} with params: {optimizer_params}")

    return optimizer
