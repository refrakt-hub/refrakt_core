"""# loss_builder.py"""
from typing import Any, Dict, Union

from omegaconf import OmegaConf


def build_loss(
    cfg: OmegaConf, modules: Dict, device: str
) -> Union[Any, Dict[str, Any]]:
    """Build loss function(s)"""
    print("Building loss function...")

    if cfg.loss.get("generator") or cfg.loss.get("discriminator"):
        # Handle GAN-style loss without explicit 'components' key
        loss_fn = {}
        for comp_name in ["generator", "discriminator"]:
            comp_cfg = cfg.loss.get(comp_name)
            if comp_cfg:
                loss_name = comp_cfg["name"]
                loss_params = comp_cfg.get("params", {})
                loss_fn[comp_name] = modules["get_loss"](loss_name, **loss_params).to(
                    device
                )
                print(f"Loss ({comp_name}): {loss_name} with params: {loss_params}")
    elif cfg.loss.get("components"):
        # Handle multi-component loss with explicit 'components' key
        loss_fn = {}
        for comp_name, comp_cfg in cfg.loss.components.items():
            loss_name = comp_cfg["name"]
            loss_params = comp_cfg.get("params", {})
            loss_fn[comp_name] = modules["get_loss"](loss_name, **loss_params).to(
                device
            )
            print(f"Loss ({comp_name}): {loss_name} with params: {loss_params}")
    else:
        # Standard single loss
        loss_name = cfg.loss.name
        loss_params = cfg.loss.get("params", {})
        loss_fn = modules["get_loss"](loss_name, **loss_params).to(device)
        print(f"Loss: {loss_name} with params: {loss_params}")

    return loss_fn
