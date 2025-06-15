# refrakt_core/builders/loss_builder.py

from typing import Any, Dict, Union
from omegaconf import OmegaConf

from refrakt_core.wrappers.losses.loss_wrapper import LossWrapper
from refrakt_core.registry.loss_registry import get_loss, get_loss_mode


def build_loss(
    cfg: OmegaConf, modules: Dict[str, Any], device: str
) -> Union[LossWrapper, Dict[str, LossWrapper]]:
    """
    Build and wrap loss functions. Supports:
    - Single loss
    - GAN-style (generator/discriminator)
    - Multi-component losses (DINO, MAE, MSN, etc.)
    Returns LossWrapper or dict of LossWrappers.
    """
    print("Building loss function...")

    def wrap(name: str, params: Dict[str, Any]) -> LossWrapper:
        raw_loss = modules["get_loss"](name, **params).to(device)
        mode = get_loss_mode(name)
        return LossWrapper(raw_loss)

    loss_cfg = cfg.loss

    if loss_cfg.get("generator") or loss_cfg.get("discriminator"):
        loss_fn = {}
        for comp_name in ["generator", "discriminator"]:
            comp_cfg = loss_cfg.get(comp_name)
            if comp_cfg:
                loss_name = comp_cfg["name"]
                loss_params = comp_cfg.get("params", {})
                loss_fn[comp_name] = wrap(loss_name, loss_params)
                print(f"Loss ({comp_name}): {loss_name} with params: {loss_params}")
        return loss_fn

    elif loss_cfg.get("components"):
        loss_fn = {}
        for comp_name, comp_cfg in loss_cfg.components.items():
            loss_name = comp_cfg["name"]
            loss_params = comp_cfg.get("params", {})
            loss_fn[comp_name] = wrap(loss_name, loss_params)
            print(f"Loss ({comp_name}): {loss_name} with params: {loss_params}")
        return loss_fn

    else:
        loss_name = loss_cfg.name
        loss_params = loss_cfg.get("params", {})
        wrapped = wrap(loss_name, loss_params)
        print(f"Loss: {loss_name} with params: {loss_params}")
        return wrapped
