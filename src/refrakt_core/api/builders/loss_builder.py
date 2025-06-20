from typing import Any, Dict, Union
from omegaconf import OmegaConf

from torch import nn 
from refrakt_core.registry.loss_registry import get_loss
from refrakt_core.wrappers.schema.default_loss import DefaultLossWrapper
def build_loss(
    cfg: OmegaConf, modules: Dict[str, Any], device: str
) -> Union[nn.Module, Dict[str, nn.Module]]:
    """
    Build and wrap loss functions to return LossOutput objects.
    Uses DefaultLossWrapper to ensure standardized output format.
    """

    def create_wrapped_loss(name: str, params: Dict[str, Any]) -> nn.Module:
        raw_loss = modules["get_loss"](name, **params)

        if isinstance(raw_loss, nn.Module) and hasattr(raw_loss, "forward"):
            return raw_loss.to(device)

        print(f"[DEBUG] Loss is not a full nn.Module, wrapping with DefaultLossWrapper: {type(raw_loss)}")
        return DefaultLossWrapper(raw_loss).to(device)

    loss_cfg = cfg.loss

    # GAN-style: generator/discriminator
    if loss_cfg.get("generator") or loss_cfg.get("discriminator"):
        loss_fn = {}
        for comp_name in ["generator", "discriminator"]:
            comp_cfg = loss_cfg.get(comp_name)
            if comp_cfg:
                loss_name = comp_cfg["name"]
                loss_params = comp_cfg.get("params", {})
                loss_fn[comp_name] = create_wrapped_loss(loss_name, loss_params)
                print(f"[INFO] Loss ({comp_name}): {loss_name} with params: {loss_params}")
        return loss_fn

    # Multi-component losses
    elif loss_cfg.get("components"):
        loss_fn = {}
        for comp_name, comp_cfg in loss_cfg.components.items():
            loss_name = comp_cfg["name"]
            loss_params = comp_cfg.get("params", {})
            loss_fn[comp_name] = create_wrapped_loss(loss_name, loss_params)
            print(f"[INFO] Loss ({comp_name}): {loss_name} with params: {loss_params}")
        return loss_fn

    # Single loss
    else:
        loss_name = loss_cfg.name
        loss_params = loss_cfg.get("params", {})
        wrapped = create_wrapped_loss(loss_name, loss_params)
        print(f"[INFO] Loss: {loss_name} with params: {loss_params}")
        return wrapped
