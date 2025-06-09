"""# model_builder.py"""
from typing import Any, Dict

from omegaconf import OmegaConf


def build_model(cfg: OmegaConf, modules: Dict, device: str) -> Any:
    """Build and initialize model"""
    print("Building model...")
    model_params = cfg.model.params or {}
    model = modules["get_model"](cfg.model.name, **model_params).to(device)
    print(f"Model: {cfg.model.name} with params: {model_params}")
    return model
