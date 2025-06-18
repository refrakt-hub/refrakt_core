import os
import glob
import torch
from omegaconf import OmegaConf
from refrakt_core.api.builders.dataloader_builder import build_dataloader
from refrakt_core.api.builders.dataset_builder import build_dataset
from refrakt_core.utils.methods import extract_visual_tensor

def _load_config(cfg):
    return OmegaConf.load(cfg) if isinstance(cfg, str) else cfg

def _build_test_loader(config):
    test_cfg = OmegaConf.merge(config.dataset, OmegaConf.create({"params": {"train": False}}))
    dataset = build_dataset(test_cfg)
    return build_dataloader(dataset, config.dataloader)

def _load_model_checkpoint(model, model_path, device, logger):
    import typing
    from collections import defaultdict
    from omegaconf.nodes import AnyNode
    from omegaconf import ListConfig, DictConfig
    from omegaconf.base import ContainerMetadata, Metadata
    from torch.serialization import add_safe_globals

    # 🔐 Allow OmegaConf configs to be unpickled safely
    add_safe_globals([ListConfig, \
            DictConfig, ContainerMetadata, \
            typing.Any, list, dict, defaultdict, \
            int, float, AnyNode, Metadata])
    
    if model_path is None:
        logger.warning("No model checkpoint provided — using random init weights")
        return 0

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        logger.info(f"Loaded model from {model_path}")
        return checkpoint.get("global_step", 0)

    # If file doesn't exist, try fallback logic
    base_dir = os.path.dirname(model_path)
    base_name = os.path.splitext(os.path.basename(model_path))[0]  # autoencoder_simple

    # Try exact match first
    exact_match = os.path.join(base_dir, f"{base_name}.pth")
    if os.path.exists(exact_match):
        checkpoint = torch.load(exact_match, map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        logger.warning(f"⚠️ Falling back to exact checkpoint: {exact_match}")
        return checkpoint.get("global_step", 0)

    # Try matching variants like _latest, _final
    pattern = os.path.join(base_dir, f"{base_name}_*.pth")
    candidates = glob.glob(pattern)
    if candidates:
        # Prefer latest or final first
        preferred = [c for c in candidates if "latest" in c or "final" in c]
        fallback_path = max(preferred or candidates, key=os.path.getmtime)

        checkpoint = torch.load(fallback_path, map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        logger.warning(f"⚠️ Falling back to available checkpoint: {fallback_path}")
        return checkpoint.get("global_step", 0)

    logger.error(f"Model path does not exist: {model_path}")
    raise FileNotFoundError(model_path)
