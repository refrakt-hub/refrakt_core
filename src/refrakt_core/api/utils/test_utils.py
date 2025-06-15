import os 
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
    if model_path is None:
        logger.warning("No model checkpoint provided — using random init weights")
        return
    if not os.path.exists(model_path):
        logger.error(f"Model path does not exist: {model_path}")
        raise FileNotFoundError(model_path)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    logger.info(f"Loaded model from {model_path}")