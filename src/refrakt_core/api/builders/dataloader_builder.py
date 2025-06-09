"""# dataloader_builder.py"""
from typing import Any

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader


def build_dataloader(dataset: Any, cfg: DictConfig) -> DataLoader:
    """Build dataloader from dataset and configuration"""
    # Convert to native Python types
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Extract parameters
    params = cfg_dict.get("params", cfg_dict)

    return DataLoader(
        dataset,
        batch_size=params["batch_size"],
        shuffle=params.get("shuffle", False),
        num_workers=params.get("num_workers", 0),
        drop_last=params.get("drop_last", False),
    )
