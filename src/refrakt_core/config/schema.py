"""
Schema definitions for validating training configurations using Pydantic.
Includes definitions for dataset, model, loss, optimizer, scheduler, and trainer configs.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class TransformConfig(BaseModel):
    """Represents a single transformation operation in a dataset pipeline."""
    name: str
    params: Optional[Dict[str, Any]] = None


class DatasetConfig(BaseModel):
    """Configuration for dataset loading and preprocessing."""
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)
    wrapper: Optional[str] = None
    transform: Optional[Union[str, List[TransformConfig]]] = None


class DataLoaderConfig(BaseModel):
    """Configuration for PyTorch DataLoader parameters."""
    params: Dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    """Configuration for model architecture and hyperparameters."""
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class LossConfig(BaseModel):
    """Configuration for loss function and its optional components."""
    name: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    components: Optional[Dict[str, Dict[str, Any]]] = None


class OptimizerConfig(BaseModel):
    """Configuration for optimizer type and hyperparameters."""
    name: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    components: Optional[Dict[str, Dict[str, Any]]] = None


class SchedulerConfig(BaseModel):
    """Configuration for learning rate scheduler."""
    name: Optional[str] = None
    params: Optional[Dict[str, Any]] = None


class TrainerConfig(BaseModel):
    """Configuration for training loop, device, and checkpointing."""
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class RefraktConfig(BaseModel):
    """Top-level config combining all components for training."""
    dataset: DatasetConfig
    dataloader: DataLoaderConfig
    model: ModelConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    scheduler: Optional[SchedulerConfig] = None
    trainer: TrainerConfig
