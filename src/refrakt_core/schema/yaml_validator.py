"""
Schema definitions for validating training configurations using Pydantic.
Includes definitions for dataset, model, loss, optimizer, scheduler, and trainer configs.
Supports pure-ML, pure-DL, and fusion pipelines.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class RuntimeConfig(BaseModel):
    """Runtime configuration for pipeline execution."""

    mode: str = Field(..., description="Pipeline mode (pipeline, inference, etc.)")
    log_type: List[str] = Field(default=[], description="Logging types")


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


class FusionConfig(BaseModel):
    """Configuration for fusion models (ML models applied to DL embeddings)."""

    type: str = Field(..., description="Fusion type (cuml, sklearn, etc.)")
    model: str = Field(..., description="Fusion model name")
    params: Dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    """Configuration for model architecture and hyperparameters."""

    name: str
    wrapper: Optional[str] = None
    type: Optional[str] = None  # "ml" or "dl"
    backend: Optional[str] = None  # "sklearn", "cuml", etc.
    params: Dict[str, Any] = Field(default_factory=dict)
    fusion: Optional[FusionConfig] = None


class LossConfig(BaseModel):
    """Configuration for loss function and its optional components."""

    name: Optional[str] = None
    mode: Optional[str] = None  # "embedding", "logits", etc.
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


class FeatureEngineeringConfig(BaseModel):
    """Configuration for feature engineering steps in ML pipelines."""

    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class TrainerConfig(BaseModel):
    """Configuration for training loop, device, and checkpointing."""

    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class RefraktConfig(BaseModel):
    """Top-level config combining all components for training.

    Supports three pipeline types:
    1. Pure-ML: Uses feature_engineering, ML models, no loss/optimizer/scheduler
    2. Pure-DL: Uses dataset/dataloader, DL models, loss/optimizer/scheduler
    3. Fusion: Combines DL models with ML fusion models
    """

    runtime: RuntimeConfig = Field(..., description="Runtime configuration")
    dataset: Optional[DatasetConfig] = None
    dataloader: Optional[DataLoaderConfig] = None
    model: ModelConfig
    loss: Optional[LossConfig] = None
    optimizer: Optional[OptimizerConfig] = None
    scheduler: Optional[SchedulerConfig] = None
    trainer: TrainerConfig
    feature_engineering: Optional[List[FeatureEngineeringConfig]] = None

    class Config:
        extra = "forbid"  # Strict validation - no extra fields allowed
