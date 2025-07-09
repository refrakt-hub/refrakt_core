"""
Refrakt API Builders Module

This module provides a collection of builder functions for constructing various
components of the Refrakt framework including models, datasets, optimizers,
schedulers, trainers, and transforms. These builders serve as factory functions
that create and configure components based on configuration parameters.

The module includes builders for:
- Model construction and configuration
- Loss function setup and parameterization
- Optimizer creation with learning rate and weight decay settings
- Learning rate scheduler configuration
- Trainer initialization with training parameters
- DataLoader creation with batch size and worker settings
- Dataset loading and preprocessing
- Transform pipeline construction

All builders follow a consistent interface pattern and integrate with the Refrakt
registry system for component discovery and instantiation.
"""

from refrakt_core.api.builders.dataloader_builder import build_dataloader
from refrakt_core.api.builders.dataset_builder import build_dataset
from refrakt_core.api.builders.loss_builder import build_loss
from refrakt_core.api.builders.model_builder import build_model
from refrakt_core.api.builders.optimizer_builder import build_optimizer
from refrakt_core.api.builders.scheduler_builder import build_scheduler
from refrakt_core.api.builders.trainer_builder import initialize_trainer
from refrakt_core.api.builders.transform_builder import build_transform

__all__ = [
    "build_model",
    "build_loss",
    "build_optimizer",
    "build_scheduler",
    "initialize_trainer",
    "build_dataloader",
    "build_dataset",
    "build_transform",
]
