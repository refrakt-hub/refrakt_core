"""
Refrakt Core API

This module provides the main programmatic interface for the Refrakt framework.
It exposes clean functions for training, testing, and inference without CLI dependencies.
"""

from refrakt_core.api.train import train
from refrakt_core.api.test import test
from refrakt_core.api.inference import inference

__all__ = ["train", "test", "inference"]
