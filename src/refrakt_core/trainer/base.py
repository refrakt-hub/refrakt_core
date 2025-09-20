"""
Base trainer class for machine learning models.

This module defines the abstract base class for all trainers in Refrakt.
It provides a standard interface for training, evaluation, saving, and loading of models.
All custom trainers should inherit from BaseTrainer and implement the required methods.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from refrakt_core.trainer.utils.base_utils import (
    determine_checkpoint_path,
    load_checkpoint,
    load_optimizer_state,
    load_scheduler_state,
)


class BaseTrainer(ABC):
    """
    Abstract base class for all trainers.

    Handles device setup, saving/loading checkpoints, and exposes an interface
    for training and evaluation to be implemented by subclasses.

    Attributes:
        device (torch.device): The device on which the model runs.
        model (Module): The model to be trained.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        save_dir (str): Directory to save checkpoints.
        model_name (str): Name of the model for checkpointing.
        artifact_dumper (Any): Optional artifact logger/dumper.
        optimizer (Optional[Union[Optimizer, Dict[str, Optimizer]]]): Optimizer(s) for training.
        scheduler (Optional[Union[Any, Dict[str, Any]]]): Scheduler(s) for learning rate.
        global_step (int): Global training step counter.
    """

    def __init__(
        self,
        model: Module,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        device: str = "cuda",
        **kwargs: Any,
    ) -> None:
        """
        Initialize the base trainer.

        Args:
            model (Module): The model to be trained.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            device (str, optional): Device to use (default: "cuda").
            **kwargs: Additional keyword arguments (e.g., save_dir, model_name, artifact_dumper).
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir: str = kwargs.pop("save_dir", "checkpoints/")
        self.model_name: str = kwargs.pop("model_name", "model")
        self.artifact_dumper = kwargs.pop("artifact_dumper", None)
        self.optimizer: Optional[Union[Optimizer, Dict[str, Optimizer]]] = None
        self.scheduler: Optional[Union[Any, Dict[str, Any]]] = None
        self.global_step: int = 0

    @abstractmethod
    def train(self, num_epochs: int) -> Dict[str, float]:
        """
        Train the model for the specified number of epochs.

        Args:
            num_epochs (int): Number of epochs to train.
        """
        pass

    @abstractmethod
    def evaluate(self) -> Any:
        """
        Evaluate the model on validation or test data.

        Returns:
            Any: Evaluation metric(s) or results, as defined by the subclass.
        """
        pass

    def get_checkpoint_path(self, suffix: str = "final") -> str:
        """
        Get the checkpoint file path.

        Args:
            suffix (str): Suffix to append to the model name.

        Returns:
            str: Complete checkpoint file path.
        """
        if suffix == "best_model":
            return os.path.join(self.save_dir, f"{self.model_name}.pth")
        return os.path.join(self.save_dir, f"{self.model_name}_{suffix}.pth")

    def save(self, path: Optional[str] = None, suffix: str = "final") -> None:
        """
        Save model, optimizer, and scheduler state to disk.

        Args:
            path (Optional[str]): Custom file path to save the checkpoint. If None, uses default path.
            suffix (str): Suffix to generate default checkpoint name if path is None.
        """
        if path is None:
            path = self.get_checkpoint_path(suffix)
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        checkpoint: Dict[str, Any] = {
            "model_state_dict": self.model.state_dict(),
            "model_name": self.model_name,
            "global_step": self.global_step,
        }

        if self.optimizer is not None:
            if isinstance(self.optimizer, dict):
                checkpoint["optimizer_state_dict"] = {
                    k: opt.state_dict() for k, opt in self.optimizer.items()
                }
            else:
                checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()

        if self.scheduler is not None:
            if isinstance(self.scheduler, dict):
                checkpoint["scheduler_state_dict"] = {
                    k: sch.state_dict() for k, sch in self.scheduler.items()
                }
            else:
                checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        try:
            torch.save(checkpoint, path)
            print(f"[INFO] Model saved to: {path}")
        except (OSError, RuntimeError) as e:
            print(f"[ERROR] Failed to save model: {e}")

    def load(self, path: Optional[str] = None, suffix: str = "final") -> None:
        """
        Load model, optimizer, and scheduler state from disk.
        Tries to load base model first (without suffix), falls back to suffixed version if needed.
        Special handling for 'best_model' suffix remains unchanged.

        Args:
            path (Optional[str]): Custom file path to load the checkpoint. If None, uses default path.
            suffix (str): Suffix to fall back to if base model isn't found.

        Raises:
            OSError: If loading fails due to file issues.
            RuntimeError: If loading fails due to state issues.
            KeyError: If expected keys are missing in the checkpoint.
        """
        try:
            checkpoint_path = determine_checkpoint_path(
                path=path,
                suffix=suffix,
                save_dir=self.save_dir,
                model_name=self.model_name,
            )

            checkpoint = load_checkpoint(checkpoint_path, self.device.type)

            self.model.load_state_dict(checkpoint["model_state_dict"])

            if self.optimizer is not None and "optimizer_state_dict" in checkpoint:
                load_optimizer_state(self.optimizer, checkpoint["optimizer_state_dict"])

            if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
                load_scheduler_state(self.scheduler, checkpoint["scheduler_state_dict"])

            self.global_step = checkpoint.get("global_step", 0)

            print(f"[INFO] Successfully loaded from: {checkpoint_path}")

        except (OSError, RuntimeError, KeyError) as e:
            print(f"[ERROR] Failed to load model: {e}")
            raise
