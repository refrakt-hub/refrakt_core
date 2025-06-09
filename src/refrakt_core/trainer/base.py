"""
Base trainer class for machine learning models.

Defines a standard interface for training, evaluation, saving, and loading of models.
Subclasses must implement `train` and `evaluate` methods.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from omegaconf.listconfig import ListConfig

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class BaseTrainer(ABC):
    """
    Abstract base class for all trainers.

    Handles device setup, saving/loading checkpoints, and exposes an interface
    for training and evaluation to be implemented by subclasses.
    """

    def __init__(
        self,
        model: Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
        **kwargs: Any,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir: str = kwargs.pop("save_dir", "checkpoints/")
        self.model_name: str = kwargs.pop("model_name", "model")
        self.optimizer: Optional[Union[Optimizer, Dict[str, Optimizer]]] = None
        self.scheduler: Optional[Union[Any, Dict[str, Any]]] = None

    @abstractmethod
    def train(self, num_epochs: int) -> None:
        """
        Train the model for the specified number of epochs.
        """
        pass

    @abstractmethod
    def evaluate(self) -> Any:
        """
        Evaluate the model on validation or test data.
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
            path (Optional[str]): Custom file path to save the checkpoint.
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

        Args:
            path (Optional[str]): Custom file path to load the checkpoint.
            suffix (str): Suffix to generate default checkpoint name if path is None.
        """
        if path is None:
            path = self.get_checkpoint_path(suffix)

        try:
            torch.serialization.add_safe_globals([ListConfig])
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])

            if self.optimizer is not None and "optimizer_state_dict" in checkpoint:
                optimizer_state = checkpoint["optimizer_state_dict"]
                if isinstance(self.optimizer, dict):
                    for k, opt in self.optimizer.items():
                        if k in optimizer_state:
                            opt.load_state_dict(optimizer_state[k])
                else:
                    self.optimizer.load_state_dict(optimizer_state)

            if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
                scheduler_state = checkpoint["scheduler_state_dict"]
                if isinstance(self.scheduler, dict):
                    for k, sch in self.scheduler.items():
                        if k in scheduler_state:
                            sch.load_state_dict(scheduler_state[k])
                else:
                    self.scheduler.load_state_dict(scheduler_state)

            print(f"[INFO] Model loaded from: {path}")
        except (OSError, RuntimeError, KeyError) as e:
            print(f"[ERROR] Failed to load model: {e}")
