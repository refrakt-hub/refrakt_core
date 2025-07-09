from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from refrakt_core.integrations.fusion.helpers import (
    process_labels,
    unpack_batch,
    validate_model_output,
)
from refrakt_core.registry.trainer_registry import register_trainer
from refrakt_core.trainer.base import BaseTrainer


@register_trainer("fusion")
class FusionTrainer(BaseTrainer):
    def __init__(
        self,
        model: Module,  # Torch backbone (returns ModelOutput)
        fusion_head: Any,  # e.g. SklearnWrapper
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        device: str = "cuda",
        artifact_dumper: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model,
            train_loader,
            val_loader,
            device,
            artifact_dumper=artifact_dumper,
            **kwargs,
        )

        self.fusion_head = fusion_head
        self.extra_params = kwargs
        self.global_step = 0

    def train(self, num_epochs: int = 1) -> Dict[str, Any]:
        self.model.eval()
        X_train, y_train = self._extract_features_and_labels(self.train_loader)

        print(f"[INFO] Training fusion head on extracted features: {X_train.shape}")
        self.fusion_head.fit(X_train, y_train)

        acc = self.evaluate()
        return {"fusion_accuracy": acc}

    def evaluate(self) -> float:
        self.model.eval()
        X_val, y_val = self._extract_features_and_labels(self.val_loader)

        preds = self.fusion_head.predict(X_val)
        acc = float((preds == y_val).mean())

        print(f"\n[RESULT] Validation Accuracy: {acc * 100:.2f}%")

        if self.artifact_dumper:
            self.artifact_dumper.log_scalar_dict(
                {"fusion_accuracy": acc}, step=self.global_step, prefix="val"
            )

        return acc

    def _extract_features_and_labels(
        self, loader: DataLoader[Any]
    ) -> tuple[np.ndarray[Any, np.dtype[Any]], np.ndarray[Any, np.dtype[Any]]]:
        features, labels = [], []
        first = True
        with torch.no_grad():
            loop = tqdm(loader, desc="Extracting Features", leave=False)

            for batch in loop:
                x, y = unpack_batch(batch)
                if first:
                    first = False
                x = x.to(self.device)
                output = self.model(x)

                validate_model_output(output)

                emb = output.embeddings
                features.append(emb.detach().cpu().numpy())
                labels.append(process_labels(y))

        return np.concatenate(features), np.concatenate(labels)

    def _unpack_batch(
        self, batch: Union[Tuple[Any, ...], List[Any], Dict[str, Any]]
    ) -> Tuple[Any, ...]:
        return unpack_batch(batch)
