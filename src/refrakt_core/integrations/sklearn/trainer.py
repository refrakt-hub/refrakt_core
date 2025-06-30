from typing import Any, Dict, Optional, Union
from torch.nn import Module
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm

from refrakt_core.schema.model_output import ModelOutput
from refrakt_core.registry.trainer_registry import register_trainer
from refrakt_core.trainer.base import BaseTrainer


@register_trainer("fusion")
class FusionTrainer(BaseTrainer):
    def __init__(
        self,
        model: Module,  # Torch backbone (returns ModelOutput)
        fusion_head: Any,  # e.g. SklearnWrapper
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
        artifact_dumper: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, 
                         train_loader, 
                         val_loader, device, 
                         artifact_dumper=artifact_dumper, 
                         **kwargs)

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
        acc = (preds == y_val).mean()

        print(f"\n[RESULT] Validation Accuracy: {acc * 100:.2f}%")

        if self.artifact_dumper:
            self.artifact_dumper.log_scalar_dict({"fusion_accuracy": acc}, step=self.global_step, prefix="val")

        return acc

    def _extract_features_and_labels(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        features, labels = [], []
        first = True
        with torch.no_grad():
            loop = tqdm(loader, desc="Extracting Features", leave=False)

            for batch in loop:
                x, y = self._unpack_batch(batch)
                if first:
                    print(f"[DEBUG] y type: {type(y)}, y shape: {getattr(y, 'shape', None)}")
                    first = False
                x = x.to(self.device)
                output = self.model(x)

                if not isinstance(output, ModelOutput) or output.embeddings is None:
                    raise ValueError("Backbone must return `ModelOutput` with `embeddings` for fusion mode.")

                emb = output.embeddings
                features.append(emb.detach().cpu().numpy())
                y_cpu = y.detach().cpu()
                if y_cpu.ndim > 1:
                    y_cpu = y_cpu.view(-1)
                labels.append(y_cpu.numpy())

        return np.concatenate(features), np.concatenate(labels)

    def _unpack_batch(self, batch: Union[tuple, list, Dict[str, torch.Tensor]]) -> tuple:
        # Handle SimCLR-style batches: (img1, img2, label)
        if isinstance(batch, (tuple, list)):
            if len(batch) == 3:
                return batch[0], batch[2]  # img1, label
            return batch[0], batch[1]
        if isinstance(batch, dict):
            return batch["input"], batch["target"]
        raise TypeError("Unsupported batch format")
