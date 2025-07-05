import os
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F

from refrakt_core.schema.loss_output import LossOutput
from refrakt_core.schema.model_output import ModelOutput


class ArtifactDumper:
    """
    Logs and saves model outputs and losses for visualization, analysis, or explainability.
    """

    def __init__(
        self,
        enabled: bool = True,
        model_name: Optional[str] = None,
        base_path: str = "./artifacts",
        auto_flush: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        logger: Optional[Any] = None,  # <--- new
        log_every: int = 1,
    ):
        self.enabled = enabled
        self.model_name = model_name
        self.base_path = base_path
        self.auto_flush = auto_flush
        self.logger = logger
        self.log_every = log_every  # ✅ store it

        self.buffer: Dict[str, Dict[str, Any]] = {}
        self.metadata: Dict[str, Any] = metadata or {}
        os.makedirs(self.base_path, exist_ok=True)

    def log_output(
        self,
        output: ModelOutput,
        batch_id: Union[int, str],
        targets: Optional[torch.Tensor] = None,
        filenames: Optional[List[str]] = None,
    ):
        if not self.enabled:
            return

        # Skip if batch_id doesn't meet log frequency
        if isinstance(batch_id, int) and batch_id % self.log_every != 0:
            return

        record = {}
        for field in [
            "logits",
            "embeddings",
            "image",
            "reconstruction",
            "targets",
            "attention_maps",
            "loss_components",
            "extra",
        ]:
            value = getattr(output, field, None)
            if value is not None:
                if field == "loss_components" and isinstance(value, dict):
                    record[field] = {
                        k: v.detach().cpu() if torch.is_tensor(v) else v
                        for k, v in value.items()
                    }
                elif torch.is_tensor(value):
                    record[field] = value.detach().cpu()
                else:
                    record[field] = value

        if targets is not None and "targets" not in record:
            record["targets"] = targets.detach().cpu()

        if filenames is not None:
            record["filenames"] = filenames

        self.buffer[str(batch_id)] = record

        if self.auto_flush:
            self.save(filename=f"batch_{batch_id}_{self.model_name}.pt")

    def log_full_output(
        self,
        output: ModelOutput,
        loss: Optional[LossOutput] = None,
        step: Optional[int] = None,
        batch_id: Optional[Union[int, str]] = None,
        prefix: str = "train",
        filenames: Optional[List[str]] = None,
        skip_metrics_logging: bool = False,
    ):
        if not self.enabled:
            return

        # Respect log_every
        if isinstance(batch_id, int) and batch_id % self.log_every != 0:
            return

        batch_key = str(batch_id) if batch_id is not None else f"step_{step}"
        record = {}

        # === Log from ModelOutput ===
        for field in [
            "logits",
            "embeddings",
            "image",
            "reconstruction",
            "targets",
            "attention_maps",
            "loss_components",
            "extra",
        ]:
            value = getattr(output, field, None)
            if value is not None:
                if field == "loss_components" and isinstance(value, dict):
                    record[field] = {
                        k: v.detach().cpu() if torch.is_tensor(v) else v
                        for k, v in value.items()
                    }
                elif torch.is_tensor(value):
                    record[field] = value.detach().cpu()
                else:
                    record[field] = value

        if filenames is not None:
            record["filenames"] = filenames

        # === Log from LossOutput ===
        if loss is not None:
            # LossOutput.total is always a torch.Tensor
            record["loss_total"] = float(loss.total.item())
            record["loss_components_full"] = {
                k: float(v.item()) for k, v in loss.components.items()
            }

        self.buffer[batch_key] = record

        # === Push scalar metrics to logger (W&B, TensorBoard) ===
        if self.logger and step is not None and not skip_metrics_logging:
            scalar_dict = {}
            if hasattr(output, "summary") and callable(output.summary):
                scalar_dict.update(output.summary())
            if loss and hasattr(loss, "summary"):
                scalar_dict.update(loss.summary())
            if scalar_dict:
                self.logger.log_metrics(scalar_dict, step=step, prefix=prefix)

        # === Save to disk if auto_flush ===
        if self.auto_flush:
            self.save(filename=f"batch_{batch_key}_{self.model_name}.pt")

    def should_log_step(self, step: int) -> bool:
        if not hasattr(self, "_logged_steps"):
            self._logged_steps = set()
        if step in self._logged_steps:
            return False
        self._logged_steps.add(step)
        return True

    def log_loss(
        self,
        loss: Union[LossOutput, Dict[str, torch.Tensor]],
        step: Union[int, str],
        prefix: Optional[str] = None,
    ):
        if not self.enabled:
            return

        key = f"{prefix}_{step}" if prefix else str(step)
        record = self.buffer.get(key, {})

        if isinstance(loss, LossOutput):
            # Handle both tensor and float types for loss.total
            if hasattr(loss.total, 'item'):
                record["loss_total"] = float(loss.total.item())
            else:
                record["loss_total"] = float(loss.total)
            record["loss_components"] = {
                k: float(v.item()) if hasattr(v, 'item') else float(v)
                for k, v in loss.components.items()
            }
        elif isinstance(loss, dict):
            record["loss_dict"] = {k: float(v.item()) for k, v in loss.items()}

        self.buffer[key] = record

    def log_scalar_dict(
        self, scalar_dict: Dict[str, float], step: int, prefix: str = ""
    ):
        if not self.enabled:
            return

        if self.logger:
            # Don't apply prefix here - let log_metrics handle it
            self.logger.log_metrics(
                scalar_dict, step=step, prefix=prefix if prefix else None
            )

    def save(self, filename: Optional[str] = None):
        if not self.enabled:
            return

        from pathlib import Path

        if filename is None:
            filename = f"artifacts_{self.model_name or 'model'}.pt"

        filename_path = Path(filename)

        if filename_path.is_absolute():
            full_path = filename_path
        elif any(part == "artifacts" for part in filename_path.parts):
            # Strip redundant 'artifacts' prefix
            while filename_path.parts[0] == "artifacts":
                filename_path = filename_path.relative_to("artifacts")
            full_path = Path(self.base_path) / filename_path
        else:
            full_path = Path(self.base_path) / filename_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        save_data = {
            "metadata": self.metadata,
            "outputs": self.buffer,
        }
        torch.save(save_data, str(full_path))

    def reset(self):
        self.buffer = {}

    def get_batch(self, batch_id: Union[int, str]) -> Optional[Dict[str, Any]]:
        return self.buffer.get(str(batch_id))

    def summary(self) -> Dict[str, Any]:
        num_batches = len(self.buffer)
        num_preds = sum(
            batch.get("logits", torch.empty(0)).shape[0]
            for batch in self.buffer.values()
        )
        return {
            "total_batches": num_batches,
            "total_predictions": num_preds,
            "metadata_keys": list(self.metadata.keys()),
        }
