"""
Artifact utilities.

This module contains utility functions for artifact dumping operations.
"""

from typing import Any, Dict, List, Optional, Union

import torch

from refrakt_core.schema.loss_output import LossOutput
from refrakt_core.schema.model_output import ModelOutput


def extract_output_fields(output: ModelOutput) -> Dict[str, Any]:
    """Extract fields from ModelOutput for logging."""
    record = {}
    fields = [
        "logits",
        "embeddings",
        "image",
        "reconstruction",
        "targets",
        "attention_maps",
        "loss_components",
        "extra",
    ]

    for field in fields:
        value = getattr(output, field, None)
        if value is not None:
            if field == "loss_components" and isinstance(value, dict):
                record[field] = {
                    k: v.detach().cpu() if torch.is_tensor(v) else v
                    for k, v in value.items()
                }
            elif torch.is_tensor(value):
                record[field] = value.detach().cpu()  # type: ignore[assignment]
            else:
                record[field] = value

    return record


def process_loss_output(loss: LossOutput) -> Dict[str, Any]:
    """Process LossOutput for logging."""
    record = {}
    # LossOutput.total is always a torch.Tensor
    if hasattr(loss.total, "item"):
        record["loss_total"] = float(loss.total.item())
    else:
        record["loss_total"] = float(loss.total)
    record["loss_components_full"] = {  # type: ignore[assignment]
        k: float(v.item()) if torch.is_tensor(v) else float(v)
        for k, v in loss.components.items()
    }
    return record


def create_batch_record(
    output: ModelOutput,
    loss: Optional[LossOutput] = None,
    filenames: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Create a batch record for logging."""
    record = extract_output_fields(output)

    if filenames is not None:
        record["filenames"] = filenames

    if loss is not None:
        loss_record = process_loss_output(loss)
        record.update(loss_record)

    return record


def should_log_batch(batch_id: Optional[Union[int, str]], log_every: int) -> bool:
    """Check if batch should be logged based on frequency."""
    if isinstance(batch_id, int) and batch_id % log_every != 0:
        return False
    return True
