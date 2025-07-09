"""
Utility functions for MSN trainers.
"""

from typing import Any, Dict, Optional

import torch

from refrakt_core.schema.loss_output import LossOutput


def handle_msn_training_step(
    model: Any,
    inputs: Dict[str, torch.Tensor],
    loss_fn: Any,
    optimizer: Optional[Any],
    global_step: int,
    artifact_dumper: Optional[Any] = None,
    log_every: Optional[int] = None,
    epoch: int = 0,
    step: int = 0,
) -> float:
    """
    Handle a single training step for MSN.

    Args:
        model: The MSN model
        inputs: Input dictionary with 'anchor' and 'target' keys
        loss_fn: Loss function
        optimizer: Optimizer
        global_step: Current global step
        artifact_dumper: Artifact dumper for logging
        log_every: Logging interval
        epoch: Current epoch
        step: Current step

    Returns:
        Loss value
    """
    if optimizer is not None and hasattr(optimizer, "zero_grad"):
        optimizer.zero_grad()

    output = model(inputs)
    loss_output: LossOutput = loss_fn(output)

    # Backward pass
    total_loss: torch.Tensor = loss_output.total
    total_loss.backward()  # type: ignore[no-untyped-call]
    if optimizer is not None and hasattr(optimizer, "step"):
        optimizer.step()

    # Logging
    if artifact_dumper and log_every and step % log_every == 0:
        artifact_dumper.log_full_output(
            output,
            loss=loss_output,
            step=global_step,
            batch_id=f"train_ep{epoch}_step{step}",
        )

    return loss_output.total.item()


def handle_msn_evaluation_step(
    model: Any,
    inputs: Dict[str, torch.Tensor],
    loss_fn: Any,
    global_step: int,
    artifact_dumper: Optional[Any] = None,
    log_every: Optional[int] = None,
    count: int = 0,
) -> float:
    """
    Handle a single evaluation step for MSN.

    Args:
        model: The MSN model
        inputs: Input dictionary with 'anchor' and 'target' keys
        loss_fn: Loss function
        global_step: Current global step
        artifact_dumper: Artifact dumper for logging
        log_every: Logging interval
        count: Current count for logging

    Returns:
        Loss value
    """
    output = model(inputs)
    loss_output: LossOutput = loss_fn(output)

    # Logging
    if artifact_dumper and log_every and count % log_every == 0:
        artifact_dumper.log_full_output(
            output,
            loss=loss_output,
            step=global_step,
            batch_id=f"val_step{count}",
            prefix="val",
        )

    return loss_output.total.item()


def prepare_msn_inputs(batch: Any) -> Dict[str, torch.Tensor]:
    """
    Convert batch to dictionary format expected by MSNWrapper.

    Args:
        batch: Batch from DataLoader

    Returns:
        Dictionary with 'anchor' and 'target' keys

    Raises:
        TypeError: If the batch format is unsupported
    """
    if isinstance(batch, dict) and "anchor" in batch and "target" in batch:
        return batch
    elif isinstance(batch, (tuple, list)) and len(batch) == 2:
        return {"anchor": batch[0], "target": batch[1]}
    elif isinstance(batch, torch.Tensor):
        return {"anchor": batch, "target": batch}
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")
