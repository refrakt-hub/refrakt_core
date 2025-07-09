"""
Utility functions for contrastive learning trainers.
"""

from typing import Any, Optional, Tuple

import torch

from refrakt_core.schema.loss_output import LossOutput
from refrakt_core.schema.model_output import ModelOutput


def _unpack_and_move_views_to_device(batch: Any, device: Any) -> Any:
    view1, view2 = batch
    return view1.to(device), view2.to(device)


def _contrastive_forward_and_loss(
    model: Any, view1: Any, view2: Any, device: Any, loss_fn: Any
) -> Any:
    with torch.autocast(device_type=device.type):
        out1 = model(view1)
        out2 = model(view2)
        if out1 is None or out2 is None:
            return None, None, None, False
        z1 = out1.embeddings if isinstance(out1, ModelOutput) else out1
        z2 = out2.embeddings if isinstance(out2, ModelOutput) else out2
        if not isinstance(z1, torch.Tensor) or not isinstance(z2, torch.Tensor):
            return None, None, None, False
        loss_output = loss_fn(z1, z2)
        if isinstance(loss_output, torch.Tensor):
            loss_output = LossOutput(total=loss_output)
        loss = loss_output.total
        if loss is None:
            return None, None, None, False
        return loss, out1, out2, loss_output


def _contrastive_artifact_logging(
    out1: Any,
    out2: Any,
    loss_output: Any,
    global_step: int,
    batch_id: Any,
    artifact_dumper: Any,
    prefix1: str,
    prefix2: str,
) -> None:
    if artifact_dumper and artifact_dumper.should_log_step(global_step):
        if not isinstance(out1, ModelOutput):
            out1 = ModelOutput(embeddings=out1)
        if not isinstance(out2, ModelOutput):
            out2 = ModelOutput(embeddings=out2)
        artifact_dumper.log_full_output(
            output=out1,
            loss=loss_output,
            step=global_step,
            batch_id=batch_id,
            prefix=prefix1,
        )
        artifact_dumper.log_full_output(
            output=out2,
            loss=loss_output,
            step=global_step,
            batch_id=batch_id,
            prefix=prefix2,
        )


def _contrastive_logger_metrics(
    model: Any,
    optimizer: Any,
    logger: Any,
    global_step: int,
    grad_log_interval: int,
    param_log_interval: int,
) -> None:
    if logger and optimizer is not None:
        if global_step % grad_log_interval == 0:
            logger.log_gradients(model, step=global_step, prefix="")
        if global_step % param_log_interval == 0:
            logger.log_parameters(model, step=global_step, prefix="")
            if hasattr(optimizer, "param_groups") and len(optimizer.param_groups) > 0:
                lr = optimizer.param_groups[0]["lr"]
                logger.log_metrics({"lr": lr}, step=global_step)


def handle_contrastive_training_step(
    model: Any,
    batch: Any,
    device: torch.device,
    loss_fn: Any,
    optimizer: Any,
    scaler: Any,
    global_step: int,
    artifact_dumper: Optional[Any] = None,
    batch_id: int = 0,
    grad_log_interval: int = 100,
    param_log_interval: int = 500,
    logger: Optional[Any] = None,
) -> Tuple[float, bool]:
    """
    Handle a single training step for contrastive learning.
    """
    try:
        view1, view2 = _unpack_and_move_views_to_device(batch, device)
        loss, out1, out2, loss_output = _contrastive_forward_and_loss(
            model, view1, view2, device, loss_fn
        )
        if loss is None:
            return 0.0, False
        if isinstance(optimizer, dict) or optimizer is None:
            raise RuntimeError(
                "ContrastiveTrainer expects a single optimizer, not a dict or None."
            )
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        _contrastive_artifact_logging(
            out1,
            out2,
            loss_output,
            global_step,
            batch_id,
            artifact_dumper,
            "train/view1",
            "train/view2",
        )
        _contrastive_logger_metrics(
            model, optimizer, logger, global_step, grad_log_interval, param_log_interval
        )
        return loss.item(), True
    except (RuntimeError, ValueError, TypeError):
        return 0.0, False


def handle_contrastive_evaluation_step(
    model: Any,
    batch: Any,
    device: torch.device,
    loss_fn: Any,
    global_step: int,
    artifact_dumper: Optional[Any] = None,
    batch_id: int = 0,
) -> Tuple[float, bool]:
    """
    Handle a single evaluation step for contrastive learning.
    """
    try:
        view1, view2 = _unpack_and_move_views_to_device(batch, device)
        loss, out1, out2, loss_output = _contrastive_forward_and_loss(
            model, view1, view2, device, loss_fn
        )
        if loss is None:
            return 0.0, False
        _contrastive_artifact_logging(
            out1,
            out2,
            loss_output,
            global_step,
            f"val_{batch_id}",
            artifact_dumper,
            "val/view1",
            "val/view2",
        )
        return loss.item(), True
    except (RuntimeError, ValueError, TypeError):
        return 0.0, False
