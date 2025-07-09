"""
Utility functions for DINO trainers.
"""

from typing import Any, List, Optional, Tuple

import torch

from refrakt_core.schema.model_output import ModelOutput


def _forward_pass_dino(
    model: Any, views: List[torch.Tensor], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform forward pass for DINO model."""
    with torch.autocast(device_type=device.type):
        student_out = torch.stack(
            [model(view, teacher=False).embeddings for view in views],
            dim=1,
        )
        teacher_out = model(views[0], teacher=True).embeddings.unsqueeze(1)

        if student_out is None or teacher_out is None:
            raise ValueError("Model outputs are None")

        return student_out, teacher_out


def _backward_pass_dino(
    loss: torch.Tensor, optimizer: Any, scaler: Any, model: Any
) -> None:
    """Perform backward pass for DINO training."""
    if isinstance(optimizer, dict) or optimizer is None:
        raise RuntimeError(
            "DINOTrainer expects a single optimizer, not a dict or None."
        )
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    model.update_teacher()


def _log_dino_metrics(
    model: Any,
    student_out: torch.Tensor,
    loss_output: Any,
    views: List[torch.Tensor],
    global_step: int,
    batch_id: int,
    artifact_dumper: Optional[Any],
    logger: Optional[Any],
    grad_log_interval: int,
    param_log_interval: int,
    optimizer: Any,
) -> None:
    """Log metrics for DINO training."""
    # Artifact logging
    if artifact_dumper and artifact_dumper.should_log_step(global_step):
        model_output = ModelOutput(
            embeddings=student_out,
            attention_maps=(
                getattr(
                    getattr(model.dino_model, "backbone", None),
                    "get_attention_maps",
                    lambda x: None,
                )(views[0])
                if hasattr(model, "dino_model")
                else None
            ),
            loss_components=loss_output.components,
            extra={
                "backbone": getattr(model, "wrapper_config", {}).get(
                    "backbone", "unknown"
                )
            },
        )
        artifact_dumper.log_full_output(
            output=model_output,
            loss=loss_output,
            step=global_step,
            batch_id=batch_id,
            prefix="train",
        )

    # Logger metrics
    if logger:
        if global_step % grad_log_interval == 0:
            logger.log_gradients(model, step=global_step, prefix="")
        if global_step % param_log_interval == 0:
            logger.log_parameters(model, step=global_step, prefix="")
            lr = optimizer.param_groups[0]["lr"]
            logger.log_metrics({"lr": lr}, step=global_step)


def handle_dino_training_step(
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
    Handle a single training step for DINO learning.

    Args:
        model: The model to train
        batch: Input batch (list of views)
        device: Target device
        loss_fn: Loss function
        optimizer: Optimizer
        scaler: Gradient scaler
        global_step: Current global step
        artifact_dumper: Artifact dumper for logging
        batch_id: Current batch ID
        grad_log_interval: Interval for gradient logging
        param_log_interval: Interval for parameter logging
        logger: Logger for metrics

    Returns:
        Tuple of (loss_value, success_flag)
    """
    try:
        views = batch

        # Forward pass
        student_out, teacher_out = _forward_pass_dino(model, views, device)

        # Compute loss
        loss_output = loss_fn(student_out, teacher_out)
        loss = loss_output.total

        if loss is None:
            return 0.0, False

        # Backward pass
        _backward_pass_dino(loss, optimizer, scaler, model)

        # Logging
        _log_dino_metrics(
            model,
            student_out,
            loss_output,
            views,
            global_step,
            batch_id,
            artifact_dumper,
            logger,
            grad_log_interval,
            param_log_interval,
            optimizer,
        )

        return loss.item(), True

    except (RuntimeError, ValueError, TypeError):
        return 0.0, False


def handle_dino_evaluation_step(
    model: Any,
    batch: Any,
    device: torch.device,
    loss_fn: Any,
    global_step: int,
    artifact_dumper: Optional[Any] = None,
    batch_id: int = 0,
) -> Tuple[float, bool]:
    """
    Handle a single evaluation step for DINO learning.

    Args:
        model: The model to evaluate
        batch: Input batch (list of views)
        device: Target device
        loss_fn: Loss function
        global_step: Current global step
        artifact_dumper: Artifact dumper for logging
        batch_id: Current batch ID

    Returns:
        Tuple of (loss_value, success_flag)
    """
    try:
        views = batch

        student_out = torch.stack(
            [model(view, teacher=False).embeddings for view in views],
            dim=1,
        )
        teacher_out = model(views[0], teacher=True).embeddings.unsqueeze(1)

        loss_output = loss_fn(student_out, teacher_out)
        loss = loss_output.total

        # Logging
        if artifact_dumper and artifact_dumper.should_log_step(global_step):
            model_output = ModelOutput(embeddings=student_out)
            artifact_dumper.log_full_output(
                output=model_output,
                loss=loss_output,
                step=global_step,
                batch_id=f"val_{batch_id}",
                prefix="val",
            )

        return loss.item(), True

    except (RuntimeError, ValueError, TypeError):
        return 0.0, False
