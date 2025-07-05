"""
Supervised trainer utilities.

This module contains utility functions for the supervised trainer implementation.
"""

import torch
from typing import Any, Dict
from torch.optim import Optimizer

from refrakt_core.schema.model_output import ModelOutput


def handle_training_step(trainer_instance, batch: Any, step: int, epoch: int) -> None:
    """Handle a single training step."""
    inputs, targets = trainer_instance._unpack_batch(batch)
    inputs, targets = inputs.to(trainer_instance.device), targets.to(trainer_instance.device)

    if trainer_instance.optimizer is not None:
        if isinstance(trainer_instance.optimizer, Optimizer):
            trainer_instance.optimizer.zero_grad()
    
    output = trainer_instance.model(inputs)
    loss_output = trainer_instance.loss_fn(output, targets)

    assert isinstance(loss_output.total, torch.Tensor)
    loss_output.total.backward()  # type: ignore[no-untyped-call]

    trainer_instance._current_loss_output = loss_output
    log_training_metrics(trainer_instance, loss_output, output, step)
    log_artifacts(trainer_instance, output, loss_output, step, epoch)

    if trainer_instance.optimizer is not None:
        if isinstance(trainer_instance.optimizer, Optimizer):
            trainer_instance.optimizer.step()


def log_training_metrics(trainer_instance, loss_output: Any, output: Any, step: int) -> None:
    """Log training metrics."""
    logger = trainer_instance._get_logger()
    
    if logger and trainer_instance.global_step % trainer_instance.grad_log_interval == 0:
        logger.log_gradients(trainer_instance.model, step=trainer_instance.global_step, prefix="")
    if logger and trainer_instance.global_step % trainer_instance.param_log_interval == 0:
        logger.log_parameters(trainer_instance.model, step=trainer_instance.global_step, prefix="")
        if trainer_instance.optimizer is not None and isinstance(trainer_instance.optimizer, Optimizer):
            lr = trainer_instance.optimizer.param_groups[0]["lr"]
            logger.log_metrics({"lr": lr}, step=trainer_instance.global_step)

    loss_summary = loss_output.summary()
    if trainer_instance.artifact_dumper:
        trainer_instance.artifact_dumper.log_scalar_dict(
            loss_summary, step=trainer_instance.global_step, prefix="train"
        )

    if isinstance(output, ModelOutput) and hasattr(output, "summary"):
        output_summary = output.summary()
        if trainer_instance.artifact_dumper:
            trainer_instance.artifact_dumper.log_scalar_dict(
                output_summary, step=trainer_instance.global_step, prefix="train/output"
            )


def log_artifacts(trainer_instance, output: Any, loss_output: Any, step: int, epoch: int) -> None:
    """Log artifacts for the current step."""
    if not trainer_instance.artifact_dumper or not trainer_instance.artifact_dumper.should_log_step(trainer_instance.global_step):
        return

    # wrap everything inside a ModelOutput
    full_output = output
    if not isinstance(output, ModelOutput):
        full_output = ModelOutput(logits=output)
    
    if trainer_instance._current_batch is not None:
        inputs, targets = trainer_instance._unpack_batch(trainer_instance._current_batch)
        full_output.targets = targets
        full_output.image = inputs

    trainer_instance.artifact_dumper.log_full_output(
        full_output,
        loss=loss_output,
        step=trainer_instance.global_step,
        batch_id=f"step{trainer_instance.global_step}",
        skip_metrics_logging=True,  # Skip since metrics are already logged above
    )

    if step == 0:
        trainer_instance.artifact_dumper.log_full_output(
            full_output,
            loss=loss_output,
            step=trainer_instance.global_step,
            batch_id=f"epoch{epoch}_step{step}",
            skip_metrics_logging=True,  # Skip since metrics are already logged above
        )


def handle_epoch_end(trainer_instance, epoch: int, best_accuracy: float) -> float:
    """Handle end of epoch operations."""
    if (
        trainer_instance.scheduler
        and not isinstance(trainer_instance.scheduler, dict)
        and hasattr(trainer_instance.scheduler, "step")
    ):
        trainer_instance.scheduler.step()
        if trainer_instance.optimizer is not None and isinstance(trainer_instance.optimizer, Optimizer):
            print(
                f"Epoch {epoch + 1} complete. LR: {trainer_instance.optimizer.param_groups[0]['lr']:.6f}"
            )

    acc = trainer_instance.evaluate()
    if acc > best_accuracy:
        best_accuracy = acc
        trainer_instance.save(suffix="best_model")
        print(f"New best model saved with accuracy: {acc * 100:.2f}%")

    trainer_instance.save(suffix="latest")
    return best_accuracy 