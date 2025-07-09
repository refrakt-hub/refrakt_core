"""
Utility functions for GAN trainers.
"""

from typing import Any, Optional, Tuple

import torch


def handle_gan_epoch_training(
    model: Any,
    train_loader: Any,
    optimizer: Any,
    loss_fns: Any,
    device: torch.device,
    scaler: Any,
    global_step: int,
    artifact_dumper: Optional[Any] = None,
    grad_log_interval: int = 100,
    param_log_interval: int = 500,
    logger: Optional[Any] = None,
) -> Tuple[float, float]:
    """
    Handle training for a single GAN epoch.

    Args:
        model: The GAN model
        train_loader: Training data loader
        optimizer: Optimizer dictionary
        loss_fns: Loss functions dictionary
        device: Target device
        scaler: Gradient scaler dictionary
        global_step: Current global step
        artifact_dumper: Artifact dumper for logging
        grad_log_interval: Interval for gradient logging
        param_log_interval: Interval for parameter logging
        logger: Logger for metrics

    Returns:
        Tuple of (total_g_loss, total_d_loss)
    """
    model.train()
    total_g_loss = 0.0
    total_d_loss = 0.0

    for batch_id, batch in enumerate(train_loader):
        try:
            device_batch = _move_batch_to_device(batch, device)
            lr, hr = extract_gan_batch_data(device_batch)

            d_loss_out = None
            g_loss_out = None
            d_loss_value = 0.0
            g_loss_value = 0.0

            # Discriminator step
            if (
                optimizer is not None
                and isinstance(optimizer, dict)
                and "discriminator" in optimizer
            ):
                d_loss_out, d_loss_value = handle_gan_discriminator_step(
                    model=model,
                    lr=lr,
                    hr=hr,
                    optimizer=optimizer,
                    loss_fns=loss_fns,
                    device=device,
                    scaler=scaler,
                )

            # Generator step
            if (
                optimizer is not None
                and isinstance(optimizer, dict)
                and "generator" in optimizer
            ):
                g_loss_out, g_loss_value = handle_gan_generator_step(
                    model=model,
                    lr=lr,
                    hr=hr,
                    optimizer=optimizer,
                    loss_fns=loss_fns,
                    device=device,
                    scaler=scaler,
                )

                total_g_loss += g_loss_value
                total_d_loss += d_loss_value

            # Logging
            handle_gan_logging(
                g_loss_out=g_loss_out,
                d_loss_out=d_loss_out,
                model=model,
                lr=lr,
                hr=hr,
                global_step=global_step,
                batch_id=batch_id,
                artifact_dumper=artifact_dumper,
                logger=logger,
                grad_log_interval=grad_log_interval,
                param_log_interval=param_log_interval,
            )

        except (RuntimeError, ValueError, TypeError) as e:
            print(f"[ERROR] Batch skipped due to error: {e}")

    return total_g_loss, total_d_loss


def handle_gan_scheduler_step(
    scheduler: Optional[Any],
) -> None:
    """
    Handle scheduler step for GAN training.

    Args:
        scheduler: Scheduler or scheduler dictionary
    """
    if scheduler:
        if isinstance(scheduler, dict):
            if "generator" in scheduler:
                scheduler["generator"].step()
            if "discriminator" in scheduler:
                scheduler["discriminator"].step()
        else:
            scheduler.step()


def _move_batch_to_device(
    batch: Any,
    device: torch.device,
) -> Any:
    """
    Move batch tensors to the specified device.

    Args:
        batch: Batch to move to device
        device: Target device

    Returns:
        Batch moved to device
    """
    if isinstance(batch, dict):
        return {k: v.to(device) for k, v in batch.items()}
    return [x.to(device) for x in batch]


def extract_gan_batch_data(
    device_batch: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract low-resolution and high-resolution data from GAN batch.

    Args:
        device_batch: Batch data on device

    Returns:
        Tuple of (lr, hr) tensors
    """
    if isinstance(device_batch, dict):
        lr = device_batch["lr"]
        hr = device_batch["hr"]
    elif isinstance(device_batch, (list, tuple)) and len(device_batch) >= 2:
        lr = device_batch[0]
        hr = device_batch[1]
    else:
        raise ValueError(
            "Batch must be a dict with 'lr' and 'hr' keys or a list/tuple with at least 2 elements"
        )
    return lr, hr


def handle_gan_discriminator_step(
    model: Any,
    lr: torch.Tensor,
    hr: torch.Tensor,
    optimizer: Any,
    loss_fns: Any,
    device: torch.device,
    scaler: Any,
) -> Tuple[Optional[Any], float]:
    """
    Handle a single discriminator training step.

    Args:
        model: The GAN model
        lr: Low-resolution input
        hr: High-resolution target
        optimizer: Optimizer dictionary
        loss_fns: Loss functions dictionary
        device: Target device
        scaler: Gradient scaler dictionary

    Returns:
        Tuple of (loss_output, loss_value)
    """
    # Discriminator step implementation
    # This would contain the actual discriminator training logic
    return None, 0.0


def handle_gan_generator_step(
    model: Any,
    lr: torch.Tensor,
    hr: torch.Tensor,
    optimizer: Any,
    loss_fns: Any,
    device: torch.device,
    scaler: Any,
) -> Tuple[Optional[Any], float]:
    """
    Handle a single generator training step.

    Args:
        model: The GAN model
        lr: Low-resolution input
        hr: High-resolution target
        optimizer: Optimizer dictionary
        loss_fns: Loss functions dictionary
        device: Target device
        scaler: Gradient scaler dictionary

    Returns:
        Tuple of (loss_output, loss_value)
    """
    # Generator step implementation
    # This would contain the actual generator training logic
    return None, 0.0


def handle_gan_logging(
    g_loss_out: Optional[Any],
    d_loss_out: Optional[Any],
    model: Any,
    lr: torch.Tensor,
    hr: torch.Tensor,
    global_step: int,
    batch_id: int,
    artifact_dumper: Optional[Any] = None,
    logger: Optional[Any] = None,
    grad_log_interval: int = 100,
    param_log_interval: int = 500,
) -> None:
    """
    Handle logging for GAN training.

    Args:
        g_loss_out: Generator loss output
        d_loss_out: Discriminator loss output
        model: The GAN model
        lr: Low-resolution input
        hr: High-resolution target
        global_step: Current global step
        batch_id: Current batch ID
        artifact_dumper: Artifact dumper for logging
        logger: Logger for metrics
        grad_log_interval: Interval for gradient logging
        param_log_interval: Interval for parameter logging
    """
    # Logging implementation
    # This would contain the actual logging logic
    pass
