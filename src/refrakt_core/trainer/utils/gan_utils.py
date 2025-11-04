"""
Utility functions for GAN trainers.
"""

from typing import Any, Optional, Tuple

import torch

from refrakt_core.schema.model_output import ModelOutput


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
    visualization_hooks: Optional[list] = None,
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
    batch_count = 0

    for batch_id, batch in enumerate(train_loader):
        batch_count += 1
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
            # No else clause needed for discriminator step

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
            # No else clause needed for generator step

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
                optimizer=optimizer,
                visualization_hooks=visualization_hooks,
            )

        except (RuntimeError, ValueError, TypeError) as e:
            if logger:
                logger.error(f"Batch skipped due to error: {e}")
            else:
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
    try:
        # Use the model wrapper's training_step method for discriminator
        if hasattr(model, "training_step"):
            batch = {"lr": lr, "hr": hr}
            result = model.training_step(
                batch=batch,
                optimizer=optimizer,
                loss_fn=loss_fns,
                device=str(device),
                phase="discriminator",
            )

            if "d_loss" in result:
                d_loss_output = result["d_loss"]

                # Perform actual optimizer step
                if optimizer and "discriminator" in optimizer:
                    optimizer["discriminator"].zero_grad()
                    d_loss_output.total.backward()
                    optimizer["discriminator"].step()

                return d_loss_output, d_loss_output.total.item()
            else:
                return None, 0.0
        else:
            return None, 0.0
    except Exception as e:
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
    try:
        # Use the model wrapper's training_step method for generator
        if hasattr(model, "training_step"):
            batch = {"lr": lr, "hr": hr}
            result = model.training_step(
                batch=batch,
                optimizer=optimizer,
                loss_fn=loss_fns,
                device=str(device),
                phase="generator",
            )

            if "g_loss" in result:
                g_loss_output = result["g_loss"]

                # Perform actual optimizer step
                if optimizer and "generator" in optimizer:
                    optimizer["generator"].zero_grad()
                    g_loss_output.total.backward()
                    optimizer["generator"].step()

                return g_loss_output, g_loss_output.total.item()
            else:
                return None, 0.0
        else:
            return None, 0.0
    except Exception as e:
        return None, 0.0


def _log_gan_loss_metrics(
    g_loss_out: Optional[Any],
    d_loss_out: Optional[Any],
    artifact_dumper: Optional[Any],
    global_step: int,
) -> None:
    """Log GAN loss metrics to artifact dumper."""
    if not artifact_dumper:
        return

    loss_summary = {}

    # Generator loss metrics
    if g_loss_out and hasattr(g_loss_out, "summary"):
        g_summary = g_loss_out.summary()
        if isinstance(g_summary, dict):
            # Prefix generator losses
            for key, value in g_summary.items():
                loss_summary[f"generator/{key}"] = value

    # Discriminator loss metrics
    if d_loss_out and hasattr(d_loss_out, "summary"):
        d_summary = d_loss_out.summary()
        if isinstance(d_summary, dict):
            # Prefix discriminator losses
            for key, value in d_summary.items():
                loss_summary[f"discriminator/{key}"] = value

    # Add combined metrics if both losses exist
    if g_loss_out and d_loss_out:
        if hasattr(g_loss_out, "total") and hasattr(d_loss_out, "total"):
            g_total = (
                g_loss_out.total.item()
                if torch.is_tensor(g_loss_out.total)
                else g_loss_out.total
            )
            d_total = (
                d_loss_out.total.item()
                if torch.is_tensor(d_loss_out.total)
                else d_loss_out.total
            )
            loss_summary["combined/total_loss"] = g_total + d_total
            loss_summary["combined/g_d_ratio"] = g_total / (
                d_total + 1e-8
            )  # Avoid division by zero

    if loss_summary:
        artifact_dumper.log_scalar_dict(loss_summary, step=global_step, prefix="train")


def _log_gan_gradients_and_parameters(
    model: Any,
    logger: Optional[Any],
    global_step: int,
    grad_log_interval: int,
    param_log_interval: int,
    optimizer: Optional[Any] = None,
) -> None:
    """Log gradients and parameters for GAN model."""
    if not logger:
        return

    # Log gradients at specified intervals
    if global_step % grad_log_interval == 0:
        logger.log_gradients(model, step=global_step, prefix="gan")

    # Log parameters at specified intervals
    if global_step % param_log_interval == 0:
        logger.log_parameters(model, step=global_step, prefix="gan")

        # Log learning rates for both optimizers
        if optimizer and isinstance(optimizer, dict):
            lr_metrics = {}
            if "generator" in optimizer and hasattr(
                optimizer["generator"], "param_groups"
            ):
                lr_metrics["generator_lr"] = optimizer["generator"].param_groups[0][
                    "lr"
                ]
            if "discriminator" in optimizer and hasattr(
                optimizer["discriminator"], "param_groups"
            ):
                lr_metrics["discriminator_lr"] = optimizer[
                    "discriminator"
                ].param_groups[0]["lr"]

            if lr_metrics:
                logger.log_metrics(lr_metrics, step=global_step)


def _log_gan_artifacts(
    lr: torch.Tensor,
    hr: torch.Tensor,
    generated_output: Optional[torch.Tensor],
    g_loss_out: Optional[Any],
    d_loss_out: Optional[Any],
    artifact_dumper: Optional[Any],
    global_step: int,
    batch_id: int,
) -> None:
    """Log full GAN artifacts including images and outputs."""
    if not artifact_dumper or not artifact_dumper.should_log_step(global_step):
        return

    # Create ModelOutput for generated images
    if generated_output is not None:
        # For SRGAN, the generated output should be the super-resolved image
        gan_output = ModelOutput(
            image=generated_output,  # Generated high-res image
            targets=hr,  # Ground truth high-res image
            extra={
                "low_res_input": lr,  # Original low-res input
                "model_type": "srgan",
                "global_step": global_step,
            },
        )

        # Combine losses for logging
        combined_loss = None
        if g_loss_out and d_loss_out:
            # Create a combined loss output
            combined_loss = g_loss_out  # Use generator loss as primary
            if hasattr(combined_loss, "components") and hasattr(
                d_loss_out, "components"
            ):
                # Add discriminator loss components
                combined_loss.components.update(
                    {f"discriminator_{k}": v for k, v in d_loss_out.components.items()}
                )
        elif g_loss_out:
            combined_loss = g_loss_out
        elif d_loss_out:
            combined_loss = d_loss_out

        # Log the full output
        artifact_dumper.log_full_output(
            output=gan_output,
            loss=combined_loss,
            step=global_step,
            batch_id=f"gan_batch_{batch_id}",
            prefix="train/gan",
            skip_metrics_logging=True,  # Already logged in _log_gan_loss_metrics
        )


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
    optimizer: Optional[Any] = None,
    visualization_hooks: Optional[list] = None,
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
        optimizer: Optimizer dictionary for learning rate logging
        visualization_hooks: List of visualization components to update
    """
    # 1. Log scalar loss metrics to artifact dumper
    _log_gan_loss_metrics(
        g_loss_out=g_loss_out,
        d_loss_out=d_loss_out,
        artifact_dumper=artifact_dumper,
        global_step=global_step,
    )

    # 2. Log gradients and parameters for the model
    _log_gan_gradients_and_parameters(
        model=model,
        logger=logger,
        global_step=global_step,
        grad_log_interval=grad_log_interval,
        param_log_interval=param_log_interval,
        optimizer=optimizer,
    )

    # 3. Log full artifacts with images (if we have generated output)
    # For SRGAN, we need to get the generated image from the model
    generated_output: Optional[torch.Tensor] = None
    if hasattr(model, "generator") and lr is not None:
        try:
            # Generate super-resolved image for logging
            with torch.no_grad():
                if hasattr(model.generator, "forward"):
                    result = model.generator(lr)
                    if isinstance(result, torch.Tensor):
                        generated_output = result
                elif callable(model.generator):
                    result = model.generator(lr)
                    if isinstance(result, torch.Tensor):
                        generated_output = result
        except Exception:
            # If generation fails, skip artifact logging
            generated_output = None

    _log_gan_artifacts(
        lr=lr,
        hr=hr,
        generated_output=generated_output,
        g_loss_out=g_loss_out,
        d_loss_out=d_loss_out,
        artifact_dumper=artifact_dumper,
        global_step=global_step,
        batch_id=batch_id,
    )

    # 4. Update visualization hooks with GAN-specific data
    if visualization_hooks:
        for viz in visualization_hooks:
            viz_name = getattr(viz, "registry_name", viz.__class__.__name__)
            try:
                if viz_name == "reconstruction_viz":
                    # Update ReconstructionViz with LR inputs and generated HR outputs
                    if generated_output is not None:
                        # Convert to numpy for the viz component
                        lr_np = lr.detach().cpu().numpy()
                        generated_np = generated_output.detach().cpu().numpy()
                        viz.update(inputs=lr_np, recons=generated_np)

                elif viz_name == "sample_generation":
                    # Update SampleGeneration with generated HR images
                    if generated_output is not None:
                        generated_np = generated_output.detach().cpu().numpy()
                        viz.update(samples=generated_np)

                # Note: Other viz components (feature_attribution, etc.) would need
                # custom implementations for SRGAN compatibility

            except Exception as e:
                if logger:
                    logger.warning(f"[VizHook] Failed to update {viz_name}: {e}")
                else:
                    print(f"[VizHook] Failed to update {viz_name}: {e}")
