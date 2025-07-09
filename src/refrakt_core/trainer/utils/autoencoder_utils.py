"""
Utility functions for autoencoder trainers.
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch

from refrakt_core.schema.loss_output import LossOutput
from refrakt_core.schema.model_output import ModelOutput


def handle_autoencoder_training_step(
    model: Any,
    inputs: torch.Tensor,
    loss_fn: Any,
    optimizer: Optional[Any],
    global_step: int,
    artifact_dumper: Optional[Any] = None,
) -> Tuple[float, ModelOutput]:
    """
    Handle a single training step for autoencoder.

    Args:
        model: The autoencoder model
        inputs: Input tensor
        loss_fn: Loss function
        optimizer: Optimizer
        global_step: Current global step
        artifact_dumper: Artifact dumper for logging

    Returns:
        Tuple of (loss_value, model_output)
    """
    # Reshape inputs if they're flattened
    if inputs.dim() == 2 and hasattr(model, "expected_input_dim"):
        inputs = inputs.view(-1, *model.expected_input_dim)

    if optimizer is not None and hasattr(optimizer, "zero_grad"):
        optimizer.zero_grad()

    output = model(inputs)

    # Ensure output is ModelOutput
    if not isinstance(output, ModelOutput):
        output = _unwrap_output(output)

    # Compute loss
    loss_output = loss_fn(output, inputs)

    # Backward pass
    loss_output.total.backward()
    if optimizer is not None and hasattr(optimizer, "step"):
        optimizer.step()

    # Logging
    if artifact_dumper:
        artifact_dumper.log_scalar_dict(
            loss_output.summary(), step=global_step, prefix="train"
        )

        if isinstance(output, ModelOutput):
            artifact_dumper.log_scalar_dict(
                output.summary(), step=global_step, prefix="train"
            )

    return loss_output.total.item(), output


def handle_autoencoder_evaluation_step(
    model: Any,
    inputs: torch.Tensor,
    loss_fn: Any,
    global_step: int,
    artifact_dumper: Optional[Any] = None,
) -> float:
    """
    Handle a single evaluation step for autoencoder.

    Args:
        model: The autoencoder model
        inputs: Input tensor
        loss_fn: Loss function
        global_step: Current global step
        artifact_dumper: Artifact dumper for logging

    Returns:
        Loss value
    """
    # Reshape inputs if they're flattened
    if inputs.dim() == 2 and hasattr(model, "expected_input_dim"):
        inputs = inputs.view(-1, *model.expected_input_dim)

    output = model(inputs)
    loss_output: LossOutput = loss_fn(output, inputs)

    # Log validation metrics
    if artifact_dumper:
        artifact_dumper.log_scalar_dict(
            loss_output.summary(), step=global_step, prefix="val"
        )

    return loss_output.total.item()


def _unwrap_output(
    output: Union[ModelOutput, Dict[str, Any], torch.Tensor],
) -> ModelOutput:
    """
    Convert output to ModelOutput if not already.

    Args:
        output: Model output

    Returns:
        ModelOutput: Wrapped model output

    Raises:
        ValueError: If output is None
    """
    if output is None:
        raise ValueError("[_unwrap_output] Received None as output!")

    if isinstance(output, ModelOutput):
        return output
    elif isinstance(output, dict):
        return ModelOutput(**output)
    else:
        return ModelOutput(reconstruction=output)


def extract_autoencoder_inputs(
    batch: Union[torch.Tensor, Dict[str, Any], list[Any], tuple[Any, ...]],
) -> torch.Tensor:
    """
    Extract input tensor from a batch for autoencoder training.

    Args:
        batch: Batch from DataLoader

    Returns:
        Input tensor

    Raises:
        TypeError: If input tensor cannot be extracted
    """
    if isinstance(batch, (list, tuple)):
        if len(batch) == 0 or not isinstance(batch[0], torch.Tensor):
            raise TypeError(
                "Batch is empty or does not contain a tensor as the first element."
            )
        tensor = batch[0]
        assert isinstance(tensor, torch.Tensor)
        return tensor
    if isinstance(batch, dict):
        image = batch.get("image")
        if image is not None and isinstance(image, torch.Tensor):
            assert isinstance(image, torch.Tensor)
            return image
        input_tensor = batch.get("input")
        if input_tensor is not None and isinstance(input_tensor, torch.Tensor):
            assert isinstance(input_tensor, torch.Tensor)
            return input_tensor
        raise TypeError(
            "Batch dict does not contain a valid 'image' or 'input' tensor."
        )
    if isinstance(batch, torch.Tensor):
        assert isinstance(batch, torch.Tensor)
        return batch
    raise TypeError(f"Unsupported batch type: {type(batch)}")
