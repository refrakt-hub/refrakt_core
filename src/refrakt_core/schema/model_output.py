from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class ModelOutput:
    embeddings: Optional[Any] = None  # contrastive / latent features
    logits: Optional[Any] = None  # supervised output
    image: Optional[Any] = None  # GAN or output image
    reconstruction: Optional[Any] = None  # AE / VAE
    targets: Optional[Any] = None  # target values/labels
    attention_maps: Optional[Any] = None  # ViT, DINO
    loss_components: Dict[str, Any] = field(
        default_factory=dict
    )  # for contrastive/self-sup
    extra: Dict[str, Any] = field(default_factory=dict)

    def _add_tensor_stats(
        self, summary: Dict[str, float], tensor: Optional[Any], prefix: str
    ) -> None:
        """
        Helper method to add tensor statistics to summary.
        """
        if tensor is not None and isinstance(tensor, torch.Tensor):
            summary[f"{prefix}/mean"] = tensor.mean().item()
            if prefix != "reconstruction":  # Skip std for reconstruction
                summary[f"{prefix}/std"] = tensor.std().item()

    def _add_embeddings_stats(self, summary: Dict[str, float]) -> None:
        """
        Helper method to add embeddings statistics to summary.
        """
        if self.embeddings is not None and isinstance(self.embeddings, torch.Tensor):
            summary["embeddings/norm_mean"] = self.embeddings.norm(dim=1).mean().item()
            summary["embeddings/std"] = self.embeddings.std().item()

    def _add_loss_components(self, summary: Dict[str, float]) -> None:
        """
        Helper method to add loss components to summary.
        """
        for k, v in self.loss_components.items():
            if isinstance(v, torch.Tensor):
                summary[f"loss_component/{k}"] = v.item()

    def _add_extra_components(self, summary: Dict[str, float]) -> None:
        """
        Helper method to add extra components to summary.
        """
        for k, v in self.extra.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                summary[f"extra/{k}"] = v.item()

    def summary(self) -> Dict[str, float]:
        summary: Dict[str, float] = {}

        self._add_tensor_stats(summary, self.logits, "logits")
        self._add_embeddings_stats(summary)
        self._add_tensor_stats(summary, self.reconstruction, "reconstruction")
        self._add_tensor_stats(summary, self.attention_maps, "attention")
        self._add_loss_components(summary)
        self._add_extra_components(summary)

        return summary

    @property
    def shape(self) -> Optional[torch.Size]:
        """
        Get the shape of the primary output tensor.
        Priority order: logits > embeddings > image > reconstruction
        """
        if self.logits is not None and isinstance(self.logits, torch.Tensor):
            return self.logits.shape
        elif self.embeddings is not None and isinstance(self.embeddings, torch.Tensor):
            return self.embeddings.shape
        elif self.image is not None and isinstance(self.image, torch.Tensor):
            return self.image.shape
        elif self.reconstruction is not None and isinstance(
            self.reconstruction, torch.Tensor
        ):
            return self.reconstruction.shape
        else:
            return None

    @property
    def device(self) -> Optional[torch.device]:
        """
        Get the device of the primary output tensor.
        Priority order: logits > embeddings > image > reconstruction
        """
        if self.logits is not None and isinstance(self.logits, torch.Tensor):
            return self.logits.device
        elif self.embeddings is not None and isinstance(self.embeddings, torch.Tensor):
            return self.embeddings.device
        elif self.image is not None and isinstance(self.image, torch.Tensor):
            return self.image.device
        elif self.reconstruction is not None and isinstance(
            self.reconstruction, torch.Tensor
        ):
            return self.reconstruction.device
        else:
            return None

    def __len__(self) -> int:
        """
        Get the batch size (first dimension) of the primary output tensor.
        Priority order: logits > embeddings > image > reconstruction
        """
        if self.logits is not None and isinstance(self.logits, torch.Tensor):
            return self.logits.shape[0]
        elif self.embeddings is not None and isinstance(self.embeddings, torch.Tensor):
            return self.embeddings.shape[0]
        elif self.image is not None and isinstance(self.image, torch.Tensor):
            return self.image.shape[0]
        elif self.reconstruction is not None and isinstance(
            self.reconstruction, torch.Tensor
        ):
            return self.reconstruction.shape[0]
        else:
            return 0

    def __getitem__(self, key):
        """
        Support indexing like a tensor.
        Returns the primary tensor with the given index.
        """
        primary_tensor = self._get_primary_tensor()
        if primary_tensor is not None:
            return primary_tensor[key]
        else:
            raise IndexError("No primary tensor available for indexing")

    def gather(self, dim: int, index: torch.Tensor, *args, **kwargs):
        """
        Support torch.gather() operation on the primary tensor.
        """
        primary_tensor = self._get_primary_tensor()
        if primary_tensor is not None:
            return primary_tensor.gather(dim, index, *args, **kwargs)
        else:
            raise RuntimeError("No primary tensor available for gather operation")

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        Support torch operations on ModelOutput objects.
        This allows torch.gather() and other torch functions to work with ModelOutput.
        """
        if kwargs is None:
            kwargs = {}

        # Handle torch.gather specifically
        if func == torch.gather:
            # Extract the ModelOutput from args
            model_output = None
            other_args = []
            for arg in args:
                if isinstance(arg, ModelOutput):
                    model_output = arg
                else:
                    other_args.append(arg)

            if model_output is not None:
                primary_tensor = model_output._get_primary_tensor()
                if primary_tensor is not None:
                    # Call torch.gather on the primary tensor
                    return torch.gather(primary_tensor, *other_args, **kwargs)
                else:
                    raise RuntimeError(
                        "No primary tensor available for gather operation"
                    )

        # For other torch functions, try to delegate to the primary tensor
        # Find the first ModelOutput in args
        model_output = None
        for arg in args:
            if isinstance(arg, ModelOutput):
                model_output = arg
                break

        if model_output is not None:
            primary_tensor = model_output._get_primary_tensor()
            if primary_tensor is not None:
                # Replace ModelOutput with primary tensor in args
                new_args = []
                for arg in args:
                    if isinstance(arg, ModelOutput):
                        new_args.append(primary_tensor)
                    else:
                        new_args.append(arg)

                return func(*new_args, **kwargs)
            else:
                raise RuntimeError(
                    f"No primary tensor available for {func.__name__} operation"
                )

        # If no ModelOutput found, let PyTorch handle it normally
        return func(*args, **kwargs)

    def detach(self):
        """
        Support detach() operation on the primary tensor.
        """
        primary_tensor = self._get_primary_tensor()
        if primary_tensor is not None:
            return primary_tensor.detach()
        else:
            raise RuntimeError("No primary tensor available for detach operation")

    def cpu(self):
        """
        Support cpu() operation on the primary tensor.
        """
        primary_tensor = self._get_primary_tensor()
        if primary_tensor is not None:
            return primary_tensor.cpu()
        else:
            raise RuntimeError("No primary tensor available for cpu operation")

    def _get_primary_tensor(self) -> Optional[torch.Tensor]:
        """
        Get the primary tensor for operations.
        Priority order: logits > embeddings > image > reconstruction
        """
        if self.logits is not None and isinstance(self.logits, torch.Tensor):
            return self.logits
        elif self.embeddings is not None and isinstance(self.embeddings, torch.Tensor):
            return self.embeddings
        elif self.image is not None and isinstance(self.image, torch.Tensor):
            return self.image
        elif self.reconstruction is not None and isinstance(
            self.reconstruction, torch.Tensor
        ):
            return self.reconstruction
        else:
            return None

    def to(self, device: Any) -> "ModelOutput":
        def move(x: Any) -> Any:
            if isinstance(x, torch.Tensor):
                return x.to(device)
            elif isinstance(x, dict):
                return {k: move(v) for k, v in x.items()}
            elif isinstance(x, list):
                return [move(v) for v in x]
            elif isinstance(x, tuple):
                return tuple(move(v) for v in x)
            else:
                return x

        return ModelOutput(
            embeddings=move(self.embeddings),
            logits=move(self.logits),
            image=move(self.image),
            reconstruction=move(self.reconstruction),
            targets=move(self.targets),
            attention_maps=move(self.attention_maps),
            loss_components=move(self.loss_components),
            extra=move(self.extra),
        )
