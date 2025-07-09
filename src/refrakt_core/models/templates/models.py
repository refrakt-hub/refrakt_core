"""
Templates for foundational model types in Refrakt: classifiers, autoencoders, contrastive learners, and GANs.
"""

from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import torch

from refrakt_core.models.templates.base import BaseModel


class BaseClassifier(BaseModel):
    """
    Base class for classification models.

    Extends the BaseModel with classifier-specific functionality.

    Attributes:
        num_classes (int): Number of classification classes.
    """

    def __init__(self, num_classes: int, model_name: str = "base_classifier"):
        """
        Initialize the base classifier.

        Args:
            num_classes (int): Number of classification classes.
            model_name (str): Name identifier for the model. Defaults to "base_classifier".
        """
        super().__init__(model_name=model_name, model_type="classifier")
        self.num_classes = num_classes

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Class probabilities.
        """
        return self.predict(x, return_probs=True)


class BaseAutoEncoder(BaseModel):
    """
    Base class for autoencoder models.

    Extends the BaseModel with autoencoder-specific functionality.

    Attributes:
        hidden_dim (int): Dimension of the latent space.
    """

    def __init__(self, hidden_dim: int, model_name: str = "base_autoencoder"):
        """
        Initialize the base autoencoder.

        Args:
            hidden_dim (int): Dimension of the latent space.
            model_name (str): Name identifier for the model.
        """
        super().__init__(model_name=model_name, model_type="autoencoder")
        self.hidden_dim = hidden_dim
        self.model_name = model_name

    @abstractmethod
    def encode(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode input to latent representation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Latent representation or (mu, sigma) for VAE.
        """
        raise NotImplementedError("Subclasses must implement `encode`.")

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to output.

        Args:
            z (torch.Tensor): Latent representation.

        Returns:
            torch.Tensor: Reconstructed output.
        """
        raise NotImplementedError("Subclasses must implement `decode`.")

    def get_latent(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get latent representation for input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Latent representation or (mu, sigma) for VAE.
        """
        self.eval()
        with torch.no_grad():
            return self.encode(x.to(self.device))


class BaseContrastiveModel(BaseModel):
    """
    Base class for contrastive learning models (SimCLR, MoCo, BYOL, DINO).

    Adds support for projection heads and representation learning without relying on labels.
    """

    def __init__(
        self,
        model_name: str = "base_contrastive",
        backbone_name: str = "resnet",
        proj_dim: int = 128,
    ):
        """
        Initialize the base contrastive model.

        Args:
            model_name (str): Model identifier.
            backbone_name (str): Backbone architecture name.
            proj_dim (int): Dimension of the projection head output.
        """
        super().__init__(model_name=model_name, model_type="contrastive")
        self.backbone_name = backbone_name
        self.proj_dim = proj_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning normalized projection for contrastive loss.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized projected representation.
        """
        raise NotImplementedError("Subclasses must implement `forward`.")

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return backbone features before projection head.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Backbone features.
        """
        raise NotImplementedError("Subclasses must implement `encode`.")

    @abstractmethod
    def project(self, h: torch.Tensor) -> torch.Tensor:
        """
        Return projection head output from backbone features.

        Args:
            h (torch.Tensor): Backbone features.

        Returns:
            torch.Tensor: Projected features.
        """
        raise NotImplementedError("Subclasses must implement `project`.")

    def predict(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Predict projection or raw embedding depending on flag.

        Args:
            x (torch.Tensor): Input tensor.
            return_embedding (bool): If True, returns raw backbone features (from kwargs).

        Returns:
            torch.Tensor: Projected or raw embedding.
        """
        self.eval()
        with torch.no_grad():
            x = x.to(self.device)
            h = self.encode(x)
            return h if kwargs.get("return_embedding", False) else self.forward(x)

    def summary(self) -> Dict[str, Any]:
        """
        Return model summary.

        Returns:
            Dict[str, Any]: Summary of model properties.
        """
        base = super().summary()
        base.update({"backbone": self.backbone_name, "projection_dim": self.proj_dim})
        return base


class BaseGAN(BaseModel):
    """
    Base class for Generative Adversarial Network models.

    Attributes:
        generator (Optional[torch.nn.Module]): Generator network.
        discriminator (Optional[torch.nn.Module]): Discriminator network.
    """

    generator: Optional[torch.nn.Module]
    discriminator: Optional[torch.nn.Module]

    def __init__(self, model_name: str = "base_gan"):
        """
        Initialize the base GAN.

        Args:
            model_name (str): Model name.
        """
        super().__init__(model_name=model_name, model_type="gan")
        self.generator: Optional[torch.nn.Module] = None
        self.discriminator: Optional[torch.nn.Module] = None

    @abstractmethod
    def generate(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Generate data using generator.

        Args:
            input_data (torch.Tensor): Input to generator.

        Returns:
            torch.Tensor: Generated output.
        """
        raise NotImplementedError("Subclasses must implement `generate`.")

    @abstractmethod
    def discriminate(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Discriminate input using discriminator.

        Args:
            input_data (torch.Tensor): Input to discriminator.

        Returns:
            torch.Tensor: Output score or probability.
        """
        raise NotImplementedError("Subclasses must implement `discriminate`.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass uses the generator.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Generated output.
        """
        return self.generate(x)

    def save_model(self, path: str) -> None:
        """
        Save model weights to disk.

        Args:
            path (str): Save path.
        """
        model_state = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "generator_state_dict": (
                self.generator.state_dict() if self.generator else None
            ),
            "discriminator_state_dict": (
                self.discriminator.state_dict() if self.discriminator else None
            ),
        }
        torch.save(model_state, path)
        print(f"GAN model saved to {path}")

    def load_model(self, path: str) -> None:
        """
        Load model weights from disk.

        Args:
            path (str): Load path.
        """
        checkpoint = torch.load(path, map_location=self.device)
        if self.generator:
            if "generator_state_dict" in checkpoint:
                self.generator.load_state_dict(checkpoint["generator_state_dict"])
        if self.discriminator:
            if "discriminator_state_dict" in checkpoint:
                self.discriminator.load_state_dict(
                    checkpoint["discriminator_state_dict"]
                )

        self.model_name = checkpoint.get("model_name", self.model_name)
        self.model_type = checkpoint.get("model_type", self.model_type)
        print(f"GAN model loaded from {path}")

    def _get_param_counts(self, module: Any) -> Tuple[int, int]:
        """
        Helper to count total and trainable parameters for a module.
        """
        if module is None:
            return 0, 0
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable

    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of the GAN model.

        Returns:
            Dict[str, Any]: Model summary information.
        """
        gen_params, gen_trainable = self._get_param_counts(self.generator)
        disc_params, disc_trainable = self._get_param_counts(self.discriminator)
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "device": self.device,
            "generator_parameters": gen_params,
            "generator_trainable_parameters": gen_trainable,
            "discriminator_parameters": disc_params,
            "discriminator_trainable_parameters": disc_trainable,
            "total_parameters": gen_params + disc_params,
            "total_trainable_parameters": gen_trainable + disc_trainable,
        }

    def to_device(self, device: torch.device) -> "BaseGAN":
        """
        Move model to specified device.

        Args:
            device (torch.device): Target device.

        Returns:
            BaseGAN: Self.
        """
        self.device = device
        if self.generator is not None:
            self.generator.to(device)
        if self.discriminator is not None:
            self.discriminator.to(device)
        return self
