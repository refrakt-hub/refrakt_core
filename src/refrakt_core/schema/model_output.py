# refrakt_core/outputs/model_output.py

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class ModelOutput:
    embeddings: Optional[Any] = None         # contrastive / latent features
    logits: Optional[Any] = None             # supervised output
    image: Optional[Any] = None              # GAN or output image
    reconstruction: Optional[Any] = None     # AE / VAE
    attention_maps: Optional[Any] = None     # ViT, DINO
    loss_components: Dict[str, Any] = field(default_factory=dict)  # for contrastive/self-sup
    extra: Dict[str, Any] = field(default_factory=dict)            # custom keys (e.g. jacobians, forces)
