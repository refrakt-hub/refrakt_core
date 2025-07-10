from .inference_helpers import OmegaConf as inference_OmegaConf
from .inference_helpers import load_config as inference_load_config
from .train_helpers import OmegaConf as train_OmegaConf
from .train_helpers import load_config as train_load_config

# For compatibility, provide direct access
load_config = inference_load_config
OmegaConf = inference_OmegaConf
