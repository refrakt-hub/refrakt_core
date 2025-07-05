import os

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from refrakt_core.schema.yaml_validator import RefraktConfig

CONFIG_DIR = "./src/refrakt_core/config/"


def get_all_yaml_paths():
    return [
        os.path.join(CONFIG_DIR, f)
        for f in os.listdir(CONFIG_DIR)
        if f.endswith(".yaml") or f.endswith(".yml")
    ]


@pytest.mark.parametrize("yaml_path", get_all_yaml_paths())
def test_yaml_config_validates(yaml_path):
    config_omega = OmegaConf.load(yaml_path)
    config_dict = OmegaConf.to_container(config_omega, resolve=True)

    try:
        validated_config = RefraktConfig(**config_dict)
    except ValidationError as e:
        pytest.fail(f"Validation failed for '{yaml_path}':\n{e}")
