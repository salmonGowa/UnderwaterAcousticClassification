"""Load a YAML experiment config and resolve the data directory."""

import os
from pathlib import Path

import yaml


def load_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    config["config_path"] = str(path)
    config.setdefault("name", Path(path).stem)
    # The environment variable wins so the same config works locally and on Colab/Drive.
    config["data_root"] = os.environ.get("UAC_DATA_ROOT", config.get("data_root", "data"))
    return config


def data_path(config, *parts):
    return Path(config["data_root"]).joinpath(*parts)
