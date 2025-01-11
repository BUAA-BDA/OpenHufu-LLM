import yaml
import logging

from pathlib import Path

from openhufu.private.utlis.config_class import BaseConfig


def load_config(path: str) -> BaseConfig:
    path = Path(path)
    # check if config file exists
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")   
    # read config file
    with open(path, "r") as f:
        config = yaml.safe_load(f)
        
    return BaseConfig(**config)


def get_logger(name: str, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger