from pathlib import Path 
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_config(config_path):
    config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config file: {config_path}")
    return config

def resolve_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path