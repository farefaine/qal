# config package
# Configuration management utilities

import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file (relative to project root or absolute)
    
    Returns:
        Dictionary containing configuration
    
    Example:
        >>> from config import load_config
        >>> model_cfg = load_config('config/model/small.yaml')
        >>> train_cfg = load_config('config/training/default.yaml')
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save YAML file
    
    Example:
        >>> from config import save_config
        >>> config = {'tokenizer': {'vocab_size': 8192}}
        >>> save_config(config, 'config/tokenizer/custom.yaml')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


__all__ = [
    'load_config',
    'save_config',
]
