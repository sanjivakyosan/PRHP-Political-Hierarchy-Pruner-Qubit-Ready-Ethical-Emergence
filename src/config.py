"""
Configuration management for PRHP framework.

Copyright © sanjivakyosan 2025
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path

try:
    from .utils import get_logger
except ImportError:
    from utils import get_logger

logger = get_logger()

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default config.

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Use default config
        config_dir = Path(__file__).parent.parent / "config"
        config_path = config_dir / "default.yaml"

    config_path = Path(config_path)

    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return get_default_config()

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}, using defaults")
        return get_default_config()

def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        'simulation': {
            'levels': 9,
            'n_monte': 100,
            'seed': 42,
            'use_quantum': True,
            'track_levels': True,
            'show_progress': True
        },
        'variants': ['ADHD-collectivist', 'autistic-individualist', 'neurotypical-hybrid'],
        'dopamine_gradients': {
            'ADHD-collectivist': 0.20,
            'autistic-individualist': 0.15,
            'neurotypical-hybrid': 0.18
        },
        'targets': {
            'fidelity': 0.84,
            'phi_delta': 0.12,
            'phi_delta_tolerance': 0.025
        }
    }

def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get nested config value using dot notation.

    Args:
        config: Configuration dictionary
        key_path: Dot-separated path (e.g., 'simulation.levels')
        default: Default value if key not found

    Returns:
        Config value or default
    """
    keys = key_path.split('.')
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value

