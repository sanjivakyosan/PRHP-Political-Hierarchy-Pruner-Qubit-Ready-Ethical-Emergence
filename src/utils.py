"""
Utility functions for PRHP framework: validation, logging, and helpers.

Copyright © sanjivakyosan 2025
"""

import logging
import sys
from typing import Any, Dict, List, Optional, Union
import numpy as np

# Configure logging
def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging for PRHP framework.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger('prhp')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Get logger instance
_logger = setup_logging()

def validate_positive_int(value: Any, name: str, min_value: int = 1) -> int:
    """Validate that value is a positive integer."""
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    return int(value)

def validate_float_range(value: Any, name: str, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Validate that value is a float in specified range."""
    try:
        fval = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if not (min_val <= fval <= max_val):
        raise ValueError(f"{name} must be in [{min_val}, {max_val}], got {fval}")
    return fval

def validate_variant(variant: str) -> str:
    """Validate neuro-cultural variant."""
    valid_variants = ['ADHD-collectivist', 'autistic-individualist', 'neurotypical-hybrid', 'trauma-survivor-equity']
    if variant not in valid_variants:
        raise ValueError(f"variant must be one of {valid_variants}, got {variant}")
    return variant

def validate_variants(variants: List[str]) -> List[str]:
    """Validate list of variants."""
    if not isinstance(variants, list) or len(variants) == 0:
        raise ValueError("variants must be a non-empty list")
    return [validate_variant(v) for v in variants]

def validate_seed(seed: Optional[int]) -> Optional[int]:
    """Validate random seed."""
    if seed is not None:
        if not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {seed}")
        return int(seed)
    return None

def get_logger() -> logging.Logger:
    """Get the PRHP logger instance."""
    return _logger

