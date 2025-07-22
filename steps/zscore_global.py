import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class zscore_global_step(BaseStep):
    name = "zscore_global"
    category = "Transform"
    description = """Standardize signal using global z-score normalization.
Centers data around mean with unit standard deviation."""
    tags = ["time-series", "zscore", "standardize", "normalize", "global"]
    params = [
        {"name": "with_mean", "type": "bool", "default": "True", "help": "Center data around mean"},
        {"name": "with_std", "type": "bool", "default": "True", "help": "Scale to unit standard deviation"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate cross-field logic and business rules"""
        with_mean = params.get("with_mean", True)
        with_std = params.get("with_std", True)

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        with_mean = params["with_mean"]
        with_std = params["with_std"]
        
        # Check if signal has enough variation
        if len(y) < 2:
            raise ValueError("Signal too short for z-score normalization (minimum 2 samples)")
        
        # Calculate global statistics
        y_mean = np.mean(y)
        y_std = np.std(y)
        
        # Handle edge cases
        if y_std == 0:
            if with_std:
                raise ValueError("Cannot standardize: signal has zero standard deviation")
            y_std = 1.0  # Avoid division by zero
        
        # Apply z-score normalization
        y_standardized = y.copy()
        
        if with_mean:
            y_standardized = y_standardized - y_mean
        
        if with_std:
            y_standardized = y_standardized / y_std
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_standardized
            }
        ]
