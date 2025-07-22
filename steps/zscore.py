import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class zscore_step(BaseStep):
    name = "zscore"
    category = "Transform"
    description = """Standardize signal to zero mean and unit variance."""
    tags = ["standardize", "normalize", "zscore", "transform"]
    params = []

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        pass

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        # Calculate mean and standard deviation
        mean = np.mean(y)
        std = np.std(y)
        
        # Handle zero standard deviation
        if std == 0:
            y_standardized = np.zeros_like(y)
        else:
            y_standardized = (y - mean) / std
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_standardized
            }
        ]
