import numpy as np

from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class smooth_normalize_step(BaseStep):
    name = "normalize"
    category = "Filter"
    description = """Normalize signal using local mean and range with uniform filtering.
Removes local trends and variations while preserving global structure."""
    tags = [ "normalize", "smoothing", "local", "scaling", "range"]
    params = [
        {"name": "window", "type": "int", "default": "101", "help": "Window size for smoothing (odd number >= 3)"},
        {"name": "scale_0_1", "type": "bool", "default": "True", "help": "Rescale normalized signal to [0,1]"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        window = cls.validate_integer_parameter("window", params.get("window"), min_val=3)
        
        if window % 2 == 0:
            raise ValueError("Window size must be odd for symmetric filtering")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        from scipy.ndimage import uniform_filter1d
        window = params["window"]
        scale_0_1 = params.get("scale_0_1", True)
        
        # Check if signal is long enough for the window
        if len(y) < window:
            raise ValueError(f"Signal too short: requires at least {window} samples")
        
        # Compute local mean and range using uniform filtering
        local_mean = uniform_filter1d(y, size=window, mode='reflect')
        
        # Compute local range (max - min) in sliding window
        local_max = uniform_filter1d(np.maximum.accumulate(y), size=window, mode='reflect')
        local_min = uniform_filter1d(np.minimum.accumulate(y), size=window, mode='reflect')
        local_range = local_max - local_min
        
        # Avoid division by zero
        local_range = np.maximum(local_range, 1e-10)
        
        # Normalize: (signal - local_mean) / local_range
        y_normalized = (y - local_mean) / local_range
        
        # Optionally scale to [0,1]
        if scale_0_1:
            y_min, y_max = np.min(y_normalized), np.max(y_normalized)
            if y_max > y_min:
                y_normalized = (y_normalized - y_min) / (y_max - y_min)
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_normalized
            }
        ]
