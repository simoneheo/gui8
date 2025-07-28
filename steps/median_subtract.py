import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class median_subtract_step(BaseStep):
    name = "median subtract"
    category = "Filter"
    description = """Subtract rolling median from signal to remove local trends.
More robust than mean subtraction for signals with outliers."""
    tags = ["time-series", "rolling", "median", "subtract", "detrend", "robust"]
    params = [
        {"name": "window", "type": "int", "default": "50", "help": "Window size for rolling median computation"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate cross-field logic and business rules"""
        window = cls.validate_integer_parameter("window", params.get("window"), min_val=1)

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        window = params["window"]
        
        # Check if signal is long enough for the window
        if len(y) < window:
            raise ValueError(f"Signal too short: requires at least {window} samples")
        
        # Compute rolling median
        y_rolling_median = np.zeros_like(y)
        for i in range(len(y)):
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            y_rolling_median[i] = np.median(y[start_idx:end_idx])
        
        # Subtract rolling median from original signal
        y_detrended = y - y_rolling_median
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_detrended
            }
        ]
