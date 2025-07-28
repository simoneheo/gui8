
import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class bollinger_bands_step(BaseStep):
    name = "bollinger bands"
    category = "Features"
    description = """Compute Bollinger Bands (moving average with standard deviation bands)."""
    tags = ["bollinger", "bands", "volatility", "moving-average", "statistics"]
    params = [
        {"name": "window", "type": "int", "default": "20", "help": "Window size for moving average and std"},
        {"name": "std_multiplier", "type": "float", "default": "2.0", "help": "Multiplier for standard deviation bands"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate cross-field logic and business rules"""
        window = cls.validate_integer_parameter("window", params.get("window"), min_val=2)
        std_multiplier = cls.validate_numeric_parameter("std_multiplier", params.get("std_multiplier"), min_val=0.1)

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        window = params["window"]
        std_multiplier = params["std_multiplier"]
        
        # Check if signal is long enough for the window
        if len(y) < window:
            raise ValueError(f"Signal too short: requires at least {window} samples")
        
        # Compute moving average
        y_ma = np.zeros_like(y)
        for i in range(len(y)):
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            y_ma[i] = np.mean(y[start_idx:end_idx])
        
        # Compute moving standard deviation
        y_std = np.zeros_like(y)
        for i in range(len(y)):
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            window_data = y[start_idx:end_idx]
            if len(window_data) > 1:
                y_std[i] = np.std(window_data, ddof=1)
            else:
                y_std[i] = 0.0
        
        # Compute Bollinger Bands
        upper_band = y_ma + (std_multiplier * y_std)
        lower_band = y_ma - (std_multiplier * y_std)
        
        # Return all three bands as separate channels
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': upper_band
            },
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_ma
            },
            {
                'tags': ['time-series'],
                'x': x,
                'y': lower_band
            }
        ]
