import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class moving_absmax_step(BaseStep):
    name = "moving absmax"
    category = "Filter"
    description = """Compute moving maximum of absolute values over a sliding window.
Useful for tracking peak amplitudes and signal envelope."""
    tags = ["time-series", "moving", "absmax", "envelope", "amplitude", "sliding-window"]
    params = [
        {"name": "window", "type": "int", "default": "10", "help": "Window size in samples for absmax computation"}
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
        
        # Compute moving absolute maximum
        y_absmax = np.zeros_like(y)
        
        for i in range(len(y)):
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            window_data = np.abs(y[start_idx:end_idx])
            y_absmax[i] = np.max(window_data)
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_absmax
            }
        ]
