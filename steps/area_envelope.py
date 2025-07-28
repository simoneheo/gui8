import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class area_envelope_step(BaseStep):
    name = "area envelope"
    category = "Features"
    description = """Compute area envelope by integrating signal over sliding windows.
Measures cumulative signal energy/area in each window."""
    tags = ["area", "envelope", "integration", "energy", "sliding-window"]
    params = [
        {"name": "window", "type": "int", "default": "50", "help": "Window size in samples for area computation"}
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
        
        # Compute area envelope using sliding window integration
        y_area = np.zeros_like(y)
        
        for i in range(len(y)):
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            window_data = y[start_idx:end_idx]
            
            # Compute area under the curve in this window
            if len(window_data) > 1:
                # Use trapezoidal integration
                y_area[i] = np.trapz(window_data, dx=1.0/fs if fs > 0 else 1.0)
            else:
                y_area[i] = window_data[0] if len(window_data) == 1 else 0.0
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_area
            }
        ]