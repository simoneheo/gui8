import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class savitzky_golay_step(BaseStep):
    name = "savitzky_golay"
    category = "Filter"
    description = """Apply Savitzky-Golay smoothing filter that preserves features like peaks.
Better than simple smoothing for maintaining signal characteristics."""
    tags = ["time-series", "smoothing", "savgol", "polynomial", "edge-preserving", "scipy"]
    params = [
        {"name": "window", "type": "int", "default": "11", "help": "Window size (odd number >= 5)"},
        {"name": "polyorder", "type": "int", "default": "3", "help": "Polynomial order (must be < window size)"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate cross-field logic and business rules"""
        window = cls.validate_integer_parameter("window", params.get("window"), min_val=5)
        polyorder = cls.validate_integer_parameter("polyorder", params.get("polyorder"), min_val=1)
        
        if window % 2 == 0:
            raise ValueError("Window size must be odd")
        if polyorder >= window:
            raise ValueError(f"Polynomial order ({polyorder}) must be less than window size ({window})")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        from scipy.signal import savgol_filter
        
        window = params["window"]
        polyorder = params["polyorder"]
        
        # Check if signal is long enough for the window
        if len(y) < window:
            raise ValueError(f"Signal too short: requires at least {window} samples")
        
        # Apply Savitzky-Golay filter
        y_filtered = savgol_filter(y, window_length=window, polyorder=polyorder)
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_filtered
            }
        ]
