import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class smoother_step(BaseStep):
    name = "smoother_step"
    category = "Filter"
    description = """Apply various smoothing methods to the signal.
Supports multiple smoothing algorithms with configurable parameters."""
    tags = ["time-series", "smoothing", "filter", "noise-reduction", "signal-processing"]
    params = [
        {"name": "method", "type": "str", "default": "moving_average", "options": ["moving_average", "gaussian", "savitzky_golay", "median"], "help": "Smoothing method"},
        {"name": "window", "type": "int", "default": "11", "help": "Window size for smoothing"},
        {"name": "sigma", "type": "float", "default": "1.0", "help": "Standard deviation for Gaussian smoothing"},
        {"name": "polyorder", "type": "int", "default": "3", "help": "Polynomial order for Savitzky-Golay"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate cross-field logic and business rules"""
        method = cls.validate_string_parameter("method", params.get("method"), 
                                              valid_options=["moving_average", "gaussian", "savitzky_golay", "median"])
        window = cls.validate_integer_parameter("window", params.get("window"), min_val=3)
        sigma = cls.validate_numeric_parameter("sigma", params.get("sigma"), min_val=0.1)
        polyorder = cls.validate_integer_parameter("polyorder", params.get("polyorder"), min_val=1, max_val=5)
        
        # Validate window size constraints
        if method == "savitzky_golay" and window <= polyorder:
            raise ValueError("Window size must be greater than polynomial order for Savitzky-Golay")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        method = params["method"]
        window = params["window"]
        sigma = params["sigma"]
        polyorder = params["polyorder"]
        
        # Check if signal is long enough
        if len(y) < window:
            raise ValueError(f"Signal too short: requires at least {window} samples")
        
        # Apply smoothing based on method
        if method == "moving_average":
            # Simple moving average
            y_smoothed = np.convolve(y, np.ones(window)/window, mode='same')
            
        elif method == "gaussian":
            # Gaussian smoothing
            from scipy.ndimage import gaussian_filter1d
            y_smoothed = gaussian_filter1d(y, sigma=sigma)
            
        elif method == "savitzky_golay":
            # Savitzky-Golay smoothing
            from scipy.signal import savgol_filter
            y_smoothed = savgol_filter(y, window, polyorder)
            
        elif method == "median":
            # Median smoothing
            from scipy.signal import medfilt
            y_smoothed = medfilt(y, kernel_size=window)
            
        else:
            raise ValueError(f"Unknown smoothing method: {method}")
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_smoothed
            }
        ]