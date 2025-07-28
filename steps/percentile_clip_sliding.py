import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class percentile_clip_sliding_step(BaseStep):
    name = "percentile clip sliding"
    category = "Transform"
    description = """Clip signal values based on sliding window percentiles.
Removes outliers dynamically based on local statistics."""
    tags = ["time-series", "percentile", "clip", "outliers", "sliding", "robust"]
    params = [
        {"name": "window", "type": "int", "default": "100", "help": "Window size for percentile calculation"},
        {"name": "lower_percentile", "type": "float", "default": "5.0", "help": "Lower percentile threshold (0-50)"},
        {"name": "upper_percentile", "type": "float", "default": "95.0", "help": "Upper percentile threshold (50-100)"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate cross-field logic and business rules"""
        window = cls.validate_integer_parameter("window", params.get("window"), min_val=10)
        lower_percentile = cls.validate_numeric_parameter("lower_percentile", params.get("lower_percentile"), min_val=0.0, max_val=50.0)
        upper_percentile = cls.validate_numeric_parameter("upper_percentile", params.get("upper_percentile"), min_val=50.0, max_val=100.0)
        
        # Validate percentile relationship
        if lower_percentile >= upper_percentile:
            raise ValueError("lower_percentile must be less than upper_percentile")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        window = params["window"]
        lower_percentile = params["lower_percentile"]
        upper_percentile = params["upper_percentile"]
        
        # Check if signal is long enough
        if len(y) < window:
            raise ValueError(f"Signal too short: requires at least {window} samples")
        
        # Apply sliding percentile clipping
        y_clipped = np.zeros_like(y)
        
        for i in range(len(y)):
            # Define window boundaries
            start_idx = max(0, i - window // 2)
            end_idx = min(len(y), i + window // 2 + 1)
            
            # Extract window data
            window_data = y[start_idx:end_idx]
            
            # Calculate percentiles for this window
            lower_thresh = np.percentile(window_data, lower_percentile)
            upper_thresh = np.percentile(window_data, upper_percentile)
            
            # Clip the current value
            y_clipped[i] = np.clip(y[i], lower_thresh, upper_thresh)
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_clipped
            }
        ]
