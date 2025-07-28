import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class percentile_clip_step(BaseStep):
    name = "percentile clip"
    category = "Transform"
    description = """Clip signal values based on global percentiles.
Removes outliers based on overall signal statistics."""
    tags = ["time-series", "percentile", "clip", "outliers", "global", "robust"]
    params = [
        {"name": "lower_percentile", "type": "float", "default": "5.0", "help": "Lower percentile threshold (0-50)"},
        {"name": "upper_percentile", "type": "float", "default": "95.0", "help": "Upper percentile threshold (50-100)"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate cross-field logic and business rules"""
        lower_percentile = cls.validate_numeric_parameter("lower_percentile", params.get("lower_percentile"), min_val=0.0, max_val=50.0)
        upper_percentile = cls.validate_numeric_parameter("upper_percentile", params.get("upper_percentile"), min_val=50.0, max_val=100.0)
        
        # Validate percentile relationship
        if lower_percentile >= upper_percentile:
            raise ValueError("lower_percentile must be less than upper_percentile")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        lower_percentile = params["lower_percentile"]
        upper_percentile = params["upper_percentile"]
        
        # Calculate global percentile thresholds
        lower_threshold = np.percentile(y, lower_percentile)
        upper_threshold = np.percentile(y, upper_percentile)
        
        # Clip values to percentile range
        y_clipped = np.clip(y, lower_threshold, upper_threshold)
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_clipped
            }
        ]
