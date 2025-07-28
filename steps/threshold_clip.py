import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class threshold_clip_step(BaseStep):
    name = "threshold clip"
    category = "Transform"
    description = """Clip signal values above/below thresholds to specified values.
Useful for removing outliers and limiting extreme values."""
    tags = ["time-series", "threshold", "clip", "outliers", "limit", "bounds"]
    params = [
        {"name": "min_threshold", "type": "float", "default": "", "help": "Minimum threshold (leave blank for no lower limit)"},
        {"name": "max_threshold", "type": "float", "default": "", "help": "Maximum threshold (leave blank for no upper limit)"},
        {"name": "min_value", "type": "float", "default": "0.0", "help": "Value to assign to samples below min_threshold"},
        {"name": "max_value", "type": "float", "default": "1.0", "help": "Value to assign to samples above max_threshold"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate cross-field logic and business rules"""
        # Thresholds can be empty (no limit) or numeric values
        if params.get("min_threshold") not in [None, "", "auto"]:
            cls.validate_numeric_parameter("min_threshold", params.get("min_threshold"))
        
        if params.get("max_threshold") not in [None, "", "auto"]:
            cls.validate_numeric_parameter("max_threshold", params.get("max_threshold"))
        
        min_value = cls.validate_numeric_parameter("min_value", params.get("min_value"))
        max_value = cls.validate_numeric_parameter("max_value", params.get("max_value"))

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        min_threshold = params.get("min_threshold", "")
        max_threshold = params.get("max_threshold", "")
        min_value = params["min_value"]
        max_value = params["max_value"]
        
        # Auto-detect thresholds if not specified
        if min_threshold in [None, "", "auto"]:
            min_threshold = np.percentile(y, 5)  # 5th percentile
        
        if max_threshold in [None, "", "auto"]:
            max_threshold = np.percentile(y, 95)  # 95th percentile
        
        # Convert to float if they were strings
        min_threshold = float(min_threshold)
        max_threshold = float(max_threshold)
        
        # Apply threshold clipping
        y_clipped = y.copy()
        
        # Clip values below minimum threshold
        if min_threshold is not None:
            y_clipped[y < min_threshold] = min_value
        
        # Clip values above maximum threshold
        if max_threshold is not None:
            y_clipped[y > max_threshold] = max_value
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_clipped
            }
        ]
