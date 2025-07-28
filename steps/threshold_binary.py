import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class threshold_binary_step(BaseStep):
    name = "threshold binary"
    category = "Transform"
    description = """Convert signal to binary (0/1) based on threshold comparison.
Values above threshold become 1, values below become 0."""
    tags = ["time-series", "threshold", "binary", "binarization", "comparison"]
    params = [
        {"name": "threshold", "type": "float", "default": "0.0", "help": "Threshold value for binary conversion"},
        {"name": "above_value", "type": "float", "default": "1.0", "help": "Output value for samples above threshold"},
        {"name": "below_value", "type": "float", "default": "0.0", "help": "Output value for samples below threshold"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate cross-field logic and business rules"""
        threshold = cls.validate_numeric_parameter("threshold", params.get("threshold"))
        above_value = cls.validate_numeric_parameter("above_value", params.get("above_value"))
        below_value = cls.validate_numeric_parameter("below_value", params.get("below_value"))

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        threshold = params.get("threshold", 0.0)
        above_value = params.get("above_value", 1.0)
        below_value = params.get("below_value", 0.0)
        
        # Apply binary thresholding
        y_binary = np.where(y > threshold, above_value, below_value)
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_binary
            }
        ]
