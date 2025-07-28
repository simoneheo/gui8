import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class log_transform_step(BaseStep):
    name = "log transform"
    category = "Transform"
    description = """Apply logarithmic transformation to the signal."""
    tags = ["time-series", "log", "transform", "nonlinear"]
    params = [
        {"name": "base", "type": "str", "default": "natural", "options": ["natural", "10", "2"], "help": "Logarithm base"},
        {"name": "offset", "type": "float", "default": "1.0", "help": "Offset to add before taking log"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        base = cls.validate_string_parameter("base", params.get("base"), 
                                            valid_options=["natural", "10", "2"])
        offset = cls.validate_numeric_parameter("offset", params.get("offset"))

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        base = params["base"]
        offset = params["offset"]
        
        # Add offset to avoid log(0) or log(negative)
        y_offset = y + offset
        
        # Check for negative values after offset
        if np.any(y_offset <= 0):
            raise ValueError("All values must be positive after adding offset")
        
        # Apply logarithmic transformation
        if base == "natural":
            y_log = np.log(y_offset)
        elif base == "10":
            y_log = np.log10(y_offset)
        elif base == "2":
            y_log = np.log2(y_offset)
        else:
            raise ValueError(f"Unknown log base: {base}")
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_log
            }
        ]
