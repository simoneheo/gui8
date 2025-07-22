import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class exp_transform_step(BaseStep):
    name = "exp_transform"
    category = "Transform"
    description = """Apply exponential transformation to the signal. 
    
    Base options:
    - natural: Uses e (≈2.718) as the base, applies e^y transformation
    - 10: Uses 10 as the base, applies 10^y transformation (common for log-scale data)
    - 2: Uses 2 as the base, applies 2^y transformation (useful for binary/digital signals)"""
    tags = ["time-series", "exponential", "exp", "transform", "nonlinear"]
    params = [
        {"name": "base", "type": "str", "default": "natural", "options": ["natural", "10", "2"], "help": "Exponential base"},
        {"name": "scale", "type": "float", "default": "1.0", "help": "Scaling factor"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        base = cls.validate_string_parameter("base", params.get("base"), 
                                            valid_options=["natural", "10", "2"])
        scale = cls.validate_numeric_parameter("scale", params.get("scale"))

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        base = params["base"]
        scale = params["scale"]
        
        # Apply scaling first
        y_scaled = y * scale
        
        # Apply exponential transformation
        if base == "natural":
            y_exp = np.exp(y_scaled)
        elif base == "10":
            y_exp = 10 ** y_scaled
        elif base == "2":
            y_exp = 2 ** y_scaled
        else:
            raise ValueError(f"Unknown exponential base: {base}")
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_exp
            }
        ]
