import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class modulo_step(BaseStep):
    name = "modulo"
    category = "Transform"
    description = """Apply modulo operation to signal values.
Wraps values to a specified range [0, divisor) or [offset, offset+divisor)."""
    tags = ["time-series", "modulo", "wrap", "cyclic", "periodic", "arithmetic"]
    params = [
        {"name": "divisor", "type": "float", "default": "1.0", "help": "Divisor for modulo operation"},
        {"name": "offset", "type": "float", "default": "0.0", "help": "Offset for the modulo range"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate cross-field logic and business rules"""
        divisor = cls.validate_numeric_parameter("divisor", params.get("divisor"), min_val=1e-10)
        offset = cls.validate_numeric_parameter("offset", params.get("offset"))

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        divisor = params["divisor"]
        offset = params["offset"]
        
        # Apply modulo operation with offset
        y_modulo = ((y - offset) % divisor) + offset
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_modulo
            }
        ]
