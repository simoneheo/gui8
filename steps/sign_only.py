import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class sign_only_step(BaseStep):
    name = "sign_only"
    category = "Transform"
    description = """Extract only the sign of signal values (-1, 0, +1).
Useful for analyzing signal direction and zero crossings."""
    tags = ["time-series", "sign", "direction", "binary", "zero-crossing"]
    params = [
        # No parameters needed for sign extraction
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate cross-field logic and business rules"""
        # No parameters to validate for sign extraction
        pass

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        # Extract sign of signal values
        y_sign = np.sign(y)
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_sign
            }
        ]
