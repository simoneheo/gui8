import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class reciprocal_step(BaseStep):
    name = "reciprocal"
    category = "Transform"
    description = """Transform signal by taking the reciprocal of each value.
Useful for inverting signals and certain normalization tasks."""
    tags = ["time-series", "reciprocal", "invert", "transform", "normalize"]
    params = [
        {"name": "epsilon", "type": "float", "default": "1e-8", "help": "Small value to avoid division by zero"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        epsilon = cls.validate_numeric_parameter("epsilon", params.get("epsilon"), min_val=0.0)

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        epsilon = params["epsilon"]
        
        # Avoid division by zero
        y_safe = np.where(np.abs(y) < epsilon, np.sign(y) * epsilon + (y == 0) * epsilon, y)
        y_reciprocal = 1.0 / y_safe
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_reciprocal
            }
        ]
