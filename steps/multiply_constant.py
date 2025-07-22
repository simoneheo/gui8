import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class multiply_constant_step(BaseStep):
    name = "multiply_constant"
    category = "Transform"
    description = """Multiply the signal by a constant value."""
    tags = ["time-series", "multiply", "constant", "transform"]
    params = [
        {"name": "constant", "type": "float", "default": "2.0", "help": "Constant value to multiply"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        constant = cls.validate_numeric_parameter("constant", params.get("constant"))

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        constant = params["constant"]
        y_multiplied = y * constant
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_multiplied
            }
        ]
