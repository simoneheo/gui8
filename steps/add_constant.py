import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class add_constant_step(BaseStep):
    name = "add_constant"
    category = "Transform"
    description = """Add a constant value to the signal."""
    tags = ["time-series", "add", "constant", "transform"]
    params = [
        {"name": "constant", "type": "float", "default": "1.0", "help": "Constant value to add"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        constant = cls.validate_numeric_parameter("constant", params.get("constant"))

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        constant = params["constant"]
        y_added = y + constant
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_added
            }
        ]