import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class abs_val_step(BaseStep):
    name = "abs_val"
    category = "Transform"
    description = """Take the absolute value of the signal."""
    tags = ["time-series", "absolute", "abs", "transform"]
    params = []

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        pass

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        y_abs = np.abs(y)
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_abs
            }
        ]
