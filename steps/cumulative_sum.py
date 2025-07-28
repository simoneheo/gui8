import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class cumulative_sum_step(BaseStep):
    name = "cumulative sum"
    category = "Features"
    description = """Compute cumulative sum of the signal.
Shows the running total up to each point in time."""
    tags = ["time-series", "cumulative", "sum", "running", "integral", "accumulation"]
    params = [
        # No parameters needed for cumulative sum
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate cross-field logic and business rules"""
        # No parameters to validate for cumulative sum
        pass

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        # Compute cumulative sum
        y_cumsum = np.cumsum(y)
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_cumsum
            }
        ]
