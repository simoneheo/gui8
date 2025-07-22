
import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class cumulative_max_step(BaseStep):
    name = "cumulative_max"
    category = "Features"
    description = """Compute cumulative maximum of the signal.
Shows the running maximum value up to each point in time."""
    tags = ["cumulative", "maximum", "running", "envelope", "monotonic"]
    params = [
        # No parameters needed for cumulative max
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate cross-field logic and business rules"""
        # No parameters to validate for cumulative max
        pass

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        # Compute cumulative maximum
        y_cummax = np.maximum.accumulate(y)
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_cummax
            }
        ]
