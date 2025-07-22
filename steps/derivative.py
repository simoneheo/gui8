import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class derivative_step(BaseStep):
    name = "derivative"
    category = "Transform"
    description = """Compute the derivative (rate of change) of the signal using finite differences.
Useful for detecting edges, transitions, and changes in signal behavior."""
    tags = [ "derivative", "finite-difference", "rate-of-change", "gradient"]
    params = [
        {"name": "method", "type": "str", "default": "forward", "options": ["forward", "backward", "central"], "help": "Finite difference method"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate cross-field logic and business rules"""
        method = cls.validate_string_parameter("method", params.get("method"), 
                                              valid_options=["forward", "backward", "central"])

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        method = params.get("method", "forward")
        
        if len(y) < 2:
            raise ValueError("Signal too short for derivative computation (minimum 2 samples)")
        
        if method == "forward":
            y_new = np.diff(y, append=y[-1])  # Forward difference
        elif method == "backward":
            y_new = np.diff(y, prepend=y[0])  # Backward difference
        elif method == "central":
            # Central difference
            y_new = np.zeros_like(y)
            y_new[1:-1] = (y[2:] - y[:-2]) / 2.0
            y_new[0] = y[1] - y[0]  # Forward difference for first point
            y_new[-1] = y[-1] - y[-2]  # Backward difference for last point
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_new
            }
        ]
