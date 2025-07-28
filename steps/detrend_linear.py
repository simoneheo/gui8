import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class detrend_linear_step(BaseStep):
    name = "detrend linear"
    category = "Transform"
    description = """Remove linear trend from signal by subtracting the best-fit line.
Useful for removing drift and baseline shifts."""
    tags = ["time-series", "detrend", "linear", "baseline", "drift", "preprocessing"]
    params = [
        # No parameters needed for linear detrend
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate cross-field logic and business rules"""
        # No parameters to validate for linear detrend
        pass

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        from scipy.signal import detrend
        
        # Remove linear trend
        y_detrended = detrend(y, type='linear')
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_detrended
            }
        ]
