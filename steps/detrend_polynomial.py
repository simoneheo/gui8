import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class detrend_polynomial_step(BaseStep):
    name = "detrend_polynomial"
    category = "Transform"
    description = """Remove polynomial trend from signal by fitting and subtracting a polynomial.
Useful for removing non-linear trends and drift."""
    tags = ["time-series", "detrend", "polynomial", "trend", "drift"]
    params = [
        {"name": "order", "type": "int", "default": "2", "help": "Polynomial order (1=linear, 2=quadratic, etc.)"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        order = cls.validate_integer_parameter("order", params.get("order"), min_val=1, max_val=10)

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        order = params["order"]
        
        if len(y) < order + 1:
            raise ValueError(f"Signal too short: need at least {order + 1} samples for order {order} polynomial")

        # Fit polynomial to the data
        coeffs = np.polyfit(x, y, order)
        trend = np.polyval(coeffs, x)

        # Subtract the trend
        y_detrended = y - trend

        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_detrended
            }
        ]
