import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class loess_smooth_step(BaseStep):
    name = "loess_smooth"
    category = "Filter"
    description = """Apply LOESS (Locally Weighted Scatterplot Smoothing) to the signal.
Non-parametric smoothing that fits local polynomials."""
    tags = ["time-series", "loess", "smoothing", "non-parametric", "local-polynomial"]
    params = [
        {"name": "frac", "type": "float", "default": "0.3", "help": "Fraction of data to use in local regression (0-1)"},
        {"name": "degree", "type": "int", "default": "1", "help": "Degree of local polynomial (1=linear, 2=quadratic)"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate cross-field logic and business rules"""
        frac = cls.validate_numeric_parameter("frac", params.get("frac"), min_val=0.01, max_val=1.0)
        degree = cls.validate_integer_parameter("degree", params.get("degree"), min_val=0, max_val=2)

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        frac = params["frac"]
        degree = params["degree"]
        
        # Check if signal is long enough
        if len(y) < 10:
            raise ValueError("Signal too short for LOESS smoothing (minimum 10 samples)")
        
        # Use statsmodels for LOESS smoothing
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            y_smoothed = lowess(y, x, frac=frac, it=0, delta=0.0, return_sorted=False)
        except ImportError:
            # Fallback to simple moving average if statsmodels not available
            window_size = max(3, int(frac * len(y)))
            y_smoothed = np.convolve(y, np.ones(window_size)/window_size, mode='same')
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_smoothed
            }
        ]
