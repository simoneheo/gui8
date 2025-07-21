
import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class autocorrelation_step(BaseStep):
    name = "autocorrelation"
    category = "Feature"
    description = "Compute autocorrelation and return as bar chart"
    tags = ["time-series", "autocorrelation", "bar chart"]
    params = [
        {"name": "max_lag", "type": "int", "default": "50", "help": "Maximum lag to compute"}
    ]

   
    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        max_lag = params.get("max_lag")
        
        if max_lag is None or max_lag <= 0:
            raise ValueError("Maximum lag must be positive")
        if max_lag > 1000:
            raise ValueError("Maximum lag too large (max 1000)")


    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray]:
        """Core processing logic"""
        max_lag = params["max_lag"]
        
        # Remove mean and compute autocorrelation
        y_centered = y - np.nanmean(y)
        autocorr = np.correlate(y_centered, y_centered, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr[:max_lag] / autocorr[0]
        
        # Create lag array
        x_new = np.arange(max_lag)
        
        return x_new, autocorr
