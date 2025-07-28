import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class exp_smooth_step(BaseStep):
    name = "exp smooth"
    category = "Filter"
    description = """Apply exponential smoothing to the signal.
Uses a weighted average where recent samples have more influence."""
    tags = ["exponential", "smoothing", "filter", "weighted-average"]
    params = [
        {"name": "alpha", "type": "float", "default": "0.3", "help": "Smoothing factor (0-1, higher = more smoothing)"},
        {"name": "method", "type": "str", "default": "simple", "options": ["simple", "double", "triple"], "help": "Exponential smoothing method"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate cross-field logic and business rules"""
        alpha = cls.validate_numeric_parameter("alpha", params.get("alpha"), min_val=0.0, max_val=1.0)
        method = cls.validate_string_parameter("method", params.get("method"), 
                                              valid_options=["simple", "double", "triple"])

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        alpha = params["alpha"]
        method = params["method"]
        
        if method == "simple":
            # Simple exponential smoothing
            y_smoothed = np.zeros_like(y)
            y_smoothed[0] = y[0]  # Initialize with first value
            
            for i in range(1, len(y)):
                y_smoothed[i] = alpha * y[i] + (1 - alpha) * y_smoothed[i-1]
        
        elif method == "double":
            # Double exponential smoothing (Holt's method)
            y_smoothed = np.zeros_like(y)
            trend = np.zeros_like(y)
            
            # Initialize
            y_smoothed[0] = y[0]
            trend[0] = y[1] - y[0] if len(y) > 1 else 0
            
            for i in range(1, len(y)):
                y_prev = y_smoothed[i-1]
                trend_prev = trend[i-1]
                
                y_smoothed[i] = alpha * y[i] + (1 - alpha) * (y_prev + trend_prev)
                trend[i] = alpha * (y_smoothed[i] - y_prev) + (1 - alpha) * trend_prev
        
        elif method == "triple":
            # Triple exponential smoothing (Holt-Winters method)
            y_smoothed = np.zeros_like(y)
            trend = np.zeros_like(y)
            seasonal = np.zeros_like(y)
            
            # Simple initialization
            y_smoothed[0] = y[0]
            trend[0] = 0
            seasonal[0] = 0
            
            for i in range(1, len(y)):
                y_prev = y_smoothed[i-1]
                trend_prev = trend[i-1]
                seasonal_prev = seasonal[i-1]
                
                y_smoothed[i] = alpha * (y[i] - seasonal_prev) + (1 - alpha) * (y_prev + trend_prev)
                trend[i] = alpha * (y_smoothed[i] - y_prev) + (1 - alpha) * trend_prev
                seasonal[i] = alpha * (y[i] - y_smoothed[i]) + (1 - alpha) * seasonal_prev
        
        else:
            raise ValueError(f"Unknown exponential smoothing method: {method}")
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_smoothed
            }
        ]
