
import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class volatility_estimation_step(BaseStep):
    name = "volatility_estimation"
    category = "Features"
    description = """Estimate signal volatility using various methods.
Measures the variability and uncertainty in the signal over time."""
    tags = ["time-series", "volatility", "variance", "uncertainty", "risk", "variability"]
    params = [
        {"name": "method", "type": "str", "default": "rolling_std", "options": ["rolling_std", "garch", "ewma"], "help": "Volatility estimation method"},
        {"name": "window", "type": "int", "default": "20", "help": "Window size for rolling calculations"},
        {"name": "lambda_param", "type": "float", "default": "0.94", "help": "Decay factor for EWMA (0-1)"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate cross-field logic and business rules"""
        method = cls.validate_string_parameter("method", params.get("method"), 
                                              valid_options=["rolling_std", "garch", "ewma"])
        window = cls.validate_integer_parameter("window", params.get("window"), min_val=2)
        lambda_param = cls.validate_numeric_parameter("lambda_param", params.get("lambda_param"), min_val=0.0, max_val=1.0)

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        method = params["method"]
        window = params["window"]
        lambda_param = params["lambda_param"]
        
        # Check if signal is long enough
        if len(y) < window:
            raise ValueError(f"Signal too short: requires at least {window} samples")
        
        # Calculate returns (first differences)
        returns = np.diff(y)
        
        if method == "rolling_std":
            # Rolling standard deviation
            volatility = np.zeros_like(y)
            volatility[0] = np.std(returns[:min(window, len(returns))])
            
            for i in range(1, len(y)):
                start_idx = max(0, i - window)
                end_idx = min(len(returns), i)
                if end_idx > start_idx:
                    volatility[i] = np.std(returns[start_idx:end_idx])
                else:
                    volatility[i] = volatility[i-1]
        
        elif method == "ewma":
            # Exponentially Weighted Moving Average
            volatility = np.zeros_like(y)
            volatility[0] = np.var(returns[:min(window, len(returns))])
            
            for i in range(1, len(y)):
                if i < len(returns):
                    volatility[i] = lambda_param * volatility[i-1] + (1 - lambda_param) * returns[i-1]**2
                else:
                    volatility[i] = volatility[i-1]
            
            # Convert variance to standard deviation
            volatility = np.sqrt(volatility)
        
        elif method == "garch":
            # Simple GARCH(1,1) model
            volatility = np.zeros_like(y)
            omega = 0.000001  # Constant term
            alpha = 0.1       # ARCH term
            beta = 0.8        # GARCH term
            
            volatility[0] = np.std(returns[:min(window, len(returns))])
            
            for i in range(1, len(y)):
                if i < len(returns):
                    variance = omega + alpha * returns[i-1]**2 + beta * volatility[i-1]**2
                    volatility[i] = np.sqrt(variance)
                else:
                    volatility[i] = volatility[i-1]
        
        else:
            raise ValueError(f"Unknown volatility estimation method: {method}")
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': volatility
            }
        ]
