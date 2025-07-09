import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class derivative_step(BaseStep):
    name = "derivative"
    category = "Arithmetic"
    description = """Computes the numerical derivative of the input signal.
Supports multiple differentiation methods and orders for flexible signal analysis."""
    tags = ["time-series", "differentiation", "derivative", "gradient", "slope", "rate"]
    params = [
        {
            "name": "method", 
            "type": "str", 
            "default": "gradient", 
            "help": "Numerical differentiation method",
            "options": ["gradient", "diff"]
        },
        {
            "name": "order", 
            "type": "int", 
            "default": "1", 
            "help": "Order of derivative (1=first, 2=second, etc.)"
        }
    ]

    @classmethod
    def get_info(cls):
        return f"{cls.name} — {cls.description} (Category: {cls.category})"

    @classmethod
    def get_prompt(cls):
        return {
            "info": cls.description,
            "params": cls.params
        }

    @classmethod
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input signal data"""
        if len(y) == 0:
            raise ValueError("Input signal is empty")
        if np.all(np.isnan(y)):
            raise ValueError("Signal contains only NaN values")
        if np.all(np.isinf(y)):
            raise ValueError("Signal contains only infinite values")
        if len(y) < 2:
            raise ValueError("Signal must have at least 2 samples for derivative computation")

    @classmethod
    def _validate_parameters(cls, params: dict, total_samples: int) -> None:
        """Validate parameters and business rules"""
        method = params.get("method")
        order = params.get("order")
        
        if method not in ["gradient", "diff"]:
            raise ValueError("Method must be 'gradient' or 'diff'")
        
        if order is None or order < 1:
            raise ValueError("Derivative order must be at least 1")
        if order > 5:
            raise ValueError("Derivative order too high (maximum 5)")
        if total_samples <= order:
            raise ValueError(f"Signal length ({total_samples}) must be greater than derivative order ({order})")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output signal data"""
        if len(y_new) != len(y_original):
            raise ValueError("Output signal length differs from input")
        if np.any(np.isnan(y_new)) and not np.any(np.isnan(y_original)):
            raise ValueError("Derivative computation produced unexpected NaN values")
        if np.any(np.isinf(y_new)) and not np.any(np.isinf(y_original)):
            raise ValueError("Derivative computation produced unexpected infinite values")

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        """Parse and validate user input parameters"""
        parsed = {}
        for param in cls.params:
            name = param["name"]
            val = user_input.get(name, param.get("default"))
            try:
                if val == "":
                    parsed[name] = None
                elif param["type"] == "int":
                    parsed[name] = int(val)
                elif param["type"] == "float":
                    parsed[name] = float(val)
                else:
                    parsed[name] = val
            except ValueError as e:
                if "could not convert" in str(e) or "invalid literal" in str(e):
                    raise ValueError(f"{name} must be a valid {param['type']}")
                raise e
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        """Apply derivative computation to the channel data."""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params, len(y))
            
            # Process the data
            y_new = cls.script(x, y, params)
            
            # Validate output data
            cls._validate_output_data(y, y_new)
            
            return cls.create_new_channel(
                parent=channel,
                xdata=x,
                ydata=y_new,
                params=params,
                suffix="Derivative"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Derivative computation failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic for derivative computation"""
        method = params["method"]
        order = params["order"]
        
        y_new = y.copy()
        for _ in range(order):
            if method == 'gradient':
                y_new = np.gradient(y_new)
            elif method == 'diff':
                y_new = np.diff(y_new)
                # Pad to maintain length
                y_new = np.concatenate([y_new, [y_new[-1]]])
            else:
                raise ValueError(f"Unknown method: {method}")
        
        return y_new
