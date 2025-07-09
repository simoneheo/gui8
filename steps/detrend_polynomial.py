import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class detrend_polynomial_step(BaseStep):
    name = "detrend_polynomial"
    category = "Transform"
    description = """Remove polynomial trend from the signal.
Fits a polynomial of specified degree to the signal and subtracts it to remove long-term trends."""
    tags = ["time-series", "detrend", "polynomial", "polyfit", "baseline", "trend", "remove"]
    params = [
        {
            "name": "degree", 
            "type": "int", 
            "default": "2", 
            "help": "Degree of polynomial to fit and remove (1=linear, 2=quadratic, etc.)"
        }
    ]

    @classmethod
    def get_info(cls): 
        return f"{cls.name} — {cls.description} (Category: {cls.category})"

    @classmethod
    def get_prompt(cls): 
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def _validate_input_data(cls, x: np.ndarray, y: np.ndarray) -> None:
        """Validate input signal data"""
        if len(x) == 0 or len(y) == 0:
            raise ValueError("Input signal is empty")
        if len(x) != len(y):
            raise ValueError(f"Time and signal data length mismatch: {len(x)} vs {len(y)}")
        if len(x) < 2:
            raise ValueError("Signal too short for polynomial detrending (minimum 2 samples)")
        if np.any(np.isnan(x)):
            raise ValueError("Time data contains NaN values")
        if np.any(np.isinf(x)):
            raise ValueError("Time data contains infinite values")
        if not np.all(np.diff(x) > 0):
            raise ValueError("Time data must be monotonically increasing")

    @classmethod
    def _validate_parameters(cls, params: dict, total_samples: int) -> None:
        """Validate parameters and business rules"""
        degree = params.get("degree")
        
        if degree is None or degree < 0:
            raise ValueError("Polynomial degree must be non-negative")
        if degree >= total_samples:
            raise ValueError(f"Polynomial degree ({degree}) must be less than signal length ({total_samples})")
        if degree > 10:
            raise ValueError(f"Polynomial degree too high ({degree}). Maximum allowed is 10")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output signal data"""
        if len(y_new) != len(y_original):
            raise ValueError("Output signal length differs from input")
        if np.any(np.isnan(y_new)) and not np.any(np.isnan(y_original)):
            raise ValueError("Polynomial detrending produced unexpected NaN values")
        if np.any(np.isinf(y_new)) and not np.any(np.isinf(y_original)):
            raise ValueError("Polynomial detrending produced unexpected infinite values")

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
        """Apply polynomial detrending to the channel data."""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Validate input data and parameters
            cls._validate_input_data(x, y)
            cls._validate_parameters(params, len(x))
            
            # Process the data
            y_new = cls.script(x, y, params)
            
            # Validate output data
            cls._validate_output_data(y, y_new)
            
            return cls.create_new_channel(
                parent=channel, 
                xdata=x, 
                ydata=y_new, 
                params=params,
                suffix="Detrend"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Polynomial detrending failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic for polynomial detrending"""
        from numpy.polynomial import Polynomial
        
        degree = params["degree"]
        p = Polynomial.fit(x, y, deg=degree)
        trend = p(x)
        y_new = y - trend
        
        return y_new
