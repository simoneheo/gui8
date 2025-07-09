
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
    def get_info(cls): 
        return f"{cls.name} — Autocorrelation analysis (Category: {cls.category})"

    @classmethod
    def get_prompt(cls): 
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input signal data"""
        if len(y) == 0:
            raise ValueError("Input signal is empty")
        if np.all(np.isnan(y)):
            raise ValueError("Signal contains only NaN values")
        if np.all(np.isinf(y)):
            raise ValueError("Signal contains only infinite values")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        max_lag = params.get("max_lag")
        
        if max_lag is None or max_lag <= 0:
            raise ValueError("Maximum lag must be positive")
        if max_lag > 1000:
            raise ValueError("Maximum lag too large (max 1000)")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output signal data"""
        if len(y_new) == 0:
            raise ValueError("Autocorrelation produced empty output")
        if np.any(np.isnan(y_new)):
            raise ValueError("Autocorrelation produced NaN values")
        if np.any(np.isinf(y_new)):
            raise ValueError("Autocorrelation produced infinite values")

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
                elif param["type"] == "bool":
                    parsed[name] = bool(val)
                else:
                    parsed[name] = val
            except ValueError as e:
                if "could not convert" in str(e) or "invalid literal" in str(e):
                    raise ValueError(f"{name} must be a valid {param['type']}")
                raise e
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        """Apply the processing step to a channel"""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            
            # Process the data
            x_new, y_new = cls.script(x, y, params)
            
            # Validate output data
            cls._validate_output_data(y, y_new)
            
            return cls.create_new_channel(
                parent=channel,
                xdata=x_new,
                ydata=y_new,
                params=params,
                suffix="Autocorr"
            )
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Autocorrelation processing failed: {str(e)}")

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
