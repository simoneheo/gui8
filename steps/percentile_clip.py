import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class percentile_clip_step(BaseStep):
    name = "percentile_clip"
    category = "General"
    description = """Clip signal values to specified percentile range to remove outliers.
    
This step computes the specified percentiles of the entire signal and clips all values
to fall within that range. Values below the lower percentile are set to the lower bound,
and values above the upper percentile are set to the upper bound.

• **Lower percentile**: Values below this percentile are clipped (0-100)
• **Upper percentile**: Values above this percentile are clipped (0-100)

Useful for:
• **Outlier removal**: Remove extreme values that may be artifacts or noise
• **Data cleaning**: Ensure signal values fall within reasonable bounds
• **Robust analysis**: Reduce the impact of outliers on subsequent processing
• **Signal normalization**: Constrain signal to a specific range"""
    tags = ["time-series", "outlier-removal", "data-cleaning", "percentile", "clipping", "robust"]
    params = [
        {'name': 'lower', 'type': 'float', 'default': '1.0', 'help': 'Lower percentile (0-100)'}, 
        {'name': 'upper', 'type': 'float', 'default': '99.0', 'help': 'Upper percentile (0-100)'}
    ]

    @classmethod
    def get_info(cls): 
        return f"{cls.name} — Clip signal to specified percentile range (Category: {cls.category})"

    @classmethod
    def get_prompt(cls): 
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input signal data"""
        if len(y) == 0:
            raise ValueError("Input signal is empty")
        if np.any(np.isnan(y)):
            raise ValueError("Input signal contains NaN values")
        if np.any(np.isinf(y)):
            raise ValueError("Input signal contains infinite values")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        lower = params.get("lower")
        upper = params.get("upper")
        
        if not (0 <= lower <= 100):
            raise ValueError(f"Lower percentile must be between 0 and 100, got {lower}")
        if not (0 <= upper <= 100):
            raise ValueError(f"Upper percentile must be between 0 and 100, got {upper}")
        if lower >= upper:
            raise ValueError(f"Lower percentile must be less than upper percentile, got lower={lower}, upper={upper}")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output signal data"""
        if len(y_new) != len(y_original):
            raise ValueError("Output signal length differs from input")
        if np.any(np.isnan(y_new)) or np.any(np.isinf(y_new)):
            raise ValueError("Clipping resulted in invalid values")

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
        """Apply percentile clipping to the channel data."""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            
            # Process the data
            y_new = cls.script(x, y, params)
            
            # Validate output data
            cls._validate_output_data(y, y_new)
            
            return cls.create_new_channel(
                parent=channel, 
                xdata=x, 
                ydata=y_new, 
                params=params,
                suffix="PercentileClip"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Percentile clipping failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic for percentile clipping"""
        lower = params.get("lower", 1.0)
        upper = params.get("upper", 99.0)
        
        lo = np.percentile(y, lower)
        hi = np.percentile(y, upper)
        
        if np.isnan(lo) or np.isnan(hi):
            raise ValueError("Failed to compute percentiles")
        if np.isinf(lo) or np.isinf(hi):
            raise ValueError("Computed percentiles are infinite")
            
        y_new = np.clip(y, lo, hi)
        return y_new
