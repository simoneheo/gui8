import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class threshold_clip_step(BaseStep):
    name = "threshold_clip"
    category = "Arithmetic"
    description = """Clamp signal values to specified minimum and maximum thresholds.
    
This step limits signal values to fall within a specified range by clipping values
that exceed the minimum or maximum thresholds. Values below the minimum are set to
the minimum, and values above the maximum are set to the maximum.

• **Minimum value**: Lower bound for signal values
• **Maximum value**: Upper bound for signal values

Useful for:
• **Range limiting**: Prevent signal values from exceeding physical limits
• **Outlier removal**: Clip extreme values that may be artifacts
• **Signal conditioning**: Prepare signals for systems with specific input ranges
• **Data normalization**: Ensure values fall within expected bounds"""
    tags = ["time-series", "clipping", "range-limiting", "threshold", "bounds", "saturate"]
    params = [
        {
            "name": "min_val", 
            "type": "float", 
            "default": "-1.0", 
            "help": "Minimum clip value (values below this become min_val)"
        },
        {
            "name": "max_val", 
            "type": "float", 
            "default": "1.0", 
            "help": "Maximum clip value (values above this become max_val)"
        },
    ]

    @classmethod
    def get_info(cls): 
        return f"{cls.name} — Clamp values below/above threshold (Category: {cls.category})"

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
        min_val = params.get("min_val")
        max_val = params.get("max_val")
        
        if np.isnan(min_val) or np.isinf(min_val):
            raise ValueError(f"min_val must be a finite number, got {min_val}")
        if np.isnan(max_val) or np.isinf(max_val):
            raise ValueError(f"max_val must be a finite number, got {max_val}")
        if min_val >= max_val:
            raise ValueError(f"min_val must be less than max_val, got min_val={min_val}, max_val={max_val}")

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
                elif param["type"] == "float":
                    parsed[name] = float(val)
                elif param["type"] == "int":
                    parsed[name] = int(val)
                else:
                    parsed[name] = val
            except ValueError as e:
                if "could not convert" in str(e) or "invalid literal" in str(e):
                    raise ValueError(f"{name} must be a valid {param['type']}")
                raise e
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        """Apply threshold clipping to the channel data."""
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
                suffix="ThresholdClip"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Threshold clipping failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic for threshold clipping"""
        min_val = params.get("min_val", -1.0)
        max_val = params.get("max_val", 1.0)
        
        y_new = np.clip(y, min_val, max_val)
        return y_new
