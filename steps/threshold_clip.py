import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class threshold_clip_step(BaseStep):
    name = "threshold_clip"
    category = "Arithmetic"
    description = "Clamp values below/above threshold"
    tags = ["time-series"]
    params = [
        {"name": "min_val", "type": "float", "default": "-1.0", "help": "Minimum clip value"},
        {"name": "max_val", "type": "float", "default": "1.0", "help": "Maximum clip value"},
    ]

    @classmethod
    def get_info(cls): return f"{cls.name} — {cls.description} (Category: {cls.category})"

    @classmethod
    def get_prompt(cls): return {"info": cls.description, "params": cls.params}

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        try:
            parsed = {}
            for param in cls.params:
                value = user_input.get(param["name"], param["default"])
                parsed[param["name"]] = float(value) if param["type"] == "float" else int(value)
            
            # Parameter validation
            min_val = parsed["min_val"]
            max_val = parsed["max_val"]
            
            if np.isnan(min_val) or np.isinf(min_val):
                raise ValueError(f"min_val must be a finite number, got {min_val}")
            if np.isnan(max_val) or np.isinf(max_val):
                raise ValueError(f"max_val must be a finite number, got {max_val}")
            if min_val >= max_val:
                raise ValueError(f"min_val must be less than max_val, got min_val={min_val}, max_val={max_val}")
            
            return parsed
        except ValueError as e:
            raise ValueError(f"Parameter validation failed: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to parse input parameters: {str(e)}")

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        # Input validation
        if channel is None:
            raise ValueError("Channel is None")
        if channel.ydata is None or len(channel.ydata) == 0:
            raise ValueError("Channel has no data")
        if channel.xdata is None or len(channel.xdata) == 0:
            raise ValueError("Channel has no x-axis data")
        if len(channel.xdata) != len(channel.ydata):
            raise ValueError("X and Y data lengths don't match")
        
        # Check for NaN values
        if np.any(np.isnan(channel.ydata)):
            raise ValueError("Input signal contains NaN values")
        if np.any(np.isinf(channel.ydata)):
            raise ValueError("Input signal contains infinite values")
        
        y = channel.ydata
        x = channel.xdata
        min_val = params['min_val']
        max_val = params['max_val']
        
        try:
            y_new = np.clip(y, min_val, max_val)
            
            # Validate result
            if np.any(np.isnan(y_new)) or np.any(np.isinf(y_new)):
                raise ValueError("Clipping resulted in invalid values")
            
            return cls.create_new_channel(parent=channel, xdata=x, ydata=y_new, params=params)
            
        except Exception as e:
            raise ValueError(f"Threshold clipping failed: {str(e)}")
