import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

def log_transform(y, offset=1.0):
    # Input validation
    if len(y) == 0:
        raise ValueError("Input signal is empty")
    
    # Check for NaN values
    if np.any(np.isnan(y)):
        raise ValueError("Input signal contains NaN values")
    if np.any(np.isinf(y)):
        raise ValueError("Input signal contains infinite values")
    
    # Parameter validation
    if np.isnan(offset) or np.isinf(offset):
        raise ValueError(f"Offset must be a finite number, got {offset}")
    
    try:
        y_shifted = y + offset
        
        if np.any(y_shifted <= 0):
            min_val = np.min(y)
            raise ValueError(f"Log transform undefined for non-positive values. Signal minimum: {min_val}, offset: {offset}. Try offset > {-min_val}")
        
        result = np.log(y_shifted)
        
        # Validate result
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            raise ValueError("Log transform resulted in invalid values")
        
        return result
        
    except Exception as e:
        raise ValueError(f"Log transform failed: {str(e)}")

@register_step
class log_transform_step(BaseStep):
    name = "log_transform"
    category = "General"
    description = "Apply natural log to positive values for compression."
    tags = ["time-series"]
    params = [{'name': 'offset', 'type': 'float', 'default': '1.0', 'help': 'Offset added to signal before log transform'}]

    @classmethod
    def get_info(cls): return f"{cls.name} — {cls.description} (Category: {cls.category})"
    @classmethod
    def get_prompt(cls): return {"info": cls.description, "params": cls.params}
    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        try:
            parsed = {}
            for param in cls.params:
                if param["name"] == "fs": continue
                value = user_input.get(param["name"], param.get("default"))
                parsed[param["name"]] = float(value) if param["type"] == "float" else value
            
            # Parameter validation
            offset = parsed.get("offset", 1.0)
            
            if np.isnan(offset) or np.isinf(offset):
                raise ValueError(f"Offset must be a finite number, got {offset}")
            
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
        
        try:
            # Extract parameters
            offset = params.get("offset", 1.0)
            
            # Apply log transform
            y_new = log_transform(channel.ydata, offset=offset)
            x_new = np.linspace(channel.xdata[0], channel.xdata[-1], len(y_new))
            return cls.create_new_channel(parent=channel, xdata=x_new, ydata=y_new, params=params)
            
        except Exception as e:
            raise ValueError(f"Log transform step failed: {str(e)}")
