import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

def percentile_clip(y, lower=1.0, upper=99.0):
    # Input validation
    if len(y) == 0:
        raise ValueError("Input signal is empty")
    
    # Check for NaN values
    if np.any(np.isnan(y)):
        raise ValueError("Input signal contains NaN values")
    if np.any(np.isinf(y)):
        raise ValueError("Input signal contains infinite values")
    
    # Parameter validation
    if not (0 <= lower < upper <= 100):
        raise ValueError(f"Invalid percentile range: lower={lower}, upper={upper}. Must have 0 ≤ lower < upper ≤ 100")
    
    try:
        lo = np.percentile(y, lower)
        hi = np.percentile(y, upper)
        
        if np.isnan(lo) or np.isnan(hi):
            raise ValueError("Failed to compute percentiles")
        if np.isinf(lo) or np.isinf(hi):
            raise ValueError("Computed percentiles are infinite")
        
        result = np.clip(y, lo, hi)
        
        # Validate result
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            raise ValueError("Clipping resulted in invalid values")
        
        return result
        
    except Exception as e:
        raise ValueError(f"Percentile clipping failed: {str(e)}")

@register_step
class percentile_clip_step(BaseStep):
    name = "percentile_clip"
    category = "General"
    description = "Clip signal to specified percentile range."
    tags = ["time-series"]
    params = [{'name': 'lower', 'type': 'float', 'default': '1.0', 'help': 'Lower percentile (0-100)'}, 
    {'name': 'upper', 'type': 'float', 'default': '99.0', 'help': 'Upper percentile (0-100)'}]

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
            lower = parsed.get("lower", 1.0)
            upper = parsed.get("upper", 99.0)
            
            if not (0 <= lower <= 100):
                raise ValueError(f"Lower percentile must be between 0 and 100, got {lower}")
            if not (0 <= upper <= 100):
                raise ValueError(f"Upper percentile must be between 0 and 100, got {upper}")
            if lower >= upper:
                raise ValueError(f"Lower percentile must be less than upper percentile, got lower={lower}, upper={upper}")
            
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
            lower = params.get("lower", 1.0)
            upper = params.get("upper", 99.0)
            
            # Apply percentile clipping
            y_new = percentile_clip(channel.ydata, lower=lower, upper=upper)
            x_new = np.linspace(channel.xdata[0], channel.xdata[-1], len(y_new))
            return cls.create_new_channel(parent=channel, xdata=x_new, ydata=y_new, params=params)
            
        except Exception as e:
            raise ValueError(f"Percentile clipping step failed: {str(e)}")
