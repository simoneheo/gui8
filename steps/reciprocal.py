import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class reciprocal_step(BaseStep):
    name = "reciprocal"
    category = "Arithmetic"
    description = "Replace each y with 1/y (zeros become zeros)"
    tags = ["time-series"]
    params = [
    ]

    @classmethod
    def get_info(cls): return f"{cls.name} — {cls.description} (Category: {cls.category})"

    @classmethod
    def get_prompt(cls): return {"info": cls.description, "params": cls.params}

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        parsed = {}
        for param in cls.params:
            value = user_input.get(param["name"], param["default"])
            parsed[param["name"]] = float(value) if param["type"] == "float" else int(value)
        return parsed

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
        
        try:
            # Compute reciprocal, handling zeros safely
            y_new = np.where(y != 0, 1.0 / y, 0.0)
            
            # Validate result - check for new infinities that might arise from very small values
            if np.any(np.isinf(y_new)):
                # Handle very small values that create infinities
                very_small_mask = np.abs(y) < 1e-15
                if np.any(very_small_mask & (y != 0)):
                    raise ValueError("Signal contains values too small for reciprocal computation")
            
            if np.any(np.isnan(y_new)):
                raise ValueError("Reciprocal computation resulted in NaN values")
            
            return cls.create_new_channel(parent=channel, xdata=x, ydata=y_new, params=params)
            
        except Exception as e:
            raise ValueError(f"Reciprocal computation failed: {str(e)}")
