import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class rank_transform_step(BaseStep):
    name = "rank_transform"
    category = "Transform"
    description = "Replace each value with its rank (0-based indexing)"
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
            raise ValueError("Input signal contains NaN values (cannot rank NaN values)")
        if np.any(np.isinf(channel.ydata)):
            raise ValueError("Input signal contains infinite values")
        
        y = channel.ydata
        x = channel.xdata
        
        try:
            # Compute ranks using argsort of argsort
            # This gives 0-based ranks where the smallest value gets rank 0
            y_new = np.argsort(np.argsort(y)).astype(float)
            
            # Validate result
            if np.any(np.isnan(y_new)) or np.any(np.isinf(y_new)):
                raise ValueError("Rank computation resulted in invalid values")
            
            # Check that ranks are in expected range
            if len(y) > 0:
                if np.min(y_new) < 0 or np.max(y_new) >= len(y):
                    raise ValueError(f"Computed ranks out of expected range [0, {len(y)-1}]")
            
            return cls.create_new_channel(parent=channel, xdata=x, ydata=y_new, params=params)
            
        except Exception as e:
            raise ValueError(f"Rank transform failed: {str(e)}")
