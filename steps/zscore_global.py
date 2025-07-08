import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

def zscore_global(y):
    # Input validation
    if len(y) == 0:
        raise ValueError("Input signal is empty")
    
    # Check for NaN values
    if np.any(np.isnan(y)):
        raise ValueError("Input signal contains NaN values")
    
    # Check for infinite values
    if np.any(np.isinf(y)):
        raise ValueError("Input signal contains infinite values")
    
    mean = np.mean(y)
    std = np.std(y)
    
    if std == 0:
        raise ValueError("Standard deviation is zero. Cannot normalize constant signal.")
    
    if np.isnan(std) or np.isinf(std):
        raise ValueError("Failed to compute standard deviation")
        
    try:
        result = (y - mean) / std
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            raise ValueError("Normalization resulted in invalid values")
        return result
    except Exception as e:
        raise ValueError(f"Z-score normalization failed: {str(e)}")

@register_step
class zscore_normalize_step(BaseStep):
    name = "zscore_global"
    category = "General"
    description = "Standardize signal to mean 0 and std 1."
    tags = ["time-series"]
    params = []

    @classmethod
    def get_info(cls): return f"{cls.name} — {cls.description} (Category: {cls.category})"
    @classmethod
    def get_prompt(cls): return {"info": cls.description, "params": cls.params}
    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        parsed = {}
        for param in cls.params:
            if param["name"] == "fs": continue
            value = user_input.get(param["name"], param.get("default"))
            parsed[param["name"]] = float(value) if param["type"] == "float" else value
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
        
        try:
            y_new = zscore_normalize(channel.ydata)
            x_new = np.linspace(channel.xdata[0], channel.xdata[-1], len(y_new))
            return cls.create_new_channel(parent=channel, xdata=x_new, ydata=y_new, params=params)
        except Exception as e:
            raise ValueError(f"Z-score normalization step failed: {str(e)}")
