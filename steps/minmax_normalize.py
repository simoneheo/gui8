import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

def minmax_normalize(y, range_min=0.0, range_max=1.0):
    ymin, ymax = np.min(y), np.max(y)
    if ymax - ymin == 0:
        raise ValueError("Signal has zero dynamic range. Cannot normalize.")
    norm = (y - ymin) / (ymax - ymin)
    return norm * (range_max - range_min) + range_min

@register_step
class minmax_normalize_step(BaseStep):
    name = "minmax_normalize"
    category = "General"
    description = "Normalize signal to a specific range."
    tags = ["time-series"]
    params = [
        {
            'name': 'range_min', 
            'type': 'float', 
            'default': '0.0', 
            'help': 'Target minimum value of normalized signal',
   
        }, 
        {
            'name': 'range_max', 
            'type': 'float', 
            'default': '1.0', 
            'help': 'Target maximum value of normalized signal',
            
        }
    ]

    @classmethod
    def get_info(cls): return f"{cls.name} — {cls.description} (Category: {cls.category})"
    @classmethod
    def get_prompt(cls): return {"info": cls.description, "params": cls.params}
    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        parsed = {}
        
        try:
            range_min = float(user_input.get("range_min", "0.0"))
            range_max = float(user_input.get("range_max", "1.0"))
            
            if range_min >= range_max:
                raise ValueError("range_max must be greater than range_min")
            
            parsed["range_min"] = range_min
            parsed["range_max"] = range_max
            
        except ValueError as e:
            if "could not convert" in str(e):
                raise ValueError("Range values must be valid numbers")
            raise e
            
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        # Validate input data
        if len(channel.ydata) == 0:
            raise ValueError("Input signal is empty")
        if np.all(np.isnan(channel.ydata)):
            raise ValueError("Signal contains only NaN values")
        if np.all(np.isinf(channel.ydata)):
            raise ValueError("Signal contains only infinite values")
            
        try:
            y_new = minmax_normalize(channel.ydata, **params)
            x_new = np.linspace(channel.xdata[0], channel.xdata[-1], len(y_new))
            return cls.create_new_channel(parent=channel, xdata=x_new, ydata=y_new, params=params)
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Normalization failed: {str(e)}")
