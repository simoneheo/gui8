import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

def rolling_mean_subtract(y, window_size=51):

    if window_size < 3:
        raise ValueError("Window size must be >= 3")
    if window_size % 2 == 0:
        window_size += 1  # Make odd for symmetry
    from scipy.ndimage import uniform_filter1d
    baseline = uniform_filter1d(y, size=window_size, mode='nearest')
    return y - baseline

@register_step
class rolling_mean_subtract_step(BaseStep):
    name = "rolling_mean_subtract"
    category = "Arithmetic"
    description = "Subtract rolling mean for local baseline correction."
    tags = ["time-series"]
    params = [{'name': 'window_size', 'type': 'int', 'default': '51', 'help': 'Window size for rolling mean'}]

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
        
        y_new = rolling_mean_subtract(channel.ydata)
        x_new = np.linspace(channel.xdata[0], channel.xdata[-1], len(y_new))
        return cls.create_new_channel(parent=channel, xdata=x_new, ydata=y_new, params=params)
