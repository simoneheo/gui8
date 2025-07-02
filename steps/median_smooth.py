import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

def median_smooth(y, window_size=5):
    from scipy.signal import medfilt
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError(f"Window size must be odd and >= 3, got {window_size}")
    if len(y) < window_size:
        raise ValueError(f"Signal too short for smoothing: requires signal length > {window_size}, got {len(y)}")
    return medfilt(y, kernel_size=window_size)

@register_step
class median_smooth_step(BaseStep):
    name = "median_smooth"
    category = "Smoother"
    description = "Applies a median filter for smoothing."
    tags = ["time-series"]
    params = [
        {"name": "window_size", "type": "int", "default": "5", "help": "Window size (odd integer >= 3)"}
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
            parsed[param["name"]] = int(value)
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        y_new = median_smooth(channel.ydata, **params)
        x_new = np.linspace(channel.xdata[0], channel.xdata[-1], len(y_new))
        return cls.create_new_channel(parent=channel, xdata=x_new, ydata=y_new, params=params)
