import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

def median_subtract(y):

    return y - np.median(y)

@register_step
class median_subtract_step(BaseStep):
    name = "median_subtract"
    category = "Arithmetic"
    description = "Subtract median value from signal for robust baseline removal."
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
        
        y_new = median_subtract(channel.ydata)
        x_new = np.linspace(channel.xdata[0], channel.xdata[-1], len(y_new))
        return cls.create_new_channel(parent=channel, xdata=x_new, ydata=y_new, params=params)
