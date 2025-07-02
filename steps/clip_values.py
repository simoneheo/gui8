import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

def clip_values(y, min_val=-3.0, max_val=3.0):
    return np.clip(y, min_val, max_val)

@register_step
class clip_values_step(BaseStep):
    name = "clip_values"
    category = "General"
    description = "Clip signal values to a specified range."
    tags = ["time-series"]
    params = [
        {'name': 'min_val', 'type': 'float', 'default': '-3.0', 'help': 'Minimum allowed value'}, 
    {'name': 'max_val', 'type': 'float', 'default': '3.0', 'help': 'Maximum allowed value'}
    ]

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
        
        y_new = clip_values(channel.ydata, params.get('min_val', -3.0), params.get('max_val', 3.0))
        x_new = np.linspace(channel.xdata[0], channel.xdata[-1], len(y_new))
        return cls.create_new_channel(parent=channel, xdata=x_new, ydata=y_new, params=params)
