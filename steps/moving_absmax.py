import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class moving_absmax_step(BaseStep):
    name = "moving_absmax"
    category = "Envelope"
    description = "Max absolute value in sliding window"
    tags = ["time-series"]
    params = [
        {"name": "window", "type": "int", "default": "25", "help": "Sliding window size (samples)"},
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
        y = channel.ydata
        x = channel.xdata
        window = int(params['window'])
        y_new = np.array([np.max(np.abs(y[max(0,i-window//2):i+window//2+1])) for i in range(len(y))])
        return cls.create_new_channel(parent=channel, xdata=x, ydata=y_new, params=params)
