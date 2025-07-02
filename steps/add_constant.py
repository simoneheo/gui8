import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class add_constant_step(BaseStep):
    name = "add_constant"
    category = "Arithmetic"
    description = "Add fixed value to all points"
    tags = ["time-series"]
    params = [
        {"name": "value", "type": "float", "default": "1.0", "help": "Value to add"},
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
        value = params['value']
        y_new = y + value
        return cls.create_new_channel(parent=channel, xdata=x, ydata=y_new, params=params)
