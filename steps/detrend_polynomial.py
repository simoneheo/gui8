import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class detrend_polynomial_step(BaseStep):
    name = "detrend_polynomial"
    category = "Transform"
    description = "Remove polynomial trend"
    tags = ["time-series"]
    params = [
        {"name": "degree", "type": "int", "default": "2", "help": "Degree of polynomial"},
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
        degree = int(params['degree'])
        from numpy.polynomial import Polynomial
        p = Polynomial.fit(x, y, deg=degree)
        trend = p(x)
        y_new = y - trend
        return cls.create_new_channel(parent=channel, xdata=x, ydata=y_new, params=params)
