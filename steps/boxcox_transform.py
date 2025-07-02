import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

def boxcox_transform(y, lmbda=0.0):
    from scipy.stats import boxcox
    y_shifted = y - np.min(y) + 1e-4
    try:
        return boxcox(y_shifted, lmbda=lmbda)
    except ValueError as e:
        raise ValueError(f"Box-Cox failed: {str(e)}")

@register_step
class boxcox_transform_step(BaseStep):
    name = "boxcox_transform"
    category = "General"
    description = "Apply Box-Cox transform to make data more normal-like."
    tags = ["time-series"]
    params = [
        {
            'name': 'lmbda', 
            'type': 'float', 
            'default': '0.0', 
            'help': 'Lambda parameter: 0=log transform, 0.5=square root, 1=no transform, -1=reciprocal',
            'options': ['0.0', '0.5', '1.0', '-0.5', '-1.0', '2.0']
        }
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
        y_new = boxcox_transform(channel.ydata, **params)
        x_new = np.linspace(channel.xdata[0], channel.xdata[-1], len(y_new))
        return cls.create_new_channel(parent=channel, xdata=x_new, ydata=y_new, params=params)
