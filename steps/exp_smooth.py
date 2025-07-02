import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel


def exp_smooth(y, alpha=0.2):
    if not (0 < alpha <= 1):
        raise ValueError(f"Alpha must be in (0, 1], got {alpha}")
    result = np.zeros_like(y)
    result[0] = y[0]
    for i in range(1, len(y)):
        result[i] = alpha * y[i] + (1 - alpha) * result[i - 1]
    return result


@register_step
class exp_smooth_step(BaseStep):
    name = "exp_smooth"
    category = "Smoother"
    description = "Applies exponential smoothing to the signal."
    tags = ["time-series"]
    params = [{'name': 'alpha', 'type': 'float', 'default': '0.2', 'help': 'Smoothing factor (0 < alpha ≤ 1)'}]

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
        y_new = exp_smooth(channel.ydata, **params)
        x_new = np.linspace(channel.xdata[0], channel.xdata[-1], len(y_new))
        return cls.create_new_channel(parent=channel, xdata=x_new, ydata=y_new, params=params)
