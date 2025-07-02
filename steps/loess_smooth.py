import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel
from statsmodels.nonparametric.smoothers_lowess import lowess

def loess_smooth(y, frac=0.1):
    if not (0 < frac < 1):
        raise ValueError(f"'frac' must be between 0 and 1, got {frac}")
    x = np.arange(len(y))
    result = lowess(y, x, frac=frac, return_sorted=False)
    return result

@register_step
class loess_smooth_step(BaseStep):
    name = "loess_smooth"
    category = "Smoother"
    description = "Applies LOESS (Locally Weighted Smoothing) to the signal."
    tags = ["time-series"]
    params = [
        {"name": "frac", "type": "float", "default": "0.1", "help": "Fraction of data used for local smoothing (0 < frac < 1)"}
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
            parsed[param["name"]] = float(value)
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        y_new = loess_smooth(channel.ydata, **params)
        x_new = np.linspace(channel.xdata[0], channel.xdata[-1], len(y_new))
        return cls.create_new_channel(parent=channel, xdata=x_new, ydata=y_new, params=params)
