import numpy as np
from scipy.ndimage import uniform_filter1d
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class smooth_normalize_step(BaseStep):
    name = "smooth_normalize"
    category = "Transform"
    description = "Normalize signal using local mean and range with uniform_filter1d"
    tags = ["time-series"]
    params = [
        {"name": "window", "type": "int", "default": "101", "help": "Window size for smoothing (odd number)"},
        {"name": "scale_0_1", "type": "bool", "default": "True", "help": "Rescale normalized signal to [0,1]"}
    ]

    @classmethod
    def get_info(cls):
        return f"{cls.name} — {cls.description} (Category: {cls.category})"

    @classmethod
    def get_prompt(cls):
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        parsed = {}
        window = int(user_input.get("window", 101))
        if window < 3:
            raise ValueError("Window size must be at least 3")
        if window % 2 == 0:
            window += 1  # Force odd for symmetry

        scale_0_1 = user_input.get("scale_0_1", "True")
        if isinstance(scale_0_1, str):
            scale_0_1 = scale_0_1.lower() in ["true", "1", "yes"]

        parsed["window"] = window
        parsed["scale_0_1"] = scale_0_1
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        y = channel.ydata
        x = channel.xdata
        window = params["window"]
        scale_0_1 = params["scale_0_1"]

        if len(y) < window:
            raise ValueError("Signal shorter than window size")

        # Smooth local mean and absolute deviation
        local_mean = uniform_filter1d(y, size=window, mode='nearest')
        local_range = uniform_filter1d(np.abs(y - local_mean), size=window, mode='nearest')

        # Avoid divide-by-zero
        local_range[local_range == 0] = 1e-8

        y_norm = (y - local_mean) / local_range

        if scale_0_1:
            y_min, y_max = np.min(y_norm), np.max(y_norm)
            if y_max - y_min == 0:
                y_norm[:] = 0
            else:
                y_norm = (y_norm - y_min) / (y_max - y_min)

        x_new = np.linspace(x[0], x[-1], len(y_norm))
        return cls.create_new_channel(parent=channel, xdata=x_new, ydata=y_norm, params=params)
