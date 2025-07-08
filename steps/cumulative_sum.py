import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class cumulative_sum_step(BaseStep):
    name = "cumulative_sum"
    category = "Transform"
    description = (
        "Compute local sum of values within sliding windows, with optional overlap. "
        "Useful for extracting local energy or density patterns."
    )
    tags = ["time-series"]
    params = [
        {
            "name": "window",
            "type": "int",
            "default": "100",
            "help": "Window size in number of samples"
        },
        {
            "name": "overlap",
            "type": "int",
            "default": "50",
            "help": "Overlap between windows in samples"
        }
    ]

    @classmethod
    def get_info(cls):
        return f"{cls.name} — {cls.description} (Category: {cls.category})"

    @classmethod
    def get_prompt(cls):
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        window = int(user_input.get("window", 100))
        overlap = int(user_input.get("overlap", 50))
        if window <= 0:
            raise ValueError("Window size must be positive")
        if overlap < 0 or overlap >= window:
            raise ValueError("Overlap must be non-negative and smaller than window size")
        return {"window": window, "overlap": overlap}

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        y = channel.ydata
        x = channel.xdata
        window = params["window"]
        overlap = params["overlap"]
        step = window - overlap

        x_out = []
        y_out = []

        for start in range(0, len(y) - window + 1, step):
            end = start + window
            window_sum = np.sum(y[start:end])
            center_x = x[start + window // 2]
            x_out.append(center_x)
            y_out.append(window_sum)

        return cls.create_new_channel(
            parent=channel,
            xdata=np.array(x_out),
            ydata=np.array(y_out),
            params=params
        )
