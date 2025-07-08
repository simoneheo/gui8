import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class moving_absmax_step(BaseStep):
    name = "moving_absmax"
    category = "Envelope"
    description = "Compute max absolute value in overlapping sliding windows"
    tags = ["time-series"]
    params = [
        {"name": "window", "type": "int", "default": "25", "help": "Window size in samples"},
        {"name": "overlap", "type": "int", "default": "0", "help": "Overlap between windows in samples"},
    ]

    @classmethod
    def get_info(cls):
        return f"{cls.name} — {cls.description} (Category: {cls.category})"

    @classmethod
    def get_prompt(cls):
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        window = int(user_input.get("window", 25))
        overlap = int(user_input.get("overlap", 0))
        if window <= 0:
            raise ValueError("Window size must be > 0")
        if overlap < 0 or overlap >= window:
            raise ValueError("Overlap must be non-negative and less than window size")
        return {"window": window, "overlap": overlap}

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        y = channel.ydata
        x = channel.xdata
        window = params["window"]
        overlap = params["overlap"]
        step = window - overlap

        absmax_values = []
        time_stamps = []

        for start in range(0, len(y) - window + 1, step):
            end = start + window
            max_val = np.max(np.abs(y[start:end]))
            center_time = x[start + window // 2]
            absmax_values.append(max_val)
            time_stamps.append(center_time)

        return cls.create_new_channel(
            parent=channel,
            xdata=np.array(time_stamps),
            ydata=np.array(absmax_values),
            params=params
        )
