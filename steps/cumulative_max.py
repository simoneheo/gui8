import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class cumulative_max_step(BaseStep):
    name = "windowed_max"
    category = "Transform"
    description = (
        "Compute the local maximum of the signal using a sliding window with optional overlap. "
        "This is a localized version of the cumulative maximum and is useful for envelope tracking."
    )
    tags = ["time-series"]
    params = [
        {
            "name": "window",
            "type": "int",
            "default": "100",
            "help": "Window size in samples (must be > 0)"
        },
        {
            "name": "overlap",
            "type": "int",
            "default": "50",
            "help": "Overlap between windows in samples (must be < window)"
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

        y_new = np.zeros_like(y)
        counts = np.zeros_like(y)

        for start in range(0, len(y) - window + 1, step):
            end = start + window
            max_val = np.max(y[start:end])
            y_new[start:end] += max_val
            counts[start:end] += 1

        # Handle overlapping regions by averaging
        counts[counts == 0] = 1
        y_final = y_new / counts

        return cls.create_new_channel(parent=channel, xdata=x, ydata=y_final, params=params)
