import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

def moving_average_windowed(y, window=5, overlap=0):
    if window < 1:
        raise ValueError(f"Window size must be >= 1, got {window}")
    if overlap < 0 or overlap >= window:
        raise ValueError(f"Overlap must be in [0, window), got {overlap}")
    if len(y) < window:
        raise ValueError(f"Signal too short: requires at least {window} samples")

    step = window - overlap
    result = np.zeros_like(y, dtype=float)
    count = np.zeros_like(y, dtype=int)

    for start in range(0, len(y) - window + 1, step):
        end = start + window
        segment = y[start:end]
        avg = np.mean(segment)
        result[start:end] += avg
        count[start:end] += 1

    # Avoid divide by zero
    count[count == 0] = 1
    return result / count

@register_step
class moving_average_step(BaseStep):
    name = "moving_average"
    category = "Smoother"
    description = "Applies a moving average smoother using overlapping windows."
    tags = ["time-series"]
    params = [
        {'name': 'window', 'type': 'int', 'default': '5', 'help': 'Window size in samples (must be >= 1)'},
        {'name': 'overlap', 'type': 'int', 'default': '0', 'help': 'Overlap between windows in samples (must be < window)'},
    ]

    @classmethod
    def get_info(cls): return f"{cls.name} — {cls.description} (Category: {cls.category})"
    
    @classmethod
    def get_prompt(cls): return {"info": cls.description, "params": cls.params}

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        window = int(user_input.get("window", 5))
        overlap = int(user_input.get("overlap", 0))
        if window < 1:
            raise ValueError("Window size must be at least 1")
        if overlap < 0 or overlap >= window:
            raise ValueError("Overlap must be non-negative and smaller than window size")
        return {"window": window, "overlap": overlap}

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        if len(channel.ydata) == 0:
            raise ValueError("Input signal is empty")
        if np.all(np.isnan(channel.ydata)):
            raise ValueError("Signal contains only NaN values")

        y_new = moving_average_windowed(channel.ydata, **params)
        x_new = channel.xdata  # Keep original time axis
        return cls.create_new_channel(parent=channel, xdata=x_new, ydata=y_new, params=params)
