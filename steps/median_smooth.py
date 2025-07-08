import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel
from scipy.signal import medfilt

def median_smooth_windowed(y, window_size=5, overlap=0):
    """
    Apply median filter in overlapping sliding windows.
    Each point may be part of multiple windows; results are averaged.
    """
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError(f"Window size must be odd and >= 3, got {window_size}")
    if overlap < 0 or overlap >= window_size:
        raise ValueError(f"Overlap must be in [0, window_size), got {overlap}")
    if len(y) < window_size:
        raise ValueError(f"Signal too short: length {len(y)} < window size {window_size}")

    step = window_size - overlap
    smoothed = np.zeros_like(y, dtype=float)
    counts = np.zeros_like(y, dtype=int)

    for start in range(0, len(y) - window_size + 1, step):
        end = start + window_size
        chunk = y[start:end]
        filtered = medfilt(chunk, kernel_size=window_size)
        smoothed[start:end] += filtered
        counts[start:end] += 1

    # Avoid divide by zero
    counts[counts == 0] = 1
    return smoothed / counts

@register_step
class median_smooth_step(BaseStep):
    name = "median_smooth"
    category = "Smoother"
    description = "Applies a median filter in overlapping sliding windows."
    tags = ["time-series"]
    params = [
        {"name": "window_size", "type": "int", "default": "5", "help": "Window size (odd integer >= 3)"},
        {"name": "overlap", "type": "int", "default": "2", "help": "Overlap in samples (must be less than window size)"}
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
            parsed[param["name"]] = int(value)
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        y_new = median_smooth_windowed(channel.ydata, **params)
        x_new = channel.xdata  # Keep original time axis
        return cls.create_new_channel(parent=channel, xdata=x_new, ydata=y_new, params=params)
