import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

def minmax_normalize(y, range_min=0.0, range_max=1.0):
    ymin, ymax = np.min(y), np.max(y)
    if ymax - ymin == 0:
        raise ValueError("Signal has zero dynamic range. Cannot normalize.")
    norm = (y - ymin) / (ymax - ymin)
    return norm * (range_max - range_min) + range_min

@register_step
class minmax_normalize_step(BaseStep):
    name = "minmax_normalize"
    category = "General"
    description = "Normalize signal to a specific range using overlapping windows."
    tags = ["time-series"]
    params = [
        {"name": "range_min", "type": "float", "default": "0.0", "help": "Target minimum value of normalized signal"},
        {"name": "range_max", "type": "float", "default": "1.0", "help": "Target maximum value of normalized signal"},
        {"name": "window", "type": "int", "default": "100", "help": "Window size in samples"},
        {"name": "overlap", "type": "int", "default": "50", "help": "Overlap in samples (must be < window)"}
    ]

    @classmethod
    def get_info(cls): return f"{cls.name} — {cls.description} (Category: {cls.category})"
    
    @classmethod
    def get_prompt(cls): return {"info": cls.description, "params": cls.params}
    
    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        parsed = {}
        range_min = float(user_input.get("range_min", 0.0))
        range_max = float(user_input.get("range_max", 1.0))
        if range_min >= range_max:
            raise ValueError("range_max must be greater than range_min")

        window = int(user_input.get("window", 100))
        overlap = int(user_input.get("overlap", 50))
        if window <= 0:
            raise ValueError("Window size must be positive")
        if overlap < 0 or overlap >= window:
            raise ValueError("Overlap must be non-negative and less than window size")

        parsed["range_min"] = range_min
        parsed["range_max"] = range_max
        parsed["window"] = window
        parsed["overlap"] = overlap
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        y = channel.ydata
        x = channel.xdata
        range_min = params["range_min"]
        range_max = params["range_max"]
        window = params["window"]
        overlap = params["overlap"]
        step = window - overlap

        if len(y) < window:
            raise ValueError("Signal is shorter than window size")

        y_out = np.zeros_like(y, dtype=float)
        count = np.zeros_like(y, dtype=int)

        for start in range(0, len(y) - window + 1, step):
            end = start + window
            segment = y[start:end]
            try:
                norm_segment = minmax_normalize(segment, range_min, range_max)
            except ValueError:
                norm_segment = np.zeros_like(segment)
            y_out[start:end] += norm_segment
            count[start:end] += 1

        count[count == 0] = 1  # Prevent divide by zero
