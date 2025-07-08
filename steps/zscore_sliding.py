import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

def sliding_zscore(y, window, overlap):
    step = window - overlap
    if step < 1:
        raise ValueError(f"Invalid window/overlap: step={step} must be >= 1")

    y_out = np.zeros_like(y, dtype=float)
    count = np.zeros_like(y, dtype=int)

    for start in range(0, len(y) - window + 1, step):
        end = start + window
        segment = y[start:end]
        mean = np.mean(segment)
        std = np.std(segment)

        if std == 0 or np.isnan(std) or np.isinf(std):
            norm_segment = np.zeros_like(segment)
        else:
            norm_segment = (segment - mean) / std

        y_out[start:end] += norm_segment
        count[start:end] += 1

    count[count == 0] = 1
    return y_out / count

@register_step
class zscore_sliding_step(BaseStep):
    name = "zscore_sliding"
    category = "General"
    description = "Standardize signal using sliding window (mean 0, std 1)."
    tags = ["time-series"]
    params = [
        {"name": "window", "type": "int", "default": "100", "help": "Window size in samples"},
        {"name": "overlap", "type": "int", "default": "50", "help": "Overlap in samples (must be < window)"}
    ]

    @classmethod
    def get_info(cls): return f"{cls.name} — {cls.description} (Category: {cls.category})"

    @classmethod
    def get_prompt(cls): return {"info": cls.description, "params": cls.params}

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        window = int(user_input.get("window", 100))
        overlap = int(user_input.get("overlap", 50))
        if window <= 0:
            raise ValueError("Window size must be positive")
        if overlap < 0 or overlap >= window:
            raise ValueError("Overlap must be non-negative and less than window size")
        return {"window": window, "overlap": overlap}

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        y = channel.ydata
        x = channel.xdata
        if len(y) < params["window"]:
            raise ValueError("Signal is shorter than window size")

        try:
            y_new = sliding_zscore(y, **params)
            return cls.create_new_channel(parent=channel, xdata=x, ydata=y_new, params=params)
        except Exception as e:
