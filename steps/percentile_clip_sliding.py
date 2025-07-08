import numpy as np
from steps.base_step import BaseStep
from steps.process_registry import register_step
from channel import Channel

def percentile_clip_sliding(y, lower=1.0, upper=99.0, window=100, overlap=0.5):
    if not (0 <= lower < upper <= 100):
        raise ValueError(f"Invalid percentile range: {lower}–{upper}")
    if window < 2:
        raise ValueError("Window size must be >= 2")
    if not (0 <= overlap < 1):
        raise ValueError("Overlap must be between 0 and 1")

    step = max(1, int(window * (1 - overlap)))
    output = np.copy(y)
    for start in range(0, len(y) - window + 1, step):
        end = start + window
        segment = y[start:end]
        lo = np.percentile(segment, lower)
        hi = np.percentile(segment, upper)
        output[start:end] = np.clip(segment, lo, hi)

    return output

@register_step
class percentile_clip_sliding_step(BaseStep):
    name = "percentile_clip_sliding"
    category = "General"
    description = "Applies percentile clipping within sliding windows."
    tags = ["time-series"]
    params = [
        {'name': 'lower', 'type': 'float', 'default': '1.0', 'help': 'Lower percentile (0–100)'},
        {'name': 'upper', 'type': 'float', 'default': '99.0', 'help': 'Upper percentile (0–100)'},
        {'name': 'window', 'type': 'int', 'default': '100', 'help': 'Window size in samples'},
        {'name': 'overlap', 'type': 'float', 'default': '0.5', 'help': 'Overlap fraction [0.0–0.9]'}
    ]

    @classmethod
    def get_info(cls): return f"{cls.name} — {cls.description} (Category: {cls.category})"

    @classmethod
    def get_prompt(cls): return {"info": cls.description, "params": cls.params}

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        return {
            'lower': float(user_input.get('lower', 1.0)),
            'upper': float(user_input.get('upper', 99.0)),
            'window': int(user_input.get('window', 100)),
            'overlap': float(user_input.get('overlap', 0.5))
        }

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        y = channel.ydata
        x = channel.xdata

        y_new = sliding_percentile_clip(
            y,
            lower=params['lower'],
            upper=params['upper'],
            window=params['window'],
            overlap=params['overlap']
        )

        return cls.create_new_channel(
            parent=channel,
            xdata=x,
            ydata=y_new,
            params=params
        )
