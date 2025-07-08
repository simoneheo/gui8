import numpy as np
from steps.base_step import BaseStep
from steps.process_registry import register_step
from channel import Channel

@register_step
class median_sliding_step(BaseStep):
    name = "median_sliding"
    category = "Smoother"
    description = "Computes median over sliding sample windows."
    tags = ["time-series"]
    params = [
        {"name": "window", "type": "int", "default": "100", "help": "Window size in samples"},
        {"name": "overlap", "type": "float", "default": "0.5", "help": "Overlap fraction [0.0 - 0.9]"}
    ]

    @classmethod
    def get_info(cls): return f"{cls.name} — {cls.description} (Category: {cls.category})"

    @classmethod
    def get_prompt(cls): return {"info": cls.description, "params": cls.params}

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        window = int(user_input.get("window", 100))
        overlap = float(user_input.get("overlap", 0.5))
        if window < 1:
            raise ValueError("Window must be at least 1 sample")
        if not (0.0 <= overlap < 1.0):
            raise ValueError("Overlap must be between 0.0 and 0.9")
        return {"window": window, "overlap": overlap}

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        x, y = channel.xdata, channel.ydata
        win = params["window"]
        ovlp = params["overlap"]
        step = int(win * (1 - ovlp))

        if win > len(y):
            raise ValueError(f"Window size ({win}) exceeds signal length ({len(y)})")

        if step < 1:
            raise ValueError(f"Invalid step size: {step}. Increase window or reduce overlap.")

        x_new = []
        y_new = []

        for start in range(0, len(y) - win + 1, step):
            end = start + win
            segment_y = y[start:end]
            segment_x = x[start:end]

            try:
                result = np.median(segment_y)
                if np.isnan(result) or np.isinf(result):
                    continue
                center_x = segment_x[len(segment_x) // 2]
                x_new.append(center_x)
                y_new.append(result)
            except Exception:
                continue

        if len(y_new) == 0:
            raise ValueError("No valid windows found")

        return cls.create_new_channel(parent=channel, xdata=np.array(x_new), ydata=np.array(y_new), params=params)
