import numpy as np
from steps.base_step import BaseStep
from steps.process_registry import register_step
from channel import Channel

@register_step
class EnergySlidingStep(BaseStep):
    name = "energy_sliding"
    category = "Features"
    description = "Computes signal energy (sum of squares) in overlapping sliding windows."
    tags = ["time-series", "feature"]
    params = [
        {"name": "window", "type": "int", "default": "100", "help": "Window size in samples"},
        {"name": "overlap", "type": "float", "default": "0.5", "help": "Overlap fraction [0.0 - 0.9]"}
    ]

    @classmethod
    def get_info(cls): 
        return f"{cls.name} — {cls.description} (Category: {cls.category})"
    
    @classmethod
    def get_prompt(cls): 
        return {"info": cls.description, "params": cls.params}
    
    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        try:
            window = int(user_input.get("window", 100))
            overlap = float(user_input.get("overlap", 0.5))

            if window <= 0:
                raise ValueError("Window size must be > 0")
            if not (0.0 <= overlap < 1.0):
                raise ValueError("Overlap must be between 0.0 and 0.9")
            return {"window": window, "overlap": overlap}
        except Exception as e:
            raise ValueError(f"Invalid parameters: {str(e)}")

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        x, y = channel.xdata, channel.ydata
        window = params["window"]
        overlap = params["overlap"]
        step = int(window * (1 - overlap))

        if len(y) < window:
            raise ValueError(f"Signal too short for window size {window}")
        if step < 1:
            raise ValueError(f"Step size too small (step={step}). Increase window or reduce overlap")

        x_new, y_new = [], []

        for start in range(0, len(y) - window + 1, step):
            end = start + window
            segment = y[start:end]
            center_x = x[start + window // 2]

            try:
                energy = np.sum(segment ** 2)
                if not np.isfinite(energy):
                    continue
                x_new.append(center_x)
                y_new.append(energy)
            except Exception:
                continue

        if len(y_new) == 0:
            raise ValueError("No valid windows computed")

        return cls.create_new_channel(
            parent=channel,
            xdata=np.array(x_new),
            ydata=np.array(y_new),
            params=params
        )
