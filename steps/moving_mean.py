
import numpy as np
import scipy.signal
import scipy.stats
from steps.base_step import BaseStep
from steps.process_registry import register_step
from channel import Channel

@register_step
class MovingMeanStep(BaseStep):
    name = "moving_mean"
    category = "Features"
    description = "Computes moving mean over sliding windows."
    tags = ["time-series", "feature"]
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
        return {
            "window": int(user_input.get("window", 100)),
            "overlap": float(user_input.get("overlap", 0.5))
        }

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        x, y = channel.xdata, channel.ydata
        win, ovlp = params["window"], params["overlap"]
        step = int(win * (1 - ovlp))
        if win <= 1 or step < 1:
            raise ValueError(f"Invalid window/overlap settings: window={win}, step={step}")
        
        indices = range(0, len(y) - win + 1, step)
        x_new = [x[i + win // 2] for i in indices]
        y_new = [np.mean(window) for i in indices for window in [y[i:i+win]]]

        return cls.create_new_channel(parent=channel, xdata=np.array(x_new), ydata=np.array(y_new), params=params)
