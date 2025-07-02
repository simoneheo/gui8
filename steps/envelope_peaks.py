import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class envelope_peaks_step(BaseStep):
    name = "envelope_peaks"
    category = "Envelope"
    description = "Envelope by connecting peaks"
    tags = ["time-series"]
    params = [
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
            parsed[param["name"]] = float(value) if param["type"] == "float" else int(value)
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        y = channel.ydata
        x = channel.xdata
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(y)
        y_new = np.interp(np.arange(len(y)), peaks, y[peaks])
        return cls.create_new_channel(parent=channel, xdata=x, ydata=y_new, params=params)
