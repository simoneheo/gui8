import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class area_envelope_step(BaseStep):
    name = "area_envelope"
    category = "Envelope"
    description = (
        "Compute sliding window area (sum of absolute values) with optional overlap. "
        "Used for envelope tracking or energy estimation."
    )
    tags = ["time-series"]
    params = [
        {"name": "window", "type": "int", "default": "25", "help": "Window size in samples"},
        {"name": "overlap", "type": "int", "default": "0", "help": "Overlap between windows in samples"},
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
        y = np.abs(channel.ydata)
        x = channel.xdata
        window = params['window']
        overlap = params['overlap']
        step = max(1, window - overlap)

        envelope = np.zeros_like(y)
        counts = np.zeros_like(y)

        for start in range(0, len(y) - window + 1, step):
            end = start + window
            envelope[start:end] += np.sum(y[start:end])
            counts[start:end] += 1

        # Avoid division by zero
        counts[counts == 0] = 1
        y_new = envelope / counts

        return cls.create_new_channel(parent=channel, xdata=x, ydata=y_new, params=params)
