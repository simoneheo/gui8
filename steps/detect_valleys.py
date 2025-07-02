import numpy as np
from steps.base_step import BaseStep
from steps.process_registry import register_step
from channel import Channel
from scipy.signal import find_peaks
@register_step
class DetectValleysStep(BaseStep):
    name = "detect_valleys"
    category = "Event"
    description = "Detects valley (minimum) points in the signal using inverted peak detection."
    tags = ["time-series", "event"]
    params = [{'name': 'height', 'type': 'float', 'default': '0.0', 'help': 'Maximum valley depth (negative amplitude). Valleys above this value are ignored. Use negative values for depth below baseline.'}]

    @classmethod
    def get_info(cls): return f"{cls.name} — {cls.description} (Category: {cls.category})"
    @classmethod
    def get_prompt(cls): return {"info": cls.description, "params": cls.params}
    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        parsed = {}

        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        x, y = channel.xdata, channel.ydata

        try:
        
            indices, _ = find_peaks(-y)
        except Exception as e:
            raise ValueError(f"Failed during event detection: {str(e)}")

        return cls.create_new_channel(parent=channel, xdata=x[indices], ydata=y[indices], params=params)
