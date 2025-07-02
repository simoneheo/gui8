import numpy as np
from steps.base_step import BaseStep
from steps.process_registry import register_step
from channel import Channel

@register_step
class DetectZeroCrossingsStep(BaseStep):
    name = "detect_zero_crossings"
    category = "Event"
    description = "Detects zero-crossing events where signal changes sign (positive to negative or vice versa)."
    tags = ["time-series", "event"]
    params = [{'name': 'threshold', 'type': 'float', 'default': '0.0', 'help': 'Crossing threshold value. Events occur when signal crosses this level. Use 0.0 for true zero-crossings, or signal offset for baseline crossings.'}]

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
            indices = np.where(np.diff(np.sign(y)) != 0)[0]
        except Exception as e:
            raise ValueError(f"Failed during event detection: {str(e)}")

        return cls.create_new_channel(parent=channel, xdata=x[indices], ydata=y[indices], params=params)
