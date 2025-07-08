import numpy as np
from steps.base_step import BaseStep
from steps.process_registry import register_step
from channel import Channel
from scipy.signal import argrelextrema

@register_step
class DetectExtremaStep(BaseStep):
    name = "detect_extrema"
    category = "Event"
    description = (
        "Detects local extrema (peaks and valleys) in overlapping sliding windows. "
        "Window and overlap are in **samples**. Extrema are filtered by height relative to signal mean."
    )
    tags = ["time-series", "event"]
    params = [
        {'name': 'min_height', 'type': 'float', 'default': '0.1', 'help': 'Minimum height/depth as a fraction of signal range (0.0–1.0).'},
        {'name': 'window', 'type': 'int', 'default': '200', 'help': 'Window size in samples.'},
        {'name': 'overlap', 'type': 'int', 'default': '100', 'help': 'Overlap between windows in samples.'}
    ]

    @classmethod
    def get_info(cls):
        return f"{cls.name} — {cls.description} (Category: {cls.category})"

    @classmethod
    def get_prompt(cls):
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        return {
            'min_height': float(user_input.get('min_height', 0.1)),
            'window': int(user_input.get('window', 200)),
            'overlap': int(user_input.get('overlap', 100)),
        }

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        x, y = channel.xdata, channel.ydata
        min_height = params['min_height']
        window_samples = params['window']
        overlap_samples = params['overlap']

        if len(x) < 2:
            raise ValueError("Signal too short")

        if window_samples < 2:
            raise ValueError("Window must be at least 2 samples")
        if overlap_samples < 0 or overlap_samples >= window_samples:
            raise ValueError("Overlap must be non-negative and smaller than window size")

        step = window_samples - overlap_samples
        signal_range = np.max(y) - np.min(y)
        height_thresh = min_height * signal_range
        signal_mean = np.mean(y)

        extrema_indices = []

        for start in range(0, len(y) - window_samples + 1, step):
            end = start + window_samples
            y_window = y[start:end]

            # Use order relative to window size
            order = max(1, window_samples // 10)

            local_max = argrelextrema(y_window, np.greater, order=order)[0]
            local_min = argrelextrema(y_window, np.less, order=order)[0]

            for i in local_max:
                idx = start + i
                if y[idx] - signal_mean >= height_thresh:
                    extrema_indices.append(idx)

            for i in local_min:
                idx = start + i
                if signal_mean - y[idx] >= height_thresh:
                    extrema_indices.append(idx)

        if not extrema_indices:
            return cls.create_new_channel(parent=channel, xdata=np.array([]), ydata=np.array([]), params=params)

        extrema_indices = np.unique(extrema_indices)

        return cls.create_new_channel(
            parent=channel,
            xdata=x[extrema_indices],
            ydata=y[extrema_indices],
            params=params
        )
