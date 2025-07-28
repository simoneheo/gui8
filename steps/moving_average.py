import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class moving_average_step(BaseStep):
    name = "moving average"
    category = "Filter"
    description = """Apply moving average smoothing using overlapping windows to reduce noise and smooth signal variations."""
    tags = ["time-series", "smoothing", "noise-reduction", "window", "sliding", "average", "moving"]
    params = [
        {'name': 'window', 'type': 'int', 'default': '5', 'help': 'Window size in samples (must be >= 1)'},
        {'name': 'overlap', 'type': 'int', 'default': '0', 'help': 'Overlap between windows in samples (must be < window)'},
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate cross-field logic and business rules"""
        window = cls.validate_integer_parameter("window", params.get("window"), min_val=1)
        overlap = cls.validate_integer_parameter("overlap", params.get("overlap"), min_val=0, max_val=window-1)

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        window = params["window"]
        overlap = params["overlap"]
        
        # Check if signal is long enough for the window
        if len(y) < window:
            raise ValueError(f"Signal too short: requires at least {window} samples")

        step = window - overlap
        result = np.zeros_like(y, dtype=float)
        count = np.zeros_like(y, dtype=int)

        for start in range(0, len(y) - window + 1, step):
            end = start + window
            segment = y[start:end]
            avg = np.mean(segment)
            result[start:end] += avg
            count[start:end] += 1

        # Avoid divide by zero
        count[count == 0] = 1
        y_new = result / count
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_new
            }
        ]
