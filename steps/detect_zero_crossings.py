import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class detect_zero_crossings_step(BaseStep):
    name = "detect_zero_crossings"
    category = "Features"
    description = """Detect zero crossings in the signal where values change sign.
Returns interpolated X-values where the signal crosses zero (Y=0)."""
    tags = ["zero-crossing", "sign-change", "events", "detection", "marker"]
    params = [
        {"name": "threshold", "type": "float", "default": "0.0", "help": "Threshold around zero for crossing detection"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        cls.validate_numeric_parameter("threshold", params.get("threshold"), min_val=0.0)

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        threshold = params.get("threshold", 0.0)

        crossings = []

        for i in range(len(y) - 1):
            y0, y1 = y[i], y[i + 1]
            x0, x1 = x[i], x[i + 1]

            # Simple zero crossing
            if threshold == 0.0:
                if y0 * y1 < 0:
                    # Linear interpolation: x_zero = x0 - y0 * (x1 - x0) / (y1 - y0)
                    if y1 != y0:
                        x_cross = x0 - y0 * (x1 - x0) / (y1 - y0)
                        crossings.append(x_cross)
            else:
                # Thresholded crossing
                if (y0 < -threshold and y1 > threshold) or (y0 > threshold and y1 < -threshold):
                    if y1 != y0:
                        x_cross = x0 - y0 * (x1 - x0) / (y1 - y0)
                        crossings.append(x_cross)

        return [
            {
                'tags': ['time-series'],
                'x': np.array(crossings),
                'y': np.zeros(len(crossings))  # all y-values at crossing are 0
            }
        ]
