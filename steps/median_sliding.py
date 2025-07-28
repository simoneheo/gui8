import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class median_sliding_step(BaseStep):
    name = "median sliding"
    category = "Filter"
    description = """Compute median values over sliding sample windows to reduce noise and identify trends."""
    tags = ["time-series", "smoothing", "noise-reduction", "median", "sliding", "robust"]
    params = [
        {"name": "window", "type": "int", "default": "100", "help": "Window size in samples"},
        {"name": "overlap", "type": "int", "default": "50", "help": "Overlap in samples"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        window = cls.validate_integer_parameter("window", params.get("window"), min_val=1)
        overlap = cls.validate_integer_parameter("overlap", params.get("overlap"), min_val=0)
        
        # Validate overlap constraint
        if overlap >= window:
            raise ValueError("Overlap must be less than window size")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        window = params["window"]
        overlap = params["overlap"]
        step = window - overlap
        
        if window > len(y):
            raise ValueError(f"Window size ({window}) exceeds signal length ({len(y)})")
        
        if step < 1:
            raise ValueError(f"Invalid step size: {step}. Increase window or reduce overlap.")
        
        x_new = []
        y_new = []
        
        for start in range(0, len(y) - window + 1, step):
            end = start + window
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
        
        x_new = np.array(x_new)
        y_new = np.array(y_new)
        
        return [
            {
                'tags': ['time-series'],
                'x': x_new,
                'y': y_new
            }
        ]
