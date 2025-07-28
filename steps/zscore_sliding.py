import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class zscore_sliding_step(BaseStep):
    name = "zscore sliding"
    category = "Transform"
    description = """Standardize signal using sliding window z-score normalization.
Centers data around local mean and standard deviation."""
    tags = ["time-series", "zscore", "standardize", "normalize", "sliding", "local"]
    params = [
        {"name": "window", "type": "int", "default": "100", "help": "Window size for local statistics"},
        {"name": "with_mean", "type": "bool", "default": "True", "help": "Center data around mean"},
        {"name": "with_std", "type": "bool", "default": "True", "help": "Scale to unit standard deviation"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        window = cls.validate_integer_parameter("window", params.get("window"), min_val=2)
        with_mean = params.get("with_mean", True)
        with_std = params.get("with_std", True)

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        window = params["window"]
        with_mean = params["with_mean"]
        with_std = params["with_std"]
        
        if len(y) < window:
            raise ValueError(f"Signal too short: requires at least {window} samples")
        
        y_standardized = np.zeros_like(y)
        half_window = window // 2
        for i in range(len(y)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(y), i + half_window + 1)
            window_data = y[start_idx:end_idx]
            mean = np.mean(window_data)
            std = np.std(window_data)
            val = y[i]
            if with_mean:
                val = val - mean
            if with_std:
                if std == 0:
                    raise ValueError(f"Zero standard deviation in window at index {i}")
                val = val / std
            y_standardized[i] = val
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_standardized
            }
        ]
