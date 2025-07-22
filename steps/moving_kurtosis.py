
import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class moving_kurtosis_step(BaseStep):
    name = "moving_kurtosis"
    category = "Features"
    description = """Compute moving kurtosis over a sliding window.
Measures the 'tailedness' of the signal distribution in each window."""
    tags = ["time-series", "kurtosis", "statistics", "distribution", "sliding-window", "moments"]
    params = [
        {"name": "window", "type": "int", "default": "50", "help": "Window size in samples for kurtosis computation"},
        {"name": "overlap", "type": "int", "default": "25", "help": "Overlap between consecutive windows in samples"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate cross-field logic and business rules"""
        window = cls.validate_integer_parameter("window", params.get("window"), min_val=4)
        overlap = cls.validate_integer_parameter("overlap", params.get("overlap"), min_val=0, max_val=window-1)

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        window = params["window"]
        overlap = params["overlap"]
        
        # Check if signal is long enough for the window
        if len(y) < window:
            raise ValueError(f"Signal too short: requires at least {window} samples")
        
        # Calculate step size
        step = window - overlap
        
        # Compute moving kurtosis with overlap
        y_kurtosis = np.zeros_like(y)
        
        for i in range(0, len(y), step):
            start_idx = i
            end_idx = min(i + window, len(y))
            window_data = y[start_idx:end_idx]
            
            if len(window_data) >= 4:  # Need at least 4 points for kurtosis
                # Compute kurtosis directly
                mean = np.mean(window_data)
                std = np.std(window_data, ddof=1)
                
                if std == 0:
                    kurtosis = np.nan
                else:
                    # Fisher's kurtosis (excess kurtosis)
                    kurtosis = np.mean(((window_data - mean) / std) ** 4) - 3
                
                # Assign kurtosis to all samples in this window
                y_kurtosis[start_idx:end_idx] = kurtosis
            else:
                # Assign NaN to samples in this window
                y_kurtosis[start_idx:end_idx] = np.nan
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_kurtosis
            }
        ]
