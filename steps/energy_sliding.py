import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class energy_sliding_step(BaseStep):
    name = "energy_sliding"
    category = "Features"
    description = """Compute signal energy over sliding windows.
Measures the power/energy content of the signal in each window."""
    tags = ["energy", "power", "sliding-window", "signal-strength"]
    params = [
        {"name": "window", "type": "int", "default": "50", "help": "Window size in samples for energy computation"},
        {"name": "overlap", "type": "int", "default": "25", "help": "Overlap between consecutive windows in samples"}
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
        
        # Calculate step size
        step = window - overlap
        
        # Compute sliding energy with overlap
        y_energy = np.zeros_like(y)
        
        for i in range(0, len(y), step):
            start_idx = i
            end_idx = min(i + window, len(y))
            window_data = y[start_idx:end_idx]
            
            # Compute energy as sum of squared values
            energy = np.sum(window_data**2)
            
            # Assign energy to all samples in this window
            y_energy[start_idx:end_idx] = energy
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_energy
            }
        ]
