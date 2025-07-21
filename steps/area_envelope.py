import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class area_envelope_step(BaseStep):
    name = "area_envelope"
    category = "Transform"
    description = "Compute sliding window area envelope for signal energy estimation"
    tags = ["time-series", "envelope", "energy", "area", "magnitude", "amplitude"]
    params = [
        {"name": "window", "type": "int", "default": "25", "help": "Window size in samples"},
        {"name": "overlap", "type": "int", "default": "0", "help": "Overlap between windows in samples"},
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        # Validate window parameter
        window = cls.validate_integer_parameter(
            "window", 
            params.get("window"), 
            min_val=1
        )
        
        # Validate overlap parameter
        overlap = cls.validate_integer_parameter(
            "overlap", 
            params.get("overlap"), 
            min_val=0
        )
        
        # Validate cross-parameter logic: overlap must be smaller than window
        if overlap >= window:
            raise ValueError("Overlap must be smaller than window size")
 

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> np.ndarray:
        y_abs = np.abs(y)
        window = params["window"]
        overlap = params["overlap"]
        step = max(1, window - overlap)

        envelope = np.zeros_like(y_abs)
        counts = np.zeros_like(y_abs)

        for start in range(0, len(y_abs) - window + 1, step):
            end = start + window
            envelope[start:end] += np.sum(y_abs[start:end])
            counts[start:end] += 1

        # Handle trailing part
        if end < len(y_abs):
            envelope[end:] += np.sum(y_abs[-window:])
            counts[end:] += 1

        counts[counts == 0] = 1
        y_new = envelope / counts
        return y_new