import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class hilbert_envelope_step(BaseStep):
    name = "hilbert envelope"
    category = "Transform"
    description = """Extract signal envelope using Hilbert transform.
Computes the analytic signal and returns its magnitude (envelope)."""
    tags = ["time-series", "envelope", "hilbert", "analytic", "magnitude", "amplitude"]
    params = [
        # No parameters needed for Hilbert envelope
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate cross-field logic and business rules"""
        # No parameters to validate for Hilbert envelope
        pass

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        from scipy.signal import hilbert
        
        # Compute Hilbert transform
        analytic_signal = hilbert(y)
        
        # Extract envelope (magnitude of analytic signal)
        y_envelope = np.abs(analytic_signal)
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_envelope
            }
        ]
