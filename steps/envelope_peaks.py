import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class envelope_peaks_step(BaseStep):
    name = "envelope peaks"
    category = "Features"
    description = """Extract the envelope of signal peaks to show the overall amplitude trend.
Uses peak detection and interpolation to create a smooth envelope."""
    tags = ["envelope", "peaks", "amplitude", "trend"]
    params = [
        {"name": "method", "type": "str", "default": "linear", "options": ["linear", "cubic"], "help": "Interpolation method for envelope"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate cross-field logic and business rules"""
        method = cls.validate_string_parameter("method", params.get("method"), valid_options=["linear", "cubic"])

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        method = params.get("method", "linear")
        
        from scipy.signal import find_peaks
        from scipy.interpolate import interp1d
        
        # Find peaks in the signal
        peaks, _ = find_peaks(y, height=np.mean(y))
        
        if len(peaks) < 2:
            raise ValueError("Not enough peaks found for envelope calculation (minimum 2 required)")
        
        # Extract peak values and positions
        peak_x = x[peaks]
        peak_y = y[peaks]
        
        # Add boundary points to ensure full coverage
        if peak_x[0] > x[0]:
            peak_x = np.insert(peak_x, 0, x[0])
            peak_y = np.insert(peak_y, 0, y[0])
        if peak_x[-1] < x[-1]:
            peak_x = np.append(peak_x, x[-1])
            peak_y = np.append(peak_y, y[-1])
        
        # Interpolate envelope
        try:
            interp_func = interp1d(peak_x, peak_y, kind=method, bounds_error=False, fill_value='extrapolate')
            y_envelope = interp_func(x)
        except Exception as e:
            raise ValueError(f"Envelope interpolation failed: {str(e)}")
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_envelope
            }
        ]
