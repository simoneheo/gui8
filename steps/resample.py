import numpy as np

from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class resample_step(BaseStep):
    name = "resample"
    category = "Transform"
    description = """Resample signal to a different sampling rate using various methods.
Changes the number of samples while preserving signal characteristics."""
    tags = ["time-series", "sampling", "interpolation", "scipy", "fourier", "upsampling", "downsampling"]
    params = [
        {"name": "new_fs", "type": "float", "default": "100.0", "help": "Target sampling rate in Hz"},
        {"name": "method", "type": "str", "default": "linear", "options": ["fourier", "linear", "cubic", "nearest", "polynomial", "spline"], "help": "Resampling method"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate cross-field logic and business rules"""
        new_fs = cls.validate_numeric_parameter("new_fs", params.get("new_fs"), min_val=0.1)
        method = cls.validate_string_parameter("method", params.get("method"), 
                                             valid_options=["fourier", "linear", "cubic", "nearest", "polynomial", "spline"])

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        from scipy.signal import resample as scipy_resample
        from scipy.interpolate import interp1d

        new_fs = params["new_fs"]
        method = params.get("method", "fourier")
        
        if fs is None or fs <= 0:
            raise ValueError("Original sampling frequency must be positive")
        
        # Calculate number of samples for new sampling rate
        duration = len(y) / fs
        new_num_samples = int(duration * new_fs)
        
        if new_num_samples <= 1:
            raise ValueError(f"New sampling rate too low: would result in {new_num_samples} samples")
        
        # Create new time axis
        x_resampled = np.linspace(x[0], x[-1], new_num_samples)
        
        # Apply different resampling methods
        if method == "fourier":
            # Fourier method (best for frequency domain preservation)
            y_resampled = scipy_resample(y, new_num_samples)
            
        elif method == "linear":
            # Linear interpolation (fast, good for smooth signals)
            interpolator = interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')
            y_resampled = interpolator(x_resampled)
            
        elif method == "cubic":
            # Cubic interpolation (smooth, good for most signals)
            interpolator = interp1d(x, y, kind='cubic', bounds_error=False, fill_value='extrapolate')
            y_resampled = interpolator(x_resampled)
            
        elif method == "nearest":
            # Nearest neighbor (preserves discrete values, good for categorical data)
            interpolator = interp1d(x, y, kind='nearest', bounds_error=False, fill_value='extrapolate')
            y_resampled = interpolator(x_resampled)
            
        elif method == "polynomial":
            # Polynomial interpolation (high order, may oscillate)
            interpolator = interp1d(x, y, kind=3, bounds_error=False, fill_value='extrapolate')
            y_resampled = interpolator(x_resampled)
            
        elif method == "spline":
            # Spline interpolation (smooth, good for noisy signals)
            interpolator = interp1d(x, y, kind='quadratic', bounds_error=False, fill_value='extrapolate')
            y_resampled = interpolator(x_resampled)
            
        else:
            raise ValueError(f"Unknown resampling method: {method}")
        
        return [
            {
                'tags': ['time-series'],
                'x': x_resampled,
                'y': y_resampled
            }
        ]
