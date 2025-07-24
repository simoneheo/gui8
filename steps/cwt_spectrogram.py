
import numpy as np
import pywt

from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class cwt_spectrogram(BaseStep):
    name = "cwt_spectrogram"
    category = "Spectrogram"
    description = """Compute Continuous Wavelet Transform (CWT) spectrogram for time-frequency analysis.
Provides better time-frequency resolution trade-off than STFT."""
    tags = ["stft", "cwt", "wavelet", "pywt", "continuous"]
    params = [
        {"name": "wavelet", "type": "str", "default": "morl", "options": ["morl", "mexh", "cgau1", "cgau2"], "help": "Wavelet function for CWT"},
        {"name": "scales_min", "type": "float", "default": "1.0", "help": "Minimum scale (higher frequency)"},
        {"name": "scales_max", "type": "float", "default": "128.0", "help": "Maximum scale (lower frequency)"},
        {"name": "num_scales", "type": "int", "default": "64", "help": "Number of scales to compute"},
        {"name": "reduction", "type": "str", "default": "max_intensity", "options": ["max_intensity", "sum_intensity", "mean_intensity"], "help": "Reduction method for 1D time-series"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        wavelet = cls.validate_string_parameter("wavelet", params.get("wavelet"), 
                                               valid_options=["morl", "mexh", "cgau1", "cgau2"])
        scales_min = cls.validate_numeric_parameter("scales_min", params.get("scales_min"), min_val=0.1)
        scales_max = cls.validate_numeric_parameter("scales_max", params.get("scales_max"), min_val=0.1)
        num_scales = cls.validate_integer_parameter("num_scales", params.get("num_scales"), min_val=4)
        reduction = cls.validate_string_parameter("reduction", params.get("reduction"), 
                                                valid_options=["max_intensity", "sum_intensity", "mean_intensity"])
        
        if scales_min >= scales_max:
            raise ValueError(f"scales_min ({scales_min}) must be less than scales_max ({scales_max})")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        wavelet = params["wavelet"]
        scales_min = params["scales_min"]
        scales_max = params["scales_max"]
        num_scales = params["num_scales"]
        reduction = params["reduction"]
        
        # Generate scales array (logarithmically spaced)
        scales = np.logspace(np.log10(scales_min), np.log10(scales_max), num_scales)
        
        # Compute CWT
        try:
            coefficients, frequencies = pywt.cwt(y, scales, wavelet, 1.0/fs)
        except Exception as e:
            raise ValueError(f"CWT computation failed: {str(e)}")
        
        # Take magnitude of complex coefficients
        cwt_magnitude = np.abs(coefficients)
        
        # Apply reduction method for 1D time-series
        if reduction == "max_intensity":
            reduced_data = np.max(cwt_magnitude, axis=0)
        elif reduction == "sum_intensity":
            reduced_data = np.sum(cwt_magnitude, axis=0)
        elif reduction == "mean_intensity":
            reduced_data = np.mean(cwt_magnitude, axis=0)
        else:
            raise ValueError(f"Unknown reduction method: {reduction}")
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': reduced_data
            }
        ]
