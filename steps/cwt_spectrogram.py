
import numpy as np
import pywt

from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class cwt_spectrogram(BaseStep):
    name = "cwt spectrogram"
    category = "Spectrogram"
    description = """Compute a spectrogram using Continuous Wavelet Transform (CWT).
    
Reduction methods:
• **max_intensity**: Maximum power in each time slice
• **sum_intensity**: Total power in each time slice  
• **mean_intensity**: Average power across frequencies
• **max_frequency**: Frequency with the highest power
• **centroid_freq**: Weighted average frequency
• **threshold_count**: Number of frequencies above a threshold
• **band_power**: Power in a user-defined frequency band"""
    tags = ["stft", "cwt", "wavelet", "pywt", "continuous"]
    params = [
        {"name": "wavelet", "type": "str", "default": "morl", "options": ["morl", "mexh", "cgau1", "cgau2"], "help": "Wavelet function for CWT"},
        {"name": "scales_min", "type": "float", "default": "1.0", "help": "Minimum scale (higher frequency)"},
        {"name": "scales_max", "type": "float", "default": "128.0", "help": "Maximum scale (lower frequency)"},
        {"name": "num_scales", "type": "int", "default": "64", "help": "Number of scales to compute"},
        {"name": "reduction", "type": "str", "default": "max_intensity", "options": ["max_intensity", "sum_intensity", "mean_intensity", "max_frequency", "centroid_freq", "threshold_count", "band_power"], "help": "Reduction method for 1D time-series"},
        {"name": "threshold", "type": "float", "default": "0.1", "help": "Threshold value for threshold_count reduction method"},
        {"name": "band_min_freq", "type": "float", "default": "1.0", "help": "Minimum frequency for band_power reduction method (Hz)"},
        {"name": "band_max_freq", "type": "float", "default": "10.0", "help": "Maximum frequency for band_power reduction method (Hz)"}
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
                                                valid_options=["max_intensity", "sum_intensity", "mean_intensity", "max_frequency", "centroid_freq", "threshold_count", "band_power"])
        threshold = cls.validate_numeric_parameter("threshold", params.get("threshold"), min_val=0.0)
        band_min_freq = cls.validate_numeric_parameter("band_min_freq", params.get("band_min_freq"), min_val=0.0)
        band_max_freq = cls.validate_numeric_parameter("band_max_freq", params.get("band_max_freq"), min_val=0.0)
        
        if scales_min >= scales_max:
            raise ValueError(f"scales_min ({scales_min}) must be less than scales_max ({scales_max})")
        
        if band_min_freq >= band_max_freq:
            raise ValueError(f"band_min_freq ({band_min_freq}) must be less than band_max_freq ({band_max_freq})")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        wavelet = params["wavelet"]
        scales_min = params["scales_min"]
        scales_max = params["scales_max"]
        num_scales = params["num_scales"]
        reduction = params["reduction"]
        threshold = params.get("threshold", 0.1)
        band_min_freq = params.get("band_min_freq", 1.0)
        band_max_freq = params.get("band_max_freq", 10.0)
        
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
        elif reduction == "max_frequency":
            # Find frequency with maximum power at each time point
            max_freq_indices = np.argmax(cwt_magnitude, axis=0)
            reduced_data = frequencies[max_freq_indices]
        elif reduction == "centroid_freq":
            # Compute weighted average frequency (spectral centroid)
            # Normalize weights for each time slice
            weights = cwt_magnitude / (np.sum(cwt_magnitude, axis=0, keepdims=True) + 1e-12)
            reduced_data = np.sum(frequencies[:, np.newaxis] * weights, axis=0)
        elif reduction == "threshold_count":
            # Count number of frequencies above threshold at each time point
            reduced_data = np.sum(cwt_magnitude > threshold, axis=0)
        elif reduction == "band_power":
            # Sum power within specified frequency band
            freq_mask = (frequencies >= band_min_freq) & (frequencies <= band_max_freq)
            if not np.any(freq_mask):
                raise ValueError(f"No frequencies found in band [{band_min_freq}, {band_max_freq}] Hz")
            reduced_data = np.sum(cwt_magnitude[freq_mask, :], axis=0)
        else:
            raise ValueError(f"Unknown reduction method: {reduction}")
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': reduced_data
            }
        ]
