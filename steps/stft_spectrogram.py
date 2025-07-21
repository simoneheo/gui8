
import numpy as np
from scipy.signal import stft
import pywt

from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class stft_spectrogram(BaseStep):
    name = "stft_spectrogram"
    category = "Spectrogram"
    description = """Compute a spectrogram using Short-Time Fourier Transform (STFT) and output both:
1. A 2D spectrogram channel for visualizing frequency over time.
2. A 1D time-series channel summarizing the spectrogram using a reduction method.

Reduction methods:
• **max_intensity**: Maximum power in each time slice
• **sum_intensity**: Total power in each time slice  
• **mean_intensity**: Average power across frequencies
• **max_frequency**: Frequency with the highest power
• **centroid_freq**: Weighted average frequency
• **threshold_count**: Number of frequencies above a threshold
• **band_power**: Power in a user-defined frequency band"""

    tags = ["stft", "scipy", "frequency", "window", "fft"]
    params = [
        {
            "name": "fs", 
            "type": "float", 
            "default": "auto", 
            "help": "Sampling rate (Hz) - auto-calculated from channel time data"
        },
        {
            "name": "window", 
            "type": "str", 
            "default": "hann", 
            "options": ["hann", "hamming", "blackman"], 
            "help": "Window function to use for STFT"
        },
        {
            "name": "nperseg", 
            "type": "int", 
            "default": "256", 
            "help": "Length of each segment for STFT (must be positive)"
        },
        {
            "name": "noverlap", 
            "type": "int", 
            "default": "128", 
            "help": "Number of points to overlap between segments (must be < nperseg)"
        },
        {
            "name": "reduction", 
            "type": "str", 
            "default": "max_intensity", 
            "options": ["max_intensity", "sum_intensity", "mean_intensity", "max_frequency", "centroid_freq", "threshold_count", "band_power"], 
            "help": "Reduction method for producing 1D time-series"
        },
        {
            "name": "threshold", 
            "type": "float", 
            "default": "0.1", 
            "help": "Threshold for 'threshold_count' method"
        },
        {
            "name": "band", 
            "type": "str", 
            "default": "0.1-0.5", 
            "help": "Frequency range for 'band_power' method (e.g., '0.2-0.5')"
        }
    ]


    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        # Validate nperseg parameter
        nperseg = cls.validate_integer_parameter(
            "nperseg", 
            params.get("nperseg"), 
            min_val=1
        )
        
        # Validate noverlap parameter
        noverlap = cls.validate_integer_parameter(
            "noverlap", 
            params.get("noverlap"), 
            min_val=0,
            max_val=nperseg-1
        )
        
        # Validate reduction method
        reduction = cls.validate_string_parameter(
            "reduction",
            params.get("reduction", "max_intensity"),
            valid_options=["max_intensity", "sum_intensity", "mean_intensity", "max_frequency", "centroid_freq", "threshold_count", "band_power"]
        )
        
        # Validate threshold for threshold_count method
        if reduction == "threshold_count":
            threshold = cls.validate_numeric_parameter(
                "threshold",
                params.get("threshold", 0.1),
                min_val=0.0
            )
        
        # Validate band format for band_power method
        if reduction == "band_power":
            band_str = cls.validate_string_parameter(
                "band",
                params.get("band", "0.1-0.5")
            )
            
            try:
                band_parts = band_str.split('-')
                if len(band_parts) != 2:
                    raise ValueError("Band format should be 'low-high' (e.g., '0.1-0.5')")
                band_low = float(band_parts[0])
                band_high = float(band_parts[1])
                if band_low >= band_high:
                    raise ValueError("Band low frequency must be less than high frequency")
                if band_low < 0:
                    raise ValueError("Band low frequency must be non-negative")
            except (ValueError, IndexError) as e:
                raise ValueError(f"Invalid band specification '{band_str}': {str(e)}")
  

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        """Core processing logic for STFT spectrogram computation"""
        window = params.get("window", "hann")
        nperseg = params.get("nperseg", 256)
        noverlap = params.get("noverlap", 128)
        reduction = params.get("reduction", "max_intensity")
        threshold = params.get("threshold", 0.1)
        band_str = params.get("band", "0.1-0.5")
        
        # Parse frequency band for band_power method
        band_low, band_high = 0.1, 0.5
        if reduction == "band_power":
            band_parts = band_str.split('-')
            band_low = float(band_parts[0])
            band_high = float(band_parts[1])
        
        # Compute STFT
        f, t, Zxx = stft(y, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
        
        # Convert to power spectrogram
        Pxx = np.abs(Zxx)**2
        
        # Align time axis with original signal's time axis
        if x is not None and len(x) > 0:
            t_start = x[0]
            t_aligned = t + t_start
        else:
            t_aligned = t
        
        # Apply reduction method
        if reduction == "max_intensity":
            reduced_data = np.max(Pxx, axis=0)
            ylabel = "Max Power"
        elif reduction == "sum_intensity":
            reduced_data = np.sum(Pxx, axis=0)
            ylabel = "Total Power"
        elif reduction == "mean_intensity":
            reduced_data = np.mean(Pxx, axis=0)
            ylabel = "Mean Power"
        elif reduction == "max_frequency":
            # Exclude DC component (first frequency bin) to avoid always getting 0 Hz
            if len(f) > 1:
                max_indices = np.argmax(Pxx[1:, :], axis=0) + 1
                reduced_data = f[max_indices]
            else:
                max_indices = np.argmax(Pxx, axis=0)
                reduced_data = f[max_indices]
            ylabel = "Peak Frequency (Hz)"
        elif reduction == "centroid_freq":
            # Weighted average frequency
            power_sum = np.sum(Pxx, axis=0)
            power_sum[power_sum == 0] = 1e-10
            reduced_data = np.sum(f[:, np.newaxis] * Pxx, axis=0) / power_sum
            ylabel = "Spectral Centroid (Hz)"
        elif reduction == "threshold_count":
            reduced_data = np.sum(Pxx > threshold, axis=0)
            ylabel = f"Count > {threshold}"
        elif reduction == "band_power":
            # Find frequency indices for the band
            band_indices = (f >= band_low) & (f <= band_high)
            if not np.any(band_indices):
                raise ValueError(f"No frequencies found in band {band_low}-{band_high} Hz")
            reduced_data = np.sum(Pxx[band_indices, :], axis=0)
            ylabel = f"Band Power ({band_low}-{band_high} Hz)"
        else:
            raise ValueError(f"Unknown reduction method: {reduction}")
        
        return [
            {
                'tags': ['time-series'],
                'x': t_aligned,
                'y': reduced_data
            },
            {
                'tags': ['spectrogram'],
                'x': t_aligned,
                'y': f,
                't': t_aligned,
                'f': f,
                'z': Pxx
            }
        ]
