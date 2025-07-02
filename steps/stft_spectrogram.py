
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
    description = """Computes a spectrogram using Short-Time Fourier Transform (STFT) and outputs both:
1. A 2D spectrogram channel (tag='spectrogram') for visualizing frequency over time.
2. A 1D time-series channel (tag='time-series') summarizing the spectrogram using a reduction method.

Reduction methods:
- max_intensity: Maximum power in each time slice.
- sum_intensity: Total power in each time slice.
- mean_intensity: Average power across frequencies.
- max_frequency: Frequency with the highest power.
- centroid_freq: Weighted average frequency.
- threshold_count: Number of frequencies above a threshold.
- band_power: Power in a user-defined band.
"""
    tags = ["spectrogram"]
    params = [
        {"name": "window", "type": "str", "default": "hann", "options": ["hann", "hamming", "blackman"], "help": "Window function to use for STFT."},
        {"name": "nperseg", "type": "int", "default": "256", "help": "Length of each segment for STFT."},
        {"name": "noverlap", "type": "int", "default": "128", "help": "Number of points to overlap between segments."},
        {"name": "reduction", "type": "str", "default": "max_intensity", "options": ["max_intensity", "sum_intensity", "mean_intensity", "max_frequency", "centroid_freq", "threshold_count", "band_power"], "help": "Reduction method for producing 1D time-series."},
        {"name": "threshold", "type": "float", "default": "0.1", "help": "Threshold for 'threshold_count' method."},
        {"name": "band", "type": "str", "default": "0.1-0.5", "help": "Frequency range for 'band_power' method (e.g., '0.2-0.5')."},
        {"name": "fs", "type": "float", "default": "", "help": "Sampling frequency (injected from parent channel)."}
    ]

    @classmethod
    def get_info(cls):
        return f"{cls.name} — {cls.description} (Category: {cls.category})"

    @classmethod
    def get_prompt(cls):
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        parsed = {}
        for param in cls.params:
            name = param["name"]
            val = user_input.get(name, param["default"])
            try:
                if val == "":
                    parsed[name] = None
                elif param["type"] == "float":
                    parsed[name] = float(val)
                elif param["type"] == "int":
                    parsed[name] = int(val)
                else:
                    parsed[name] = val
            except ValueError as e:
                raise ValueError(f"Invalid input for '{name}': {str(e)}")
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> list:
        # Input validation
        if len(channel.ydata) < 10:
            raise ValueError("Signal too short for STFT computation (minimum 10 samples).")
        if np.all(np.isnan(channel.ydata)):
            raise ValueError("Signal contains only NaN values.")
        
        # Get parameters
        window = params.get("window", "hann")
        nperseg = params.get("nperseg", 256)
        noverlap = params.get("noverlap", 128)
        reduction = params.get("reduction", "max_intensity")
        threshold = params.get("threshold", 0.1)
        band_str = params.get("band", "0.1-0.5")
        fs = params.get("fs", 1.0)
        
        # Validate parameters
        if nperseg <= 0:
            raise ValueError("nperseg must be positive")
        if noverlap < 0 or noverlap >= nperseg:
            raise ValueError("noverlap must be non-negative and less than nperseg")
        if nperseg > len(channel.ydata):
            nperseg = len(channel.ydata) // 2
            noverlap = nperseg // 2
        
        # Parse frequency band for band_power method
        band_low, band_high = 0.1, 0.5
        if reduction == "band_power":
            try:
                band_parts = band_str.split('-')
                if len(band_parts) == 2:
                    band_low = float(band_parts[0])
                    band_high = float(band_parts[1])
                    if band_low >= band_high:
                        raise ValueError("Band low frequency must be less than high frequency")
                    if band_low < 0 or band_high > fs/2:
                        raise ValueError(f"Band frequencies must be between 0 and {fs/2} Hz")
                else:
                    raise ValueError("Band format should be 'low-high' (e.g., '0.1-0.5')")
            except (ValueError, IndexError) as e:
                raise ValueError(f"Invalid band specification '{band_str}': {str(e)}")
        
        try:
            # Compute STFT
            f, t, Zxx = stft(channel.ydata, fs=fs, window=window, 
                           nperseg=nperseg, noverlap=noverlap)
            
            # Convert to power spectrogram
            Pxx = np.abs(Zxx)**2
            
            # Align time axis with original signal's time axis
            if hasattr(channel, 'xdata') and channel.xdata is not None and len(channel.xdata) > 0:
                # Offset time axis by the start time of the original signal
                t_start = channel.xdata[0]
                t_aligned = t + t_start
                print(f"[STFT] Original time range: {channel.xdata[0]:.3f} - {channel.xdata[-1]:.3f}s")
                print(f"[STFT] STFT time range: {t[0]:.3f} - {t[-1]:.3f}s → {t_aligned[0]:.3f} - {t_aligned[-1]:.3f}s")
            else:
                # Fallback: use STFT time axis as-is
                t_aligned = t
                print(f"[STFT] Warning: No original time axis found, using STFT time axis as-is")
            
        except Exception as e:
            raise ValueError(f"STFT computation failed: {str(e)}")
        
        # Create spectrogram channel
        spectrogram_channel = cls.create_new_channel(
            parent=channel,
            xdata=t_aligned,  # Aligned time axis
            ydata=f,  # Frequency axis
            params=params
        )
        
        # Set spectrogram-specific properties
        spectrogram_channel.tags = ["spectrogram"]
        spectrogram_channel.xlabel = "Time (s)"
        spectrogram_channel.ylabel = "Frequency (Hz)"
        spectrogram_channel.legend_label = f"{channel.legend_label} - STFT Spectrogram"
        
        # Store spectrogram data in metadata
        spectrogram_channel.metadata = {
            'Zxx': Pxx,
            'colormap': 'viridis'
        }
        
        # Apply reduction method to create time-series channel
        try:
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
                    # Find max in frequencies excluding DC (f[0] = 0 Hz)
                    max_indices = np.argmax(Pxx[1:, :], axis=0) + 1  # +1 to account for excluded DC
                    reduced_data = f[max_indices]
                else:
                    # Fallback if only one frequency bin
                    max_indices = np.argmax(Pxx, axis=0)
                    reduced_data = f[max_indices]
                ylabel = "Peak Frequency (Hz)"
            elif reduction == "centroid_freq":
                # Weighted average frequency
                power_sum = np.sum(Pxx, axis=0)
                # Avoid division by zero
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
                
        except Exception as e:
            raise ValueError(f"Reduction method '{reduction}' failed: {str(e)}")
        
        # Create time-series channel
        timeseries_channel = cls.create_new_channel(
            parent=channel,
            xdata=t_aligned,  # Same aligned time axis as spectrogram
            ydata=reduced_data,
            params=params
        )
        
        # Set time-series specific properties
        timeseries_channel.tags = ["time-series"]
        timeseries_channel.xlabel = "Time (s)"
        timeseries_channel.ylabel = ylabel
        timeseries_channel.legend_label = f"{channel.legend_label} - {reduction.replace('_', ' ').title()}"
        
        return [spectrogram_channel, timeseries_channel]
