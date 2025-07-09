
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
• **band_power**: Power in a user-defined frequency band

Useful for:
• **Time-frequency analysis**: Visualize how frequency content changes over time
• **Signal characterization**: Identify dominant frequencies and their temporal evolution
• **Feature extraction**: Extract time-varying spectral features
• **Pattern recognition**: Detect recurring frequency patterns"""
    tags = ["spectrogram", "time-frequency", "stft", "scipy", "frequency", "window", "fft"]
    params = [
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
    def get_info(cls):
        return f"{cls.name} — Compute STFT spectrogram with time-frequency analysis (Category: {cls.category})"

    @classmethod
    def get_prompt(cls):
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input signal data"""
        if len(y) < 10:
            raise ValueError("Signal too short for STFT computation (minimum 10 samples)")
        if np.all(np.isnan(y)):
            raise ValueError("Signal contains only NaN values")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        nperseg = params.get("nperseg")
        noverlap = params.get("noverlap")
        reduction = params.get("reduction", "max_intensity")
        threshold = params.get("threshold", 0.1)
        band_str = params.get("band", "0.1-0.5")
        
        if nperseg <= 0:
            raise ValueError("nperseg must be positive")
        if noverlap < 0 or noverlap >= nperseg:
            raise ValueError("noverlap must be non-negative and less than nperseg")
        
        # Validate threshold for threshold_count method
        if reduction == "threshold_count" and threshold < 0:
            raise ValueError("Threshold must be non-negative")
        
        # Validate band format for band_power method
        if reduction == "band_power":
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
    def _validate_output_data(cls, y_original: np.ndarray, spectrogram_data: np.ndarray, timeseries_data: np.ndarray) -> None:
        """Validate output data"""
        if spectrogram_data.size == 0:
            raise ValueError("STFT computation produced empty spectrogram")
        if timeseries_data.size == 0:
            raise ValueError("Reduction method produced empty time-series")
        if np.any(np.isnan(spectrogram_data)) and not np.any(np.isnan(y_original)):
            raise ValueError("STFT computation produced unexpected NaN values")

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        """Parse and validate user input parameters"""
        parsed = {}
        for param in cls.params:
            name = param["name"]
            val = user_input.get(name, param.get("default"))
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
                if "could not convert" in str(e) or "invalid literal" in str(e):
                    raise ValueError(f"{name} must be a valid {param['type']}")
                raise e
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> list:
        """Apply STFT spectrogram computation to the channel data."""
        try:
            x = channel.xdata
            y = channel.ydata
            fs = getattr(channel, 'fs', 1.0)
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            
            # Adjust nperseg if needed
            nperseg = params.get("nperseg", 256)
            if nperseg > len(y):
                nperseg = len(y) // 2
                params["nperseg"] = nperseg
                params["noverlap"] = nperseg // 2
            
            # Process the data
            spectrogram_data, timeseries_data, t_aligned, f, ylabel = cls.script(x, y, fs, params)
            
            # Validate output data
            cls._validate_output_data(y, spectrogram_data, timeseries_data)
            
            # Create spectrogram channel
            spectrogram_channel = cls.create_new_channel(
                parent=channel,
                xdata=t_aligned,
                ydata=f,
                params=params,
                suffix="STFTSpectrogram"
            )
            
            # Set spectrogram-specific properties
            spectrogram_channel.tags = ["spectrogram"]
            spectrogram_channel.xlabel = "Time (s)"
            spectrogram_channel.ylabel = "Frequency (Hz)"
            spectrogram_channel.legend_label = f"{channel.legend_label} - STFT Spectrogram"
            
            # Store spectrogram data in metadata
            spectrogram_channel.metadata = {
                'Zxx': spectrogram_data,
                'colormap': 'viridis'
            }
            
            # Create time-series channel
            timeseries_channel = cls.create_new_channel(
                parent=channel,
                xdata=t_aligned,
                ydata=timeseries_data,
                params=params,
                suffix="STFTTimeSeries"
            )
            
            # Set time-series specific properties
            timeseries_channel.tags = ["time-series"]
            timeseries_channel.xlabel = "Time (s)"
            timeseries_channel.ylabel = ylabel
            timeseries_channel.legend_label = f"{channel.legend_label} - {params.get('reduction', 'max_intensity').replace('_', ' ').title()}"
            
            return [spectrogram_channel, timeseries_channel]
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"STFT spectrogram computation failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> tuple:
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
        
        return Pxx, reduced_data, t_aligned, f, ylabel
