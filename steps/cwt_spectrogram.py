
import numpy as np
import pywt

from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class cwt_spectrogram(BaseStep):
    name = "cwt_spectrogram"
    category = "Spectrogram"
    description = """Compute a spectrogram using Continuous Wavelet Transform (CWT) and output both:
1. A 2D spectrogram channel for visualizing frequency evolution using wavelets.
2. A 1D time-series channel summarizing the spectrogram with a reduction method.

Reduction methods:
• **max_intensity**: Maximum amplitude in each time slice
• **sum_intensity**: Total wavelet energy in each time slice
• **centroid_freq**: Weighted average scale index (interpretable as pseudo-frequency)

Useful for:
• **Time-frequency analysis**: Analyze frequency content with wavelet-based approach
• **Multi-scale analysis**: Examine signal features at different scales
• **Non-stationary signals**: Handle signals with varying frequency content
• **Feature detection**: Identify transient events and frequency modulations"""
    tags = ["spectrogram", "time-frequency", "wavelet", "cwt", "frequency", "pywt", "scales"]
    params = [
        {
            "name": "wavelet", 
            "type": "str", 
            "default": "morl", 
            "options": ["morl", "cmor", "mexh"], 
            "help": "Wavelet type to use for CWT"
        },
        {
            "name": "scales", 
            "type": "str", 
            "default": "1-64", 
            "help": "Scale range as 'min-max' (e.g., '1-64', min must be >= 1)"
        },
        {
            "name": "reduction", 
            "type": "str", 
            "default": "max_intensity", 
            "options": ["max_intensity", "sum_intensity", "centroid_freq"], 
            "help": "Reduction method for summarizing the CWT"
        }
    ]

    @classmethod
    def get_info(cls): 
        return f"{cls.name} — Compute CWT spectrogram with wavelet-based time-frequency analysis (Category: {cls.category})"

    @classmethod
    def get_prompt(cls): 
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input signal data"""
        if len(y) < 10:
            raise ValueError("Signal too short for CWT computation (minimum 10 samples)")
        if np.all(np.isnan(y)):
            raise ValueError("Signal contains only NaN values")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        scale_range = params.get("scales", "1-64")
        
        try:
            scale_parts = scale_range.split("-")
            if len(scale_parts) != 2:
                raise ValueError("Scale range format should be 'min-max' (e.g., '1-64')")
            s_min, s_max = int(scale_parts[0]), int(scale_parts[1])
            if s_min < 1:
                raise ValueError("Minimum scale must be >= 1")
            if s_max <= s_min:
                raise ValueError("Maximum scale must be greater than minimum scale")
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid scale range '{scale_range}': {str(e)}")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, spectrogram_data: np.ndarray, timeseries_data: np.ndarray) -> None:
        """Validate output data"""
        if spectrogram_data.size == 0:
            raise ValueError("CWT computation produced empty spectrogram")
        if timeseries_data.size == 0:
            raise ValueError("Reduction method produced empty time-series")
        if np.any(np.isnan(spectrogram_data)) and not np.any(np.isnan(y_original)):
            raise ValueError("CWT computation produced unexpected NaN values")

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
        """Apply CWT spectrogram computation to the channel data."""
        try:
            x = channel.xdata
            y = channel.ydata
            fs = getattr(channel, 'fs', 1.0)
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            
            # Process the data
            spectrogram_data, timeseries_data, t, freqs, ylabel = cls.script(x, y, fs, params)
            
            # Validate output data
            cls._validate_output_data(y, spectrogram_data, timeseries_data)
            
            # Create spectrogram channel
            spectrogram_channel = cls.create_new_channel(
                parent=channel,
                xdata=t,
                ydata=freqs,
                params=params,
                suffix="CWTSpectrogram"
            )
            
            # Set spectrogram-specific properties
            spectrogram_channel.tags = ["spectrogram"]
            spectrogram_channel.xlabel = "Time (s)"
            spectrogram_channel.ylabel = "Scale (pseudo-freq)"
            spectrogram_channel.legend_label = f"{channel.legend_label} - CWT Spectrogram"
            spectrogram_channel.metadata = {
                "Zxx": spectrogram_data, 
                "colormap": "plasma"
            }
            
            # Create time-series channel
            timeseries_channel = cls.create_new_channel(
                parent=channel,
                xdata=t,
                ydata=timeseries_data,
                params=params,
                suffix="CWTTimeSeries"
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
                raise ValueError(f"CWT spectrogram computation failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> tuple:
        """Core processing logic for CWT spectrogram computation"""
        wavelet = params.get("wavelet", "morl")
        scale_range = params.get("scales", "1-64")
        reduction = params.get("reduction", "max_intensity")
        
        # Parse scale range
        scale_parts = scale_range.split("-")
        s_min, s_max = int(scale_parts[0]), int(scale_parts[1])
        scales = np.arange(s_min, s_max + 1)
        
        # Compute CWT
        coeffs, freqs = pywt.cwt(y, scales, wavelet, sampling_period=1/fs)
        power = np.abs(coeffs)
        
        # Create time axis
        if x is not None and len(x) > 0:
            t = np.linspace(x[0], x[-1], power.shape[1])
        else:
            t = np.linspace(0, len(y) / fs, power.shape[1])
        
        # Apply reduction method
        if reduction == "max_intensity":
            reduced_data = np.max(power, axis=0)
            ylabel = "Max Amplitude"
        elif reduction == "sum_intensity":
            reduced_data = np.sum(power, axis=0)
            ylabel = "Total Amplitude"
        elif reduction == "centroid_freq":
            norm = np.sum(power, axis=0)
            norm[norm == 0] = 1e-10
            reduced_data = np.sum(freqs[:, None] * power, axis=0) / norm
            ylabel = "Spectral Centroid (pseudo-freq)"
        else:
            raise ValueError(f"Unknown reduction method: {reduction}")
        
        return power, reduced_data, t, freqs, ylabel
