import numpy as np
from scipy.signal import welch

from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class welch_spectrogram(BaseStep):
    name = "welch_spectrogram"
    category = "Spectrogram"
    description = """Compute a sliding Welch Power Spectral Density (PSD) over time windows.
    
This step computes Welch PSD estimates for overlapping time windows and outputs both:
1. A 2D spectrogram channel showing frequency content evolution over time.
2. A 1D time-series channel based on reduction method.

Reduction methods:
• **max_intensity**: Maximum PSD in each time window
• **sum_intensity**: Total PSD in each time window
• **mean_intensity**: Average PSD across frequencies

Useful for:
• **Time-frequency analysis**: Track frequency content changes over time
• **Power spectral analysis**: Estimate power distribution across frequencies
• **Signal characterization**: Identify dominant frequency components
• **Noise analysis**: Analyze frequency content of non-stationary signals"""
    tags = ["spectrogram", "time-frequency", "welch", "psd", "scipy", "frequency", "power", "window"]
    params = [
        {
            "name": "window_duration", 
            "type": "float", 
            "default": "2.0", 
            "help": "Duration of each Welch window in seconds (must be positive)"
        },
        {
            "name": "overlap", 
            "type": "int", 
            "default": "128", 
            "help": "Overlap in samples between windows (must be < window size)"
        },
        {
            "name": "nperseg", 
            "type": "int", 
            "default": "256", 
            "help": "Length of each Welch segment within window (must be positive)"
        },
        {
            "name": "reduction", 
            "type": "str", 
            "default": "max_intensity", 
            "options": ["max_intensity", "sum_intensity", "mean_intensity"], 
            "help": "How to reduce PSD into a 1D series"
        }
    ]

    @classmethod
    def get_info(cls): 
        return f"{cls.name} — Compute sliding Welch PSD spectrogram (Category: {cls.category})"

    @classmethod
    def get_prompt(cls): 
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input signal data"""
        if len(y) < 10:
            raise ValueError("Signal too short for Welch spectrogram (minimum 10 samples)")
        if np.all(np.isnan(y)):
            raise ValueError("Signal contains only NaN values")

    @classmethod
    def _validate_parameters(cls, params: dict, fs: float = 1.0) -> None:
        """Validate parameters and business rules"""
        window_duration = params.get("window_duration")
        overlap = params.get("overlap")
        nperseg = params.get("nperseg")
        
        if window_duration <= 0:
            raise ValueError("Window duration must be positive")
        
        window_samples = int(window_duration * fs)
        if overlap < 0 or overlap >= window_samples:
            raise ValueError("Overlap must be non-negative and less than window size")
        if nperseg <= 0:
            raise ValueError("nperseg must be positive")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, spectrogram_data: np.ndarray, timeseries_data: np.ndarray) -> None:
        """Validate output data"""
        if spectrogram_data.size == 0:
            raise ValueError("Welch computation produced empty spectrogram")
        if timeseries_data.size == 0:
            raise ValueError("Reduction method produced empty time-series")
        if np.any(np.isnan(spectrogram_data)) and not np.any(np.isnan(y_original)):
            raise ValueError("Welch computation produced unexpected NaN values")

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
        """Apply Welch spectrogram computation to the channel data."""
        try:
            x = channel.xdata
            y = channel.ydata
            fs = getattr(channel, 'fs', 1.0)
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params, fs)
            
            # Validate window duration vs signal length
            window_duration = params.get("window_duration", 2.0)
            window_samples = int(window_duration * fs)
            if window_samples > len(y):
                raise ValueError(f"Window duration ({window_duration}s) is larger than signal duration")
            
            # Process the data
            spectrogram_data, timeseries_data, time_centers, frequencies, ylabel = cls.script(x, y, fs, params)
            
            # Validate output data
            cls._validate_output_data(y, spectrogram_data, timeseries_data)
            
            # Create spectrogram channel
            spectrogram_channel = cls.create_new_channel(
                parent=channel,
                xdata=time_centers,
                ydata=frequencies,
                params=params,
                suffix="WelchSpectrogram"
            )
            
            # Set spectrogram-specific properties
            spectrogram_channel.tags = ["spectrogram"]
            spectrogram_channel.xlabel = "Time (s)"
            spectrogram_channel.ylabel = "Frequency (Hz)"
            spectrogram_channel.legend_label = f"{channel.legend_label} - Welch Spectrogram"
            
            # Store spectrogram data in metadata
            spectrogram_channel.metadata = {
                'Zxx': spectrogram_data,
                'colormap': 'viridis'
            }
            
            # Create time-series channel
            timeseries_channel = cls.create_new_channel(
                parent=channel,
                xdata=time_centers,
                ydata=timeseries_data,
                params=params,
                suffix="WelchTimeSeries"
            )
            
            # Set time-series specific properties
            timeseries_channel.tags = ["time-series"]
            timeseries_channel.xlabel = "Time (s)"
            timeseries_channel.ylabel = ylabel
            timeseries_channel.legend_label = f"{channel.legend_label} - Welch {params.get('reduction', 'max_intensity').replace('_', ' ').title()}"
            
            return [spectrogram_channel, timeseries_channel]
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Welch spectrogram computation failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> tuple:
        """Core processing logic for Welch spectrogram computation"""
        window_duration = params.get("window_duration", 2.0)
        overlap = params.get("overlap", 0.5)
        nperseg = params.get("nperseg", 256)
        reduction = params.get("reduction", "max_intensity")
        
        # Calculate window parameters
        window_samples = int(window_duration * fs)
        step_samples = window_samples - overlap
        
        # Adjust nperseg if needed
        if nperseg > window_samples:
            nperseg = window_samples // 4
        
        # Generate sliding windows
        window_starts = []
        current_sample = 0
        while current_sample + window_samples <= len(y):
            window_starts.append(current_sample)
            current_sample += step_samples
        
        if len(window_starts) == 0:
            raise ValueError("No valid windows found")
        
        # Compute Welch PSD for each window
        psd_matrix = []
        time_centers = []
        frequencies = None
        
        for start_idx in window_starts:
            end_idx = start_idx + window_samples
            window_data = y[start_idx:end_idx]
            
            # Compute Welch PSD for this window
            f, pxx = welch(window_data, fs=fs, nperseg=nperseg)
            
            if frequencies is None:
                frequencies = f
            
            psd_matrix.append(pxx)
            
            # Time at center of window
            center_sample = start_idx + window_samples // 2
            if x is not None and len(x) > 0:
                center_time = x[0] + (center_sample / fs) * (x[-1] - x[0]) / (len(y) / fs)
            else:
                center_time = center_sample / fs
            time_centers.append(center_time)
        
        # Convert to numpy arrays
        psd_matrix = np.array(psd_matrix).T  # Shape: (n_frequencies, n_time_windows)
        time_centers = np.array(time_centers)
        
        # Apply reduction method
        if reduction == "max_intensity":
            reduced_data = np.max(psd_matrix, axis=0)
            ylabel = "Max PSD"
        elif reduction == "sum_intensity":
            reduced_data = np.sum(psd_matrix, axis=0)
            ylabel = "Total PSD"
        elif reduction == "mean_intensity":
            reduced_data = np.mean(psd_matrix, axis=0)
            ylabel = "Mean PSD"
        else:
            raise ValueError(f"Unknown reduction method: {reduction}")
        
        return psd_matrix, reduced_data, time_centers, frequencies, ylabel 
