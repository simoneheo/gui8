import numpy as np
from scipy.signal import welch

from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class welch_spectrogram(BaseStep):
    name = "welch_spectrogram"
    category = "Spectrogram"
    description = """Computes a sliding Welch Power Spectral Density (PSD) over time windows.
Returns:
1. A 2D spectrogram channel (tag='spectrogram') showing frequency content evolution over time.
2. A 1D time-series channel (tag='time-series') based on reduction method."""
    tags = ["spectrogram"]
    params = [
        {"name": "window_duration", "type": "float", "default": "2.0", "help": "Duration of each Welch window in seconds"},
        {"name": "overlap", "type": "float", "default": "0.5", "help": "Overlap fraction between windows [0.0-0.9]"},
        {"name": "nperseg", "type": "int", "default": "256", "help": "Length of each Welch segment within window."},
        {"name": "reduction", "type": "str", "default": "max_intensity", "options": ["max_intensity", "sum_intensity", "mean_intensity"], "help": "How to reduce PSD into a 1D series."},
        {"name": "fs", "type": "float", "default": "", "help": "Sampling frequency (injected from parent channel)."}
    ]

    @classmethod
    def get_info(cls): return f"{cls.name} — {cls.description} (Category: {cls.category})"
    @classmethod
    def get_prompt(cls): return {"info": cls.description, "params": cls.params}

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
        if len(channel.ydata) < 10:
            raise ValueError("Signal too short for Welch spectrogram.")
        if np.all(np.isnan(channel.ydata)):
            raise ValueError("Signal contains only NaNs.")

        params = cls._inject_fs_if_needed(channel, params, welch)
        fs = params.get("fs", 1.0)
        window_duration = params.get("window_duration", 2.0)
        overlap = params.get("overlap", 0.5)
        nperseg = params.get("nperseg", 256)
        reduction = params.get("reduction", "max_intensity")

        # Validate parameters
        if window_duration <= 0:
            raise ValueError("Window duration must be positive")
        if not (0.0 <= overlap < 1.0):
            raise ValueError("Overlap must be between 0.0 and 0.9")

        # Calculate window parameters
        window_samples = int(window_duration * fs)
        step_samples = int(window_samples * (1 - overlap))
        
        if window_samples > len(channel.ydata):
            raise ValueError(f"Window duration ({window_duration}s) is larger than signal duration")

        # Adjust nperseg if needed
        if nperseg > window_samples:
            nperseg = window_samples // 4

        try:
            # Generate sliding windows
            window_starts = []
            current_sample = 0
            while current_sample + window_samples <= len(channel.ydata):
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
                window_data = channel.ydata[start_idx:end_idx]
                
                # Compute Welch PSD for this window
                f, pxx = welch(window_data, fs=fs, nperseg=nperseg)
                
                if frequencies is None:
                    frequencies = f
                
                psd_matrix.append(pxx)
                
                # Time at center of window
                center_sample = start_idx + window_samples // 2
                center_time = channel.xdata[0] + (center_sample / fs) * (channel.xdata[-1] - channel.xdata[0]) / (len(channel.ydata) / fs)
                time_centers.append(center_time)

            # Convert to numpy arrays
            psd_matrix = np.array(psd_matrix).T  # Shape: (n_frequencies, n_time_windows)
            time_centers = np.array(time_centers)

        except Exception as e:
            raise ValueError(f"Welch spectrogram computation failed: {str(e)}")

        # Create spectrogram channel
        spectrogram_channel = cls.create_new_channel(
            parent=channel,
            xdata=time_centers,  # Time axis
            ydata=frequencies,   # Frequency axis
            params=params
        )
        
        # Set spectrogram-specific properties
        spectrogram_channel.tags = ["spectrogram"]
        spectrogram_channel.xlabel = "Time (s)"
        spectrogram_channel.ylabel = "Frequency (Hz)"
        spectrogram_channel.legend_label = f"{channel.legend_label} - Welch Spectrogram"
        
        # Store spectrogram data in metadata
        spectrogram_channel.metadata = {
            'Zxx': psd_matrix,
            'colormap': 'viridis'
        }

        # Apply reduction method to create time-series channel
        try:
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
        except Exception as e:
            raise ValueError(f"Reduction method '{reduction}' failed: {str(e)}")

        # Create time-series channel
        timeseries_channel = cls.create_new_channel(
            parent=channel,
            xdata=time_centers,
            ydata=reduced_data,
            params=params
        )
        
        # Set time-series specific properties
        timeseries_channel.tags = ["time-series"]
        timeseries_channel.xlabel = "Time (s)"
        timeseries_channel.ylabel = ylabel
        timeseries_channel.legend_label = f"{channel.legend_label} - Welch {reduction.replace('_', ' ').title()}"
        
        return [spectrogram_channel, timeseries_channel] 
