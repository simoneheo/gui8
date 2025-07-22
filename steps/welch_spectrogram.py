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
            "help": "Reduction method for producing 1D time-series"
        }
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        window_duration = cls.validate_numeric_parameter("window_duration", params.get("window_duration"), min_val=0.1)
        overlap = cls.validate_integer_parameter("overlap", params.get("overlap"), min_val=0)
        nperseg = cls.validate_integer_parameter("nperseg", params.get("nperseg"), min_val=4)
        reduction = cls.validate_string_parameter("reduction", params.get("reduction"), 
                                                valid_options=["max_intensity", "sum_intensity", "mean_intensity"])

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        """Core processing logic for Welch spectrogram computation"""
        window_duration = params.get("window_duration", 2.0)
        overlap = params.get("overlap", 128)
        nperseg = params.get("nperseg", 256)
        reduction = params.get("reduction", "max_intensity")
        
        # Calculate window parameters
        window_samples = int(window_duration * fs)
        step_samples = window_samples - overlap
        
        if step_samples <= 0:
            raise ValueError("Overlap must be less than window size")
        
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
        elif reduction == "sum_intensity":
            reduced_data = np.sum(psd_matrix, axis=0)
        elif reduction == "mean_intensity":
            reduced_data = np.mean(psd_matrix, axis=0)
        else:
            raise ValueError(f"Unknown reduction method: {reduction}")
        
        return [
            {
                'tags': ['time-series'],
                'x': time_centers,
                'y': reduced_data
            },
            {
                'tags': ['spectrogram'],
                't': time_centers,
                'f': frequencies,
                'z': psd_matrix
            }
        ] 
