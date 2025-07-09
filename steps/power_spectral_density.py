import numpy as np
from scipy import signal
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class power_spectral_density(BaseStep):
    name = "power_spectral_density"
    category = "Bar Chart"
    description = """Compute Power Spectral Density (PSD) creating both time-series and bar-chart outputs.
    
This step computes the power spectral density of the input signal using Welch's method:
• **PSD computation**: Uses Welch's method with overlapping windows
• **Dual output**: Creates both time-series and bar-chart channels
• **Configurable windowing**: Adjustable window size and overlap in samples
• **Frequency analysis**: Analyzes power distribution across frequencies

• **Window**: Window size in samples for PSD computation
• **Overlap**: Overlap between windows in samples
• **Sampling rate**: Signal sampling frequency (automatically detected from channel)

**Outputs:**
• **Time-series channel**: Average power over time using sliding windows
• **Bar-chart channel**: Power spectral density vs frequency

Useful for:
• **Frequency domain analysis**: Understand signal frequency content
• **Spectral characterization**: Identify dominant frequencies
• **Signal quality assessment**: Detect noise and artifacts
• **Comparative analysis**: Compare spectral properties between signals"""
    
    tags = ["spectral", "frequency", "psd", "welch", "analysis"]
    
    params = [
        {
            "name": "window",
            "type": "int",
            "default": 1024,
            "description": "Window size in samples for PSD computation",
            "help": "Larger windows provide better frequency resolution but worse time resolution"
        },
        {
            "name": "overlap",
            "type": "int", 
            "default": 512,
            "description": "Overlap between windows in samples",
            "help": "Typical overlap is 50% of window size for smooth estimation"
        },
        {
            "name": "fs",
            "type": "float",
            "default": "",
            "help": "Sampling frequency (Hz) - automatically detected from channel if not provided"
        }
    ]

    @classmethod
    def get_info(cls):
        return f"{cls.name} — {cls.description.split('.')[0]} (Category: {cls.category})"

    @classmethod
    def get_prompt(cls):
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def _get_channel_fs(cls, channel: Channel) -> float:
        """Get sampling frequency from channel"""
        if hasattr(channel, 'fs_median') and channel.fs_median:
            return float(channel.fs_median)
        elif hasattr(channel, 'fs') and channel.fs:
            return float(channel.fs)
        else:
            return 1000.0  # Default fallback

    @classmethod
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input signal data"""
        if len(y) == 0:
            raise ValueError("Input signal is empty")
        if len(y) < 64:
            raise ValueError("Signal too short for PSD computation (minimum 64 samples)")
        if np.all(np.isnan(y)):
            raise ValueError("Signal contains only NaN values")
        if np.all(np.isinf(y)):
            raise ValueError("Signal contains only infinite values")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate PSD parameters"""
        window = params.get("window", 1024)
        overlap = params.get("overlap", 512)
        fs = params.get("fs")
        
        if not isinstance(window, int) or window <= 0:
            raise ValueError("Window size must be a positive integer")
        if window < 16:
            raise ValueError("Window size too small (minimum 16 samples)")
        if window > 16384:
            raise ValueError("Window size too large (maximum 16384 samples)")
            
        if not isinstance(overlap, int) or overlap < 0:
            raise ValueError("Overlap must be a non-negative integer")
        if overlap >= window:
            raise ValueError("Overlap must be less than window size")
            
        if fs is not None and (not isinstance(fs, (int, float)) or fs <= 0):
            raise ValueError("Sampling frequency must be a positive number")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, freqs: np.ndarray, psd: np.ndarray, time_power: np.ndarray) -> None:
        """Validate output PSD data"""
        if len(freqs) == 0 or len(psd) == 0:
            raise ValueError("PSD computation produced empty results")
        if len(freqs) != len(psd):
            raise ValueError("Frequency and PSD arrays have different lengths")
        if np.any(np.isnan(psd)) or np.any(np.isinf(psd)):
            raise ValueError("PSD contains NaN or infinite values")
        if np.any(psd < 0):
            raise ValueError("PSD contains negative values")
        if len(time_power) == 0:
            raise ValueError("Time-series power computation produced empty results")
        if np.any(np.isnan(time_power)) or np.any(np.isinf(time_power)):
            raise ValueError("Time-series power contains NaN or infinite values")

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        """Parse and validate user input parameters"""
        parsed = {}
        for param in cls.params:
            name = param["name"]
            if name == "fs":
                continue  # Skip fs as it's injected from channel
            val = user_input.get(name, param.get("default"))
            try:
                if val == "":
                    parsed[name] = param.get("default")
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
        """Apply PSD computation to the channel data and return both channels"""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Get sampling frequency from channel
            fs = cls._get_channel_fs(channel)
            if fs is None:
                fs = 1000.0  # Default sampling rate
            
            # Inject sampling frequency into params
            params["fs"] = fs
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            
            # Process the data
            freqs, psd, time_points, time_power = cls.script(x, y, fs, params)
            
            # Validate output data
            cls._validate_output_data(y, freqs, psd, time_power)
            
            # Create bar-chart channel for frequency domain PSD
            bar_chart_channel = cls.create_new_channel(
                parent=channel,
                xdata=freqs,
                ydata=psd,
                params=params,
                suffix="PSD_Freq"
            )
            # Add bar-chart tag and axis labels
            bar_chart_channel.tags = ["bar-chart", "frequency", "psd"]
            bar_chart_channel.xlabel = "Frequency (Hz)"
            bar_chart_channel.ylabel = "Power Spectral Density"
            
            # Create time-series channel for power over time
            time_series_channel = cls.create_new_channel(
                parent=channel,
                xdata=time_points,
                ydata=time_power,
                params=params,
                suffix="PSD_Time"
            )
            # Add time-series tag and axis labels
            time_series_channel.tags = ["time-series", "power", "temporal"]
            time_series_channel.xlabel = "Time (s)" if hasattr(channel, 'xlabel') and 's' in str(channel.xlabel) else "Time"
            time_series_channel.ylabel = "Average Power"
            
            return [bar_chart_channel, time_series_channel]
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Power spectral density computation failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> tuple:
        """Core PSD computation logic"""
        
        # Get parameters
        window_size = int(params.get('window', 1024))
        overlap = int(params.get('overlap', 512))
        
        # Ensure window size doesn't exceed signal length
        if window_size > len(y):
            window_size = len(y) // 2
            overlap = min(overlap, window_size // 2)
        
        # Compute PSD using Welch's method
        freqs, psd = signal.welch(y, fs=fs, window='hann', nperseg=window_size, 
                                  noverlap=overlap, scaling='density')
        
        # Compute time-series power using sliding windows
        hop_size = window_size - overlap
        num_windows = (len(y) - window_size) // hop_size + 1
        
        time_power = np.zeros(num_windows)
        time_points = np.zeros(num_windows)
        
        # Calculate time offset from original data
        time_offset = 0.0
        if len(x) > 0:
            # If x data exists, use the time of the first sample as offset
            time_offset = x[0]
        
        for i in range(num_windows):
            start_idx = i * hop_size
            end_idx = start_idx + window_size
            
            # Extract window
            window_data = y[start_idx:end_idx]
            
            # Compute average power for this window
            time_power[i] = np.mean(window_data ** 2)
            
            # Time point is center of window
            center_sample = start_idx + window_size // 2
            if len(x) > center_sample:
                # Use original time data (preserves offset naturally)
                time_points[i] = x[center_sample]
            else:
                # Fallback: Calculate time with preserved offset
                time_points[i] = (center_sample / fs) + time_offset
        
        return freqs, psd, time_points, time_power 