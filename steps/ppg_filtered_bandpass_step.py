import numpy as np
import heartpy as hp
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class ppg_filtered_bandpass_step(BaseStep):
    name = "ppg_filtered_bandpass"
    category = "HeartPy"
    description = """Apply HeartPy bandpass filter to PPG signal for noise reduction.
    
This step applies HeartPy's bandpass filtering to PPG signals:
• **Bandpass filtering**: Removes low and high frequency noise
• **Optimized for PPG**: Filter parameters optimized for physiological signals
• **Automatic processing**: Uses HeartPy's default filter settings
• **Signal preservation**: Maintains important cardiac information

• **Sampling rate**: Signal sampling frequency (automatically detected from channel)
• **Output**: Bandpass filtered PPG signal

Useful for:
• **Noise reduction**: Remove electrical and motion artifacts
• **Signal conditioning**: Prepare PPG for peak detection
• **Quality enhancement**: Improve signal-to-noise ratio
• **Preprocessing**: Clean signals before heart rate analysis"""
    
    tags = ["ppg", "filtering", "heartpy", "bandpass", "noise", "preprocessing","time-series"]
    
    params = [
        {
            "name": "fs",
            "type": "float",
            "default": "",
            "help": "Sampling frequency (Hz) - automatically detected from channel if not provided"
        },
        {
            "name": "cutoff",
            "type": "float",
            "default": "0.5",
            "help": "Cutoff frequency for bandpass filter (Hz) - typical range 0.5-5.0 Hz for PPG"
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
            return 100.0  # Default fallback

    @classmethod
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input PPG signal data"""
        if len(y) == 0:
            raise ValueError("Input PPG signal is empty")
        if len(y) < 100:
            raise ValueError("PPG signal too short for filtering (minimum 100 samples)")
        if np.all(np.isnan(y)):
            raise ValueError("PPG signal contains only NaN values")
        if np.all(np.isinf(y)):
            raise ValueError("PPG signal contains only infinite values")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters"""
        fs = params.get("fs")
        if fs is not None and fs <= 0:
            raise ValueError("Sampling frequency must be positive")
        if fs is not None and fs < 10:
            raise ValueError("Sampling frequency too low for PPG filtering (minimum 10 Hz)")
        
        cutoff = params.get("cutoff")
        if cutoff is not None:
            if cutoff <= 0:
                raise ValueError("Cutoff frequency must be positive")
            if cutoff >= fs / 2 if fs else 50:
                raise ValueError("Cutoff frequency must be less than Nyquist frequency (fs/2)")
            if cutoff < 0.1:
                raise ValueError("Cutoff frequency too low for PPG filtering (minimum 0.1 Hz)")
            if cutoff > 10.0:
                raise ValueError("Cutoff frequency too high for PPG filtering (maximum 10.0 Hz)")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output PPG signal"""
        if len(y_new) == 0:
            raise ValueError("Bandpass filtering produced empty signal")
        if np.any(np.isnan(y_new)) and not np.any(np.isnan(y_original)):
            raise ValueError("Bandpass filtering produced unexpected NaN values")

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
    def apply(cls, channel: Channel, params: dict) -> Channel:
        """Apply PPG bandpass filtering to the channel data"""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Get sampling frequency from channel
            fs = cls._get_channel_fs(channel)
            if fs is None:
                fs = 100.0  # Default PPG sampling rate
            
            # Inject sampling frequency into params
            params["fs"] = fs
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            
            # Process the data
            x_new, y_new = cls.script(x, y, fs, params)
            
            # Validate output data
            cls._validate_output_data(y, y_new)
            
            return cls.create_new_channel(
                parent=channel,
                xdata=x_new,
                ydata=y_new,
                params=params,
                suffix="BandpassFiltered"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"PPG bandpass filtering failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> tuple:
        """Core PPG bandpass filtering logic"""
        
        # Get cutoff frequency from parameters
        cutoff = params.get("cutoff", 0.5)
        
        # Apply HeartPy bandpass filter with cutoff and sample rate
        filtered = hp.filter_signal(y, cutoff=cutoff, sample_rate=fs)
        
        # Return original x-axis and filtered signal
        return x, filtered