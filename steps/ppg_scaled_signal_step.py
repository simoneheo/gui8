import numpy as np
import heartpy as hp
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class ppg_scaled_signal_step(BaseStep):
    name = "ppg_scaled_signal"
    category = "HeartPy"
    description = """Scale PPG signal to standard range using HeartPy preprocessing.
    
This step applies HeartPy's signal scaling to normalize PPG signals:
• **Signal scaling**: Normalizes PPG signal to standard range
• **Amplitude normalization**: Removes amplitude variations between recordings
• **Preprocessing**: Prepares signals for consistent analysis
• **Automatic scaling**: Uses HeartPy's optimized scaling algorithm

• **Sampling rate**: Signal sampling frequency (automatically detected from channel)
• **Output**: Scaled PPG signal with normalized amplitude

Useful for:
• **Signal normalization**: Standardize PPG signals from different sources
• **Amplitude correction**: Remove recording-specific amplitude variations
• **Cross-session comparison**: Enable comparison between different recordings
• **Preprocessing**: Prepare signals for machine learning applications"""
    
    tags = ["ppg", "scaling", "heartpy", "normalization", "preprocessing", "amplitude","time-series"]
    
    params = [
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
            return 100.0  # Default fallback

    @classmethod
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input PPG signal data"""
        if len(y) == 0:
            raise ValueError("Input PPG signal is empty")
        if len(y) < 10:
            raise ValueError("PPG signal too short for scaling (minimum 10 samples)")
        if np.all(np.isnan(y)):
            raise ValueError("PPG signal contains only NaN values")
        if np.all(np.isinf(y)):
            raise ValueError("PPG signal contains only infinite values")
        if np.std(y) == 0:
            raise ValueError("PPG signal has zero variance, cannot scale")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters"""
        fs = params.get("fs")
        if fs is not None and fs <= 0:
            raise ValueError("Sampling frequency must be positive")
        if fs is not None and fs < 1:
            raise ValueError("Sampling frequency too low for PPG scaling (minimum 1 Hz)")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output PPG signal"""
        if len(y_new) == 0:
            raise ValueError("Signal scaling produced empty signal")
        if len(y_new) != len(y_original):
            raise ValueError("Output signal length differs from input signal length")
        if np.any(np.isnan(y_new)) and not np.any(np.isnan(y_original)):
            raise ValueError("Signal scaling produced unexpected NaN values")

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
        """Apply PPG signal scaling to the channel data"""
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
                suffix="Scaled"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"PPG signal scaling failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> tuple:
        """Core PPG signal scaling logic"""
        
        # Apply HeartPy signal scaling
        scaled = hp.scale_data(y)
        
        # Return original x-axis and scaled signal
        return x, scaled