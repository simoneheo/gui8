import numpy as np
import heartpy as hp
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class ppg_rr_intervals_step(BaseStep):
    name = "ppg_rr_intervals"
    category = "HeartPy"
    description = """Extract RR intervals from raw PPG using HeartPy processing.
    
This step extracts R-R intervals from PPG signals using peak detection:
• **Peak detection**: Identifies heart beats from PPG signal
• **RR interval calculation**: Computes time between consecutive beats
• **Robust processing**: Handles various PPG signal qualities
• **Time series output**: Generates RR interval time series

• **Sampling rate**: Signal sampling frequency (automatically detected from channel)
• **Output**: RR interval time series (sample indices as x-axis)

Useful for:
• **HRV analysis**: Foundation for heart rate variability calculations
• **Beat-to-beat analysis**: Study cardiac rhythm patterns
• **Heart rate monitoring**: Track heart rate variations
• **Cardiac research**: Analyze heart rate dynamics"""
    
    tags = ["ppg", "rr", "heartpy", "intervals", "cardiac", "rhythm", "beats","time-series"]
    
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
        if len(y) < 1000:
            raise ValueError("PPG signal too short for RR interval extraction (minimum 1000 samples)")
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
            raise ValueError("Sampling frequency too low for PPG RR interval extraction (minimum 10 Hz)")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output RR intervals"""
        if len(y_new) == 0:
            raise ValueError("RR interval extraction produced empty result")
        if np.any(np.isnan(y_new)):
            raise ValueError("RR intervals contain NaN values")
        if np.any(y_new <= 0):
            raise ValueError("RR intervals contain non-positive values")

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
        """Apply PPG RR interval extraction to the channel data"""
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
                suffix="RRIntervals"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"PPG RR interval extraction failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> tuple:
        """Core PPG RR interval extraction logic"""
        
        # Process PPG signal with HeartPy
        wd, _ = hp.process(y, fs)
        
        # Calculate RR intervals
        rr = hp.calc_rr(wd['peaklist'], fs)
        
        # Create new x-axis (sample indices)
        x_new = np.arange(len(rr))
        
        return x_new, rr