import numpy as np
import heartpy as hp
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class ppg_hrv_features_step(BaseStep):
    name = "ppg_hrv_features"
    category = "HeartPy"
    description = """Compute HRV features from raw PPG using HeartPy processing.
    
This step extracts comprehensive heart rate variability features from PPG signals:
• **Peak detection**: Identifies heart beats from PPG signal
• **RR interval calculation**: Computes time between consecutive beats
• **HRV metrics**: Calculates time and frequency domain HRV measures
• **Metadata storage**: Stores HRV features in channel metadata

• **Sampling rate**: Signal sampling frequency (automatically detected from channel)
• **Output**: Original PPG signal with HRV features stored in metadata

Useful for:
• **HRV analysis**: Extract comprehensive heart rate variability metrics
• **Cardiac assessment**: Evaluate autonomic nervous system function
• **Research applications**: Generate HRV features for statistical analysis
• **Health monitoring**: Track cardiac health indicators"""
    
    tags = ["ppg", "hrv", "heartpy", "features", "variability", "cardiac", "metadata","time-series"]
    
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
            raise ValueError("PPG signal too short for HRV analysis (minimum 1000 samples)")
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
            raise ValueError("Sampling frequency too low for PPG HRV analysis (minimum 10 Hz)")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output PPG signal"""
        if len(y_new) == 0:
            raise ValueError("HRV feature extraction produced empty signal")
        if len(y_new) != len(y_original):
            raise ValueError("Output signal length differs from input signal length")

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
        """Apply PPG HRV feature extraction to the channel data"""
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
            x_new, y_new, metadata = cls.script(x, y, fs, params)
            
            # Validate output data
            cls._validate_output_data(y, y_new)
            
            # Create new channel with HRV features in metadata
            new_channel = cls.create_new_channel(
                parent=channel,
                xdata=x_new,
                ydata=y_new,
                params=params,
                suffix="HRVFeaturesStored"
            )
            
            # Add HRV features to metadata
            new_channel.metadata.update(metadata)
            
            return new_channel
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"PPG HRV feature extraction failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> tuple:
        """Core PPG HRV feature extraction logic"""
        
        # Process PPG signal with HeartPy
        wd, _ = hp.process(y, fs)
        
        # Calculate RR intervals
        rr = hp.calc_rr(wd['peaklist'], fs)
        
        # Calculate HRV metrics
        hrv_metrics = hp.hrv(rr, fs)
        
        # Store HRV features in metadata
        metadata = {
            'hrv_features': hrv_metrics,
            'rr_intervals': rr,
            'peak_list': wd['peaklist']
        }
        
        # Return original signal with HRV features stored in metadata
        return x, y, metadata