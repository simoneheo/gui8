import numpy as np
from biosppy.signals import ecg
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class ecg_processing_step(BaseStep):
    name = "ecg_processing"
    category = "BioSPPy"
    description = "Apply BioSPPy ECG processing pipeline for R-peak detection and heart rate estimation"
    tags = ["time-series","biosignal", "ecg", "biosppy", "heart-rate", "filtering", "r-peaks", "hrv"]
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
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input ECG signal data"""
        if len(y) == 0:
            raise ValueError("Input ECG signal is empty")
        if len(y) < 500:
            raise ValueError("ECG signal too short for heart rate analysis (minimum 500 samples)")
        if np.all(np.isnan(y)):
            raise ValueError("ECG signal contains only NaN values")
        if np.all(np.isinf(y)):
            raise ValueError("ECG signal contains only infinite values")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters"""
        fs = params.get("fs")
        if fs is not None and fs <= 0:
            raise ValueError("Sampling frequency must be positive")
        if fs is not None and fs < 100:
            raise ValueError("Sampling frequency too low for ECG analysis (minimum 100 Hz)")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output heart rate signal"""
        if len(y_new) == 0:
            raise ValueError("ECG processing produced empty heart rate signal")
        if np.any(np.isnan(y_new)) and not np.any(np.isnan(y_original)):
            raise ValueError("ECG processing produced unexpected NaN values")
        if np.any(y_new < 0):
            raise ValueError("Heart rate values should be non-negative")
        if np.any(y_new > 300):
            raise ValueError("Heart rate values seem unrealistic (>300 bpm)")

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
        """Apply ECG processing pipeline to the channel data"""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Get sampling frequency from channel
            fs = cls._get_channel_fs(channel)
            if fs is None:
                fs = 1000.0  # Default ECG sampling rate
            
            # Inject sampling frequency into params
            params["fs"] = fs
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            
            # Process the data
            hr_times, hr_values = cls.script(x, y, fs, params)
            
            # Validate output data
            cls._validate_output_data(y, hr_values)
            
            return cls.create_new_channel(
                parent=channel,
                xdata=hr_times,
                ydata=hr_values,
                params=params,
                suffix="ECG_HR"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"ECG processing failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> tuple:
        """Core processing logic for ECG analysis"""
        try:
            # Apply BioSPPy ECG processing pipeline
            output = ecg.ecg(signal=y, sampling_rate=fs, show=False)
            
            # Extract heart rate time series
            hr_times = output['heart_rate_ts'][0]
            hr_values = output['heart_rate_ts'][1]
            
            x_new = hr_times
            y_new = hr_values
            return x_new, y_new
            
        except Exception as e:
            raise ValueError(f"BioSPPy ECG processing failed: {str(e)}")