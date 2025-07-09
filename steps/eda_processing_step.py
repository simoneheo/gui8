import numpy as np
from biosppy.signals import eda
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class eda_processing_step(BaseStep):
    name = "eda_processing"
    category = "BioSPPy"
    description = "Apply BioSPPy EDA processing pipeline for electrodermal activity filtering and analysis"
    tags = ["time-series","biosignal", "eda", "biosppy", "skin-conductance", "arousal", "stress", "autonomic"]
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
        """Validate input EDA signal data"""
        if len(y) == 0:
            raise ValueError("Input EDA signal is empty")
        if len(y) < 100:
            raise ValueError("EDA signal too short for processing (minimum 100 samples)")
        if np.all(np.isnan(y)):
            raise ValueError("EDA signal contains only NaN values")
        if np.all(np.isinf(y)):
            raise ValueError("EDA signal contains only infinite values")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters"""
        fs = params.get("fs")
        if fs is not None and fs <= 0:
            raise ValueError("Sampling frequency must be positive")
        if fs is not None and fs < 10:
            raise ValueError("Sampling frequency too low for EDA analysis (minimum 10 Hz)")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output EDA signal"""
        if len(y_new) == 0:
            raise ValueError("EDA processing produced empty signal")
        if np.any(np.isnan(y_new)) and not np.any(np.isnan(y_original)):
            raise ValueError("EDA processing produced unexpected NaN values")

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
        """Apply EDA processing pipeline to the channel data"""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Get sampling frequency from channel
            fs = cls._get_channel_fs(channel)
            if fs is None:
                fs = 100.0  # Default EDA sampling rate
            
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
                suffix="EDA_Filtered"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"EDA processing failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> tuple:
        """Core processing logic for EDA analysis"""
        try:
            # Apply BioSPPy EDA processing pipeline
            output = eda.eda(signal=y, sampling_rate=fs, show=False)
            
            # Extract filtered EDA signal
            y_new = output['filtered']
            
            # Create new time axis
            x_new = np.linspace(0, len(y_new)/fs, len(y_new))
            
            return x_new, y_new
            
        except Exception as e:
            raise ValueError(f"BioSPPy EDA processing failed: {str(e)}")