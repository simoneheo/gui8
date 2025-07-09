import numpy as np
from biosppy.signals.ecg import hamilton_segmenter
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class hamilton_segmenter_step(BaseStep):
    name = "hamilton_segmenter"
    category = "BioSPPy"
    description = """Detect QRS complexes in ECG signal using BioSPPy's Hamilton segmenter algorithm.
    
This step applies the Hamilton QRS detection algorithm to identify R-peaks in ECG signals.
The algorithm is optimized for ECG signals and provides robust QRS complex detection.

• **Sampling rate**: Signal sampling frequency (automatically detected from channel)
• **Output**: Binary marker signal with 1.0 at R-peak locations, 0.0 elsewhere

Useful for:
• **Heart rate analysis**: Extract R-R intervals for HRV analysis
• **ECG segmentation**: Identify cardiac cycles for beat-to-beat analysis
• **Arrhythmia detection**: Detect irregular heart rhythms
• **Signal quality assessment**: Evaluate ECG signal quality"""
    tags = ["time-series","biosignal", "ecg", "qrs", "biosppy", "peak-detection", "heart-rate", "r-peaks"]
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
        if len(y) < 100:
            raise ValueError("ECG signal too short for QRS detection (minimum 100 samples)")
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
        """Validate output marker signal"""
        if len(y_new) != len(y_original):
            raise ValueError("Output marker signal length differs from input")
        if np.any(np.isnan(y_new)) and not np.any(np.isnan(y_original)):
            raise ValueError("Hamilton segmenter produced unexpected NaN values")
        if not np.all(np.isin(y_new, [0.0, 1.0])):
            raise ValueError("Output should be binary marker signal (0.0 or 1.0)")

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
        """Apply Hamilton QRS detection to the channel data"""
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
            y_new = cls.script(x, y, fs, params)
            
            # Validate output data
            cls._validate_output_data(y, y_new)
            
            # Create new channel
            new_channel = cls.create_new_channel(
                parent=channel,
                xdata=x,
                ydata=y_new,
                params=params,
                suffix="HamiltonRPeaks"
            )
            
            # Set line style to None and marker to * for R-peak visualization
            new_channel.style = "None"
            new_channel.marker = "*"
            
            return new_channel
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Hamilton QRS detection failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> np.ndarray:
        """Core processing logic for Hamilton QRS detection"""
        try:
            # Apply Hamilton segmenter
            rpeaks = hamilton_segmenter(signal=y, sampling_rate=fs)[0]
            
            # Create binary marker signal
            rpeak_marker = np.zeros_like(y, dtype=float)
            
            # Ensure R-peak indices are within bounds
            valid_rpeaks = rpeaks[rpeaks < len(y)]
            rpeak_marker[valid_rpeaks] = 1.0
            
            y_new = rpeak_marker

            return y_new
            
        except Exception as e:
            raise ValueError(f"Hamilton segmenter processing failed: {str(e)}")