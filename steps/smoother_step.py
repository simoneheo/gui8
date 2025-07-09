import numpy as np

from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class smoother_step(BaseStep):
    name = "smoother"
    category = "BioSPPy"
    description = """Smooth signal using BioSPPy smoothing algorithms with configurable methods.
    
This step applies various smoothing algorithms to reduce noise and enhance signal quality:
• **Moving average**: Simple moving average filter for basic smoothing
• **Exponential**: Exponential smoothing for adaptive noise reduction
• **Configurable window size**: Adjustable smoothing window for different signal characteristics

• **Method**: Smoothing algorithm (moving_average or exponential)
• **Size**: Window size for smoothing operation
• **Output**: Smoothed signal with reduced noise

Useful for:
• **Noise reduction**: Remove high-frequency noise from biosignals
• **Signal conditioning**: Prepare signals for further analysis
• **Trend extraction**: Extract underlying trends from noisy data
• **Artifact removal**: Reduce movement artifacts in biosignals"""
    tags = ["biosignal", "smoothing", "biosppy", "noise-reduction", "filter", "moving-average", "exponential","time-series"]
    params = [
        {
            "name": "method",
            "type": "str",
            "default": "moving_average",
            "options": ["moving_average", "exponential"],
            "help": "Smoothing method (moving_average or exponential)"
        },
        {
            "name": "size",
            "type": "int",
            "default": "10",
            "help": "Window size for smoothing (must be positive)"
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
        """Validate input signal data"""
        if len(y) == 0:
            raise ValueError("Input signal is empty")
        if len(y) < 10:
            raise ValueError("Signal too short for smoothing (minimum 10 samples)")
        if np.all(np.isnan(y)):
            raise ValueError("Signal contains only NaN values")
        if np.all(np.isinf(y)):
            raise ValueError("Signal contains only infinite values")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters"""
        method = params.get("method")
        size = params.get("size")
        
        if method not in ["moving_average", "exponential"]:
            raise ValueError("Method must be 'moving_average' or 'exponential'")
        if size <= 0:
            raise ValueError("Size must be positive")
        if size > 1000:
            raise ValueError("Size too large (maximum 1000)")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output signal"""
        if len(y_new) != len(y_original):
            raise ValueError("Output signal length differs from input")
        if np.any(np.isnan(y_new)) and not np.any(np.isnan(y_original)):
            raise ValueError("Smoothing produced unexpected NaN values")

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        """Parse and validate user input parameters"""
        parsed = {}
        for param in cls.params:
            name = param["name"]
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
        """Apply smoothing to the channel data"""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            
            # Process the data
            y_new = cls.script(x, y, None, params)
            
            # Validate output data
            cls._validate_output_data(y, y_new)
            
            return cls.create_new_channel(
                parent=channel,
                xdata=x,
                ydata=y_new,
                params=params,
                suffix="Smoothed"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"BioSPPy smoothing failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> np.ndarray:
        """Core processing logic for smoothing"""
        from biosppy.tools import smoother
        method = params.get("method", "moving_average")
        size = int(params.get("size", 10))
        
        # Apply BioSPPy smoothing
        y_new = smoother(signal=y, method=method, size=size)
        
        return y_new