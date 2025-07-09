import numpy as np
from scipy.ndimage import uniform_filter1d
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class smooth_normalize_step(BaseStep):
    name = "smooth_normalize"
    category = "Transform"
    description = """Normalize signal using local mean and range with uniform filtering.
    
This step applies local normalization using a sliding window approach with uniform filtering.
The signal is normalized by subtracting the local mean and dividing by the local range,
then optionally scaled to [0,1] range.

• **Window size**: Size of the smoothing window (odd number for symmetry)
• **Scale to [0,1]**: Whether to rescale the normalized signal to [0,1] range

Useful for:
• **Local normalization**: Remove local trends and variations
• **Signal standardization**: Make signals comparable across different scales
• **Feature extraction**: Prepare signals for pattern recognition
• **Noise reduction**: Reduce local variations while preserving global structure"""
    tags = ["time-series", "normalization", "smoothing", "minmax", "scaling", "range"]
    params = [
        {"name": "window", "type": "int", "default": "101", "help": "Window size for smoothing (odd number >= 3)"},
        {"name": "scale_0_1", "type": "bool", "default": "True", "help": "Rescale normalized signal to [0,1]"}
    ]

    @classmethod
    def get_info(cls):
        return f"{cls.name} — Normalize signal using local mean and range (Category: {cls.category})"

    @classmethod
    def get_prompt(cls):
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input signal data"""
        if len(y) == 0:
            raise ValueError("Input signal is empty")
        if np.all(np.isnan(y)):
            raise ValueError("Signal contains only NaN values")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        window = params.get("window")
        
        if window < 3:
            raise ValueError("Window size must be at least 3")
        if window % 2 == 0:
            raise ValueError("Window size must be odd for symmetry")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output signal data"""
        if len(y_new) != len(y_original):
            raise ValueError("Output signal length differs from input")
        if np.any(np.isnan(y_new)) and not np.any(np.isnan(y_original)):
            raise ValueError("Normalization produced unexpected NaN values")

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
                elif param["type"] == "int":
                    parsed[name] = int(val)
                elif param["type"] == "float":
                    parsed[name] = float(val)
                elif param["type"] == "bool":
                    if isinstance(val, str):
                        parsed[name] = val.lower() in ["true", "1", "yes"]
                    else:
                        parsed[name] = bool(val)
                else:
                    parsed[name] = val
            except ValueError as e:
                if "could not convert" in str(e) or "invalid literal" in str(e):
                    raise ValueError(f"{name} must be a valid {param['type']}")
                raise e
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        """Apply smooth normalization to the channel data."""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            
            # Process the data
            y_new = cls.script(x, y, params)
            
            # Validate output data
            cls._validate_output_data(y, y_new)
            
            return cls.create_new_channel(
                parent=channel, 
                xdata=x, 
                ydata=y_new, 
                params=params,
                suffix="SmoothNorm"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Smooth normalization failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic for smooth normalization"""
        window = params["window"]
        scale_0_1 = params["scale_0_1"]

        if len(y) < window:
            raise ValueError("Signal shorter than window size")

        # Smooth local mean and absolute deviation
        local_mean = uniform_filter1d(y, size=window, mode='nearest')
        local_range = uniform_filter1d(np.abs(y - local_mean), size=window, mode='nearest')

        # Avoid divide-by-zero
        local_range[local_range == 0] = 1e-8

        y_norm = (y - local_mean) / local_range

        if scale_0_1:
            y_min, y_max = np.min(y_norm), np.max(y_norm)
            if y_max - y_min == 0:
                y_norm[:] = 0
            else:
                y_norm = (y_norm - y_min) / (y_max - y_min)

        y_new = y_norm  

        return y_new
