import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class rolling_mean_subtract_step(BaseStep):
    name = "rolling_mean_subtract"
    category = "Arithmetic"
    description = """Subtract rolling mean from signal for local baseline correction.
    
This step computes a rolling mean (moving average) of the signal and subtracts it
from the original signal to remove local trends and baseline drift. This is useful
for detrending signals while preserving local features.

• **Window size**: Size of the rolling mean window (odd integer >= 3)
• **Automatic adjustment**: Even window sizes are automatically incremented to odd

Useful for:
• **Baseline correction**: Remove slow-varying trends from signals
• **Detrending**: Eliminate drift in long-term recordings
• **Feature preservation**: Maintain local signal features while removing global trends
• **Signal conditioning**: Prepare signals for peak detection or analysis"""
    tags = ["time-series", "baseline-correction", "detrending", "window", "sliding", "mean", "subtract"]
    params = [
        {
            'name': 'window_size', 
            'type': 'int', 
            'default': '51', 
            'help': 'Window size for rolling mean (odd integer >= 3, larger = more smoothing)',
        }
    ]

    @classmethod
    def get_info(cls): 
        return f"{cls.name} — Subtract rolling mean for local baseline correction (Category: {cls.category})"

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
        window_size = params.get("window_size")
        
        if window_size < 3:
            raise ValueError(f"Window size must be >= 3, got {window_size}")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output signal data"""
        if len(y_new) != len(y_original):
            raise ValueError("Output signal length differs from input")
        if np.any(np.isnan(y_new)) and not np.any(np.isnan(y_original)):
            raise ValueError("Rolling mean subtraction produced unexpected NaN values")

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
                else:
                    parsed[name] = val
            except ValueError as e:
                if "could not convert" in str(e) or "invalid literal" in str(e):
                    raise ValueError(f"{name} must be a valid {param['type']}")
                raise e
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        """Apply rolling mean subtraction to the channel data."""
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
                suffix="RollingMeanSubtract"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Rolling mean subtraction failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic for rolling mean subtraction"""
        window_size = params.get("window_size", 51)
        
        # Make window size odd for symmetry
        if window_size % 2 == 0:
            window_size += 1
        
        from scipy.ndimage import uniform_filter1d
        baseline = uniform_filter1d(y, size=window_size, mode='nearest')
        y_new = y - baseline
        return y_new
