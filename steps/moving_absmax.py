import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class moving_absmax_step(BaseStep):
    name = "moving_absmax"
    category = "Transform"
    description = """Compute maximum absolute values in overlapping sliding windows to create signal envelope.
    
This step applies a sliding window approach where each window computes the maximum absolute value
of its samples. The result is a downsampled signal representing the envelope of the original signal.

• **Window size**: Number of samples in each sliding window
• **Overlap**: Number of samples to overlap between windows

Useful for:
• **Signal envelope detection**: Identify the overall amplitude envelope
• **Peak tracking**: Track the maximum amplitudes over time
• **Noise floor estimation**: Understand signal dynamics across windows
• **Feature extraction**: Extract amplitude-based features from time series"""
    tags = ["time-series", "envelope", "feature-extraction","window","sliding","max","abs"]
    params = [
        {"name": "window", "type": "int", "default": "25", "help": "Window size in samples (must be > 0)"},
        {"name": "overlap", "type": "int", "default": "0", "help": "Overlap between windows in samples (must be < window)"},
    ]

    @classmethod
    def get_info(cls):
        return f"{cls.name} — Compute maximum absolute values in sliding windows (Category: {cls.category})"

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
        overlap = params.get("overlap")
        
        if window <= 0:
            raise ValueError("Window size must be > 0")
        if overlap < 0 or overlap >= window:
            raise ValueError("Overlap must be non-negative and less than window size")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, x_new: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output signal data"""
        if len(x_new) == 0 or len(y_new) == 0:
            raise ValueError("No valid windows found - output is empty")
        if len(x_new) != len(y_new):
            raise ValueError("Output x and y arrays have different lengths")

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
        """Apply moving absolute maximum processing to the channel data."""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            
            # Process the data
            x_new, y_new = cls.script(x, y, params)
            
            # Validate output data
            cls._validate_output_data(y, x_new, y_new)
            
            return cls.create_new_channel(
                parent=channel,
                xdata=x_new,
                ydata=y_new,
                params=params,
                suffix="MovingAbsMax"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Moving absolute maximum processing failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray]:
        """Core processing logic for moving absolute maximum"""
        window = params["window"]
        overlap = params["overlap"]
        step = window - overlap

        absmax_values = []
        time_stamps = []

        for start in range(0, len(y) - window + 1, step):
            end = start + window
            max_val = np.max(np.abs(y[start:end]))
            center_time = x[start + window // 2]
            absmax_values.append(max_val)
            time_stamps.append(center_time)

        x_new = np.array(time_stamps)   
        y_new = np.array(absmax_values)

        return x_new, y_new
