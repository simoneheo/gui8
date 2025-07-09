import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel
from scipy.signal import medfilt

@register_step
class median_smooth_step(BaseStep):
    name = "median_smooth"
    category = "Filter"
    description = """Apply median filter smoothing using overlapping sliding windows to reduce noise while preserving signal edges.
    
This step applies a median filter within overlapping windows, where each point may be part of multiple windows.
The results are averaged across all windows containing each point, providing robust noise reduction.

• **Window size**: Size of the median filter kernel (must be odd and >= 3)
• **Overlap**: Number of samples to overlap between windows (must be < window size)

Median filtering is particularly effective for removing impulsive noise while preserving signal edges better than mean-based smoothing."""
    tags = ["time-series", "smoothing", "noise-reduction", "median", "window", "robust", "filter"]
    params = [
        {"name": "window_size", "type": "int", "default": "5", "help": "Window size (odd integer >= 3)"},
        {"name": "overlap", "type": "int", "default": "2", "help": "Overlap in samples (must be less than window size)"}
    ]

    @classmethod
    def get_info(cls): 
        return f"{cls.name} — Apply median filter smoothing with overlapping windows (Category: {cls.category})"

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
        overlap = params.get("overlap")
        
        if window_size < 3 or window_size % 2 == 0:
            raise ValueError(f"Window size must be odd and >= 3, got {window_size}")
        if overlap < 0 or overlap >= window_size:
            raise ValueError(f"Overlap must be in [0, window_size), got {overlap}")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output signal data"""
        if len(y_new) != len(y_original):
            raise ValueError("Output signal length differs from input")
        if np.any(np.isnan(y_new)) and not np.any(np.isnan(y_original)):
            raise ValueError("Median smoothing produced unexpected NaN values")

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
        """Apply median smoothing to the channel data."""
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
                suffix="MedianSmooth"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Median smoothing failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic for median smoothing"""
        window_size = params["window_size"]
        overlap = params["overlap"]
        
        if len(y) < window_size:
            raise ValueError(f"Signal too short: length {len(y)} < window size {window_size}")

        step = window_size - overlap
        smoothed = np.zeros_like(y, dtype=float)
        counts = np.zeros_like(y, dtype=int)

        for start in range(0, len(y) - window_size + 1, step):
            end = start + window_size
            chunk = y[start:end]
            filtered = medfilt(chunk, kernel_size=window_size)
            smoothed[start:end] += filtered
            counts[start:end] += 1

        # Avoid divide by zero
        counts[counts == 0] = 1
        y_new = smoothed / counts
        return y_new
