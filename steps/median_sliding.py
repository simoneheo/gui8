import numpy as np
from steps.base_step import BaseStep
from steps.process_registry import register_step
from channel import Channel

@register_step
class median_sliding_step(BaseStep):
    name = "median_sliding"
    category = "Filter"
    description = """Compute median values over sliding sample windows to reduce noise and identify trends.
    
This step applies a sliding window approach where each window computes the median of its samples.
The result is a downsampled signal representing the median values at each window center.

• **Window size**: Number of samples in each sliding window
• **Overlap**: Number of overlapping samples between consecutive windows (0 = no overlap)

Useful for noise reduction while preserving signal characteristics better than mean-based smoothing."""
    tags = ["time-series", "smoothing", "noise-reduction", "median", "window", "sliding", "robust"]
    params = [
        {"name": "window", "type": "int", "default": "100", "help": "Window size in samples (must be >= 1)"},
        {"name": "overlap", "type": "int", "default": "50", "help": "Overlap in samples (must be < window size)"}
    ]

    @classmethod
    def get_info(cls): 
        return f"{cls.name} — Compute median values over sliding windows (Category: {cls.category})"

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
        
        if window < 1:
            raise ValueError("Window size must be at least 1 sample")
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
        """Apply median sliding window processing to the channel data."""
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
                suffix="MedianSliding"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Median sliding processing failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray]:
        """Core processing logic for median sliding window"""
        window = params["window"]
        overlap = params["overlap"]
        step = window - overlap

        if window > len(y):
            raise ValueError(f"Window size ({window}) exceeds signal length ({len(y)})")

        if step < 1:
            raise ValueError(f"Invalid step size: {step}. Increase window or reduce overlap.")

        x_new = []
        y_new = []

        for start in range(0, len(y) - window + 1, step):
            end = start + window
            segment_y = y[start:end]
            segment_x = x[start:end]

            try:
                result = np.median(segment_y)
                if np.isnan(result) or np.isinf(result):
                    continue
                center_x = segment_x[len(segment_x) // 2]
                x_new.append(center_x)
                y_new.append(result)
            except Exception:
                continue

        if len(y_new) == 0:
            raise ValueError("No valid windows found")
        x_new = np.array(x_new)
        y_new = np.array(y_new) 
        return x_new, y_new
