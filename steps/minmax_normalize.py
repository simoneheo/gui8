import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class minmax_normalize_step(BaseStep):
    name = "minmax_normalize"
    category = "General"
    description = """Normalize signal to a specific range using overlapping sliding windows.
    
This step applies min-max normalization within sliding windows, scaling each window's values
to the specified range [range_min, range_max]. The results are averaged across overlapping windows.

• **Range min/max**: Target range for normalized values
• **Window size**: Number of samples in each normalization window
• **Overlap**: Number of samples to overlap between windows

Useful for adaptive normalization that adjusts to local signal characteristics rather than global statistics."""
    tags = ["time-series", "normalization", "scaling", "window", "sliding", "minmax", "range"]
    params = [
        {"name": "range_min", "type": "float", "default": "0.0", "help": "Target minimum value of normalized signal"},
        {"name": "range_max", "type": "float", "default": "1.0", "help": "Target maximum value of normalized signal"},
        {"name": "window", "type": "int", "default": "100", "help": "Window size in samples (must be > 0)"},
        {"name": "overlap", "type": "int", "default": "50", "help": "Overlap in samples (must be < window)"}
    ]

    @classmethod
    def get_info(cls): 
        return f"{cls.name} — Normalize signal to specified range using sliding windows (Category: {cls.category})"

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
        range_min = params.get("range_min")
        range_max = params.get("range_max")
        window = params.get("window")
        overlap = params.get("overlap")
        
        if range_min >= range_max:
            raise ValueError("range_max must be greater than range_min")
        if window <= 0:
            raise ValueError("Window size must be positive")
        if overlap < 0 or overlap >= window:
            raise ValueError("Overlap must be non-negative and less than window size")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output signal data"""
        if len(y_new) != len(y_original):
            raise ValueError("Output signal length differs from input")
        if np.any(np.isnan(y_new)) and not np.any(np.isnan(y_original)):
            raise ValueError("Min-max normalization produced unexpected NaN values")

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
        """Apply min-max normalization to the channel data."""
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
                suffix="MinMaxNorm"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Min-max normalization failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic for min-max normalization"""
        range_min = params["range_min"]
        range_max = params["range_max"]
        window = params["window"]
        overlap = params["overlap"]
        step = window - overlap

        if len(y) < window:
            raise ValueError("Signal is shorter than window size")

        y_out = np.zeros_like(y, dtype=float)
        count = np.zeros_like(y, dtype=int)

        for start in range(0, len(y) - window + 1, step):
            end = start + window
            segment = y[start:end]
            try:
                ymin, ymax = np.min(segment), np.max(segment)
                if ymax - ymin == 0:
                    norm_segment = np.zeros_like(segment)
                else:
                    norm = (segment - ymin) / (ymax - ymin)
                    norm_segment = norm * (range_max - range_min) + range_min
            except Exception:
                norm_segment = np.zeros_like(segment)
            y_out[start:end] += norm_segment
            count[start:end] += 1

        count[count == 0] = 1  # Prevent divide by zero
        y_new = y_out / count
        return y_new
