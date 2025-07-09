import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class zscore_sliding_step(BaseStep):
    name = "zscore_sliding"
    category = "Normalization"
    description = """Standardize signal using sliding window normalization to zero mean and unit standard deviation.
    
This step applies z-score normalization within overlapping sliding windows, allowing
the normalization to adapt to local signal characteristics. Each sample is normalized
using the mean and standard deviation of its local window.

• **Window size**: Number of samples in each normalization window
• **Overlap**: Number of overlapping samples between consecutive windows

Useful for:
• **Local normalization**: Adapt to changing signal characteristics
• **Non-stationary signals**: Handle signals with varying statistics
• **Feature extraction**: Prepare signals for pattern recognition
• **Noise reduction**: Normalize local variations while preserving global structure"""
    tags = ["time-series", "normalization", "standardization", "sliding-window", "window", "sliding", "zscore"]
    params = [
        {
            "name": "window", 
            "type": "int", 
            "default": "100", 
            "help": "Window size in samples (must be positive)"
        },
        {
            "name": "overlap", 
            "type": "int", 
            "default": "50", 
            "help": "Overlap in samples (must be < window size)"
        }
    ]

    @classmethod
    def get_info(cls): 
        return f"{cls.name} — Standardize signal using sliding window normalization (Category: {cls.category})"

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
        if np.all(np.isinf(y)):
            raise ValueError("Signal contains only infinite values")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        window = params.get("window")
        overlap = params.get("overlap")
        
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
            raise ValueError("Sliding z-score produced unexpected NaN values")
        if np.any(np.isinf(y_new)) and not np.any(np.isinf(y_original)):
            raise ValueError("Sliding z-score produced unexpected infinite values")

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
        """Apply sliding z-score normalization to the channel data."""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            
            # Validate signal length vs window size
            if len(y) < params["window"]:
                raise ValueError("Signal is shorter than window size")
            
            # Process the data
            y_new = cls.script(x, y, params)
            
            # Validate output data
            cls._validate_output_data(y, y_new)
            
            return cls.create_new_channel(
                parent=channel, 
                xdata=x, 
                ydata=y_new, 
                params=params,
                suffix="ZScoreSliding"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Sliding z-score processing failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic for sliding z-score normalization"""
        window = params["window"]
        overlap = params["overlap"]
        
        step = window - overlap
        if step < 1:
            raise ValueError(f"Invalid window/overlap: step={step} must be >= 1")

        y_out = np.zeros_like(y, dtype=float)
        count = np.zeros_like(y, dtype=int)

        for start in range(0, len(y) - window + 1, step):
            end = start + window
            segment = y[start:end]
            mean = np.mean(segment)
            std = np.std(segment)

            if std == 0 or np.isnan(std) or np.isinf(std):
                norm_segment = np.zeros_like(segment)
            else:
                norm_segment = (segment - mean) / std

            y_out[start:end] += norm_segment
            count[start:end] += 1

        count[count == 0] = 1
        y_new = y_out / count
        return y_new
