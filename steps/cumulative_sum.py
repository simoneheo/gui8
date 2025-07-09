import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class cumulative_sum_step(BaseStep):
    name = "cumulative_sum"
    category = "Transform"
    description = "Compute local sum of values within sliding windows to extract energy patterns"
    tags = ["time-series", "cumulative", "integration", "sum", "accumulate"]
    params = [
        {
            "name": "window",
            "type": "int",
            "default": "100",
            "help": "Window size in number of samples (must be > 0 and < signal length)"
        },
        {
            "name": "overlap",
            "type": "int",
            "default": "50",
            "help": "Overlap between windows in samples (must be < window)"
        }
    ]

    @classmethod
    def get_info(cls):
        return f"{cls.name} — {cls.description} (Category: {cls.category})"

    @classmethod
    def get_prompt(cls):
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def _validate_input_data(cls, x: np.ndarray, y: np.ndarray) -> None:
        """Validate input signal data"""
        if len(x) == 0 or len(y) == 0:
            raise ValueError("Input signal is empty")
        if len(x) != len(y):
            raise ValueError(f"Time and signal data length mismatch: {len(x)} vs {len(y)}")
        if len(x) < 2:
            raise ValueError("Signal too short for cumulative sum (minimum 2 samples)")
        if np.any(np.isnan(x)):
            raise ValueError("Time data contains NaN values")
        if np.any(np.isinf(x)):
            raise ValueError("Time data contains infinite values")
        if not np.all(np.diff(x) > 0):
            raise ValueError("Time data must be monotonically increasing")

    @classmethod
    def _validate_parameters(cls, params: dict, total_samples: int) -> None:
        """Validate parameters and business rules"""
        window = params.get("window")
        overlap = params.get("overlap")
        
        if window is None or window <= 0:
            raise ValueError("Window size must be positive")
        if window > total_samples:
            raise ValueError(
                f"Window size ({window} samples) is larger than signal length ({total_samples} samples). "
                f"Try a smaller window or use a longer signal."
            )
        
        if overlap is None or overlap < 0:
            raise ValueError("Overlap must be non-negative")
        if overlap >= window:
            raise ValueError("Overlap must be smaller than window size")
        
        # Check if we'll have enough windows
        step = window - overlap
        if step <= 0:
            raise ValueError(f"Step size must be positive (window - overlap = {step})")
        
        estimated_windows = int((total_samples - window) / step) + 1
        if estimated_windows < 1:
            raise ValueError(
                f"Configuration would produce no valid windows. "
                f"Window: {window} samples, Overlap: {overlap} samples, Signal: {total_samples} samples"
            )

    @classmethod
    def _validate_output_data(cls, x_output: np.ndarray, y_output: np.ndarray) -> None:
        """Validate output signal data"""
        if len(x_output) == 0 or len(y_output) == 0:
            raise ValueError("No output data generated")
        if len(x_output) != len(y_output):
            raise ValueError("Output time and signal data length mismatch")
        if np.any(np.isnan(y_output)):
            raise ValueError("Output contains NaN values")
        if np.any(np.isinf(y_output)):
            raise ValueError("Output contains infinite values")

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
        """Apply cumulative sum processing to the channel data."""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Validate input data and parameters
            cls._validate_input_data(x, y)
            cls._validate_parameters(params, len(x))
            
            # Process the data
            x_output, y_output = cls.script(x, y, params)
            
            # Validate output data
            cls._validate_output_data(x_output, y_output)
            
            return cls.create_new_channel(
                parent=channel,
                xdata=x_output,
                ydata=y_output,
                params=params,
                suffix="CumSum"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Cumulative sum processing failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray]:
        """Core processing logic for cumulative sum"""
        window = params["window"]
        overlap = params["overlap"]
        step = window - overlap

        x_new = []
        y_new = []

        # Process all complete windows
        for start in range(0, len(y) - window + 1, step):
            end = start + window
            window_sum = np.sum(y[start:end])
            center_x = x[start + window // 2]
            x_new.append(center_x)
            y_new.append(window_sum)

        # Convert to numpy arrays
        x_new = np.array(x_new)
        y_new = np.array(y_new)
        
        return x_new, y_new
