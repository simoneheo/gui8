
import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class cumulative_max_step(BaseStep):
    name = "cumulative_max"
    category = "Transform"
    description = "Compute local maximum using sliding windows to create a smooth upper envelope"
    tags = ["time-series", "envelope", "cumulative", "maximum", "peak", "tracking"]
    params = [
        {
            "name": "window",
            "type": "int",
            "default": "100",
            "help": "Window size in samples (must be > 0 and < signal length)"
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
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input signal data"""
        if len(y) == 0:
            raise ValueError("Input signal is empty")
        if np.all(np.isnan(y)):
            raise ValueError("Signal contains only NaN values")
        if np.all(np.isinf(y)):
            raise ValueError("Signal contains only infinite values")
        if len(y) < 2:
            raise ValueError("Signal too short for cumulative max (minimum 2 samples)")

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
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output signal data"""
        if len(y_new) != len(y_original):
            raise ValueError(f"Output length mismatch: expected {len(y_original)}, got {len(y_new)}")
        if np.any(np.isnan(y_new)) and not np.any(np.isnan(y_original)):
            raise ValueError("Cumulative max produced unexpected NaN values")
        if np.any(np.isinf(y_new)) and not np.any(np.isinf(y_original)):
            raise ValueError("Cumulative max produced unexpected infinite values")

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
        """Apply cumulative max processing to the channel data."""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params, len(y))
            
            # Process the data
            y_final = cls.script(x, y, params)
            
            # Validate output data
            cls._validate_output_data(y, y_final)
            
            return cls.create_new_channel(
                parent=channel, 
                xdata=x, 
                ydata=y_final, 
                params=params,
                suffix="CumMax"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Cumulative max processing failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic for cumulative max"""
        window = params["window"]
        overlap = params["overlap"]
        step = window - overlap

        y_new = np.zeros_like(y)
        counts = np.zeros_like(y)

        # Process all complete windows
        for start in range(0, len(y) - window + 1, step):
            end = start + window
            max_val = np.nanmax(y[start:end])
            if np.isnan(max_val):
                max_val = 0  # Default fallback for all-NaN window
            y_new[start:end] += max_val
            counts[start:end] += 1

        # Handle trailing samples if any
        if len(y) > window:
            last_end = ((len(y) - window) // step) * step + window
            if last_end < len(y):
                max_val = np.nanmax(y[-window:])
            if np.isnan(max_val):
                max_val = 0
                y_new[last_end:] += max_val
                counts[last_end:] += 1

        # Average overlapping regions
        counts[counts == 0] = 1  # Avoid division by zero
        y_new = y_new / counts
        
        return y_new
