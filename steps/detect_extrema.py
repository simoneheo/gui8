import numpy as np
from steps.base_step import BaseStep
from steps.process_registry import register_step
from channel import Channel
from scipy.signal import argrelextrema

@register_step
class detect_extrema_step(BaseStep):
    name = "detect_extrema"
    category = "Features"
    description = """Detects local extrema (peaks and valleys) in overlapping sliding windows.
Extrema are filtered by height relative to signal mean to identify significant events.
Window and overlap are in samples."""
    tags = ["time-series", "event", "peaks", "valleys", "scipy", "argrelextrema", "extrema", "detection"]
    params = [
        {'name': 'min_height', 'type': 'float', 'default': '0.1', 'help': 'Minimum height/depth as a fraction of signal range (0.0–1.0)'},
        {'name': 'window', 'type': 'int', 'default': '200', 'help': 'Window size in samples (must be > 0 and < signal length)'},
        {'name': 'overlap', 'type': 'int', 'default': '100', 'help': 'Overlap between windows in samples (must be < window)'}
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
            raise ValueError("Signal too short for extrema detection (minimum 2 samples)")
        if np.any(np.isnan(x)):
            raise ValueError("Time data contains NaN values")
        if np.any(np.isinf(x)):
            raise ValueError("Time data contains infinite values")
        if not np.all(np.diff(x) > 0):
            raise ValueError("Time data must be monotonically increasing")

    @classmethod
    def _validate_parameters(cls, params: dict, total_samples: int) -> None:
        """Validate parameters and business rules"""
        min_height = params.get("min_height")
        window = params.get("window")
        overlap = params.get("overlap")
        
        if min_height is None or min_height < 0.0 or min_height > 1.0:
            raise ValueError("Min height must be between 0.0 and 1.0")
        
        if window is None or window < 2:
            raise ValueError("Window must be at least 2 samples")
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
        if len(x_output) != len(y_output):
            raise ValueError("Output time and signal data length mismatch")

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
        """Apply extrema detection to the channel data."""
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
                suffix="Extrema"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Extrema detection failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray]:
        """Core processing logic for extrema detection"""
        min_height = params["min_height"]
        window_samples = params["window"]
        overlap_samples = params["overlap"]

        step = window_samples - overlap_samples
        signal_range = np.max(y) - np.min(y)
        height_thresh = min_height * signal_range
        signal_mean = np.mean(y)

        extrema_indices = []

        for start in range(0, len(y) - window_samples + 1, step):
            end = start + window_samples
            y_window = y[start:end]

            # Use order relative to window size
            order = max(1, window_samples // 10)

            local_max = argrelextrema(y_window, np.greater, order=order)[0]
            local_min = argrelextrema(y_window, np.less, order=order)[0]

            for i in local_max:
                idx = start + i
                if y[idx] - signal_mean >= height_thresh:
                    extrema_indices.append(idx)

            for i in local_min:
                idx = start + i
                if signal_mean - y[idx] >= height_thresh:
                    extrema_indices.append(idx)

        if not extrema_indices:
            return np.array([]), np.array([])

        extrema_indices = np.unique(extrema_indices)
        x_new = x[extrema_indices]
        y_new = y[extrema_indices]
        return x_new, y_new
