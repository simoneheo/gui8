import numpy as np
from steps.base_step import BaseStep
from steps.process_registry import register_step
from channel import Channel

@register_step
class energy_sliding_step(BaseStep):
    name = "energy_sliding"
    category = "Features"
    description = """Computes signal energy (sum of squares) in overlapping sliding windows.
Energy is a measure of signal power and is useful for detecting periods of high activity."""
    tags = ["time-series", "feature", "energy", "sliding-window", "window", "sliding", "power"]
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
            "help": "Overlap in samples (must be < window size). Higher values give smoother results but more computation."
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
            raise ValueError("Signal too short for energy computation (minimum 2 samples)")
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
        
        if overlap is None or overlap < 0 or overlap >= window:
            raise ValueError("Overlap must be non-negative and less than window size")
        
        # Check if we'll have enough windows
        step = window - overlap
        if step < 1:
            raise ValueError(f"Step size too small (step={step}). Increase window or reduce overlap")
        
        estimated_windows = int((total_samples - window) / step) + 1
        if estimated_windows < 1:
            raise ValueError(
                f"Configuration would produce no valid windows. "
                f"Window: {window} samples, Overlap: {overlap}, Signal: {total_samples} samples"
            )

    @classmethod
    def _validate_output_data(cls, x_output: np.ndarray, y_output: np.ndarray) -> None:
        """Validate output signal data"""
        if len(x_output) == 0 or len(y_output) == 0:
            raise ValueError("No valid windows computed")
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
        """Apply sliding energy computation to the channel data."""
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
                suffix="Energy"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Energy computation failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray]:
        """Core processing logic for sliding energy computation"""
        window = params["window"]
        overlap = params["overlap"]
        step = window - overlap

        x_new = []
        y_new = []

        for start in range(0, len(y) - window + 1, step):
            end = start + window
            segment = y[start:end]
            center_x = x[start + window // 2]

            try:
                energy = np.sum(segment ** 2)
                if not np.isfinite(energy):
                    continue
                x_new.append(center_x)
                y_new.append(energy)
            except Exception:
                continue

        # Convert to numpy arrays
        x_new = np.array(x_new)
        y_new = np.array(y_new)
        
        return x_new, y_new
