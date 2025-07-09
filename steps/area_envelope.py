import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class area_envelope_step(BaseStep):
    name = "area_envelope"
    category = "Transform"
    description = "Compute sliding window area envelope for signal energy estimation"
    tags = ["time-series", "envelope", "energy", "area", "magnitude", "amplitude"]
    params = [
        {"name": "window", "type": "int", "default": "25", "help": "Window size in samples"},
        {"name": "overlap", "type": "int", "default": "0", "help": "Overlap between windows in samples"},
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

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        window = params.get("window")
        overlap = params.get("overlap")
        
        if window is None:
            raise ValueError("Window parameter is required")
        if overlap is None:
            raise ValueError("Overlap parameter is required")
        if window <= 0:
            raise ValueError("Window size must be greater than 0")
        if overlap < 0:
            raise ValueError("Overlap cannot be negative")
        if overlap >= window:
            raise ValueError("Overlap must be smaller than window size")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output signal data"""
        if len(y_new) != len(y_original):
            raise ValueError("Output signal length differs from input")
        if np.any(np.isnan(y_new)) and not np.any(np.isnan(y_original)):
            raise ValueError("Processing produced unexpected NaN values")
        if np.any(np.isinf(y_new)) and not np.any(np.isinf(y_original)):
            raise ValueError("Processing produced unexpected infinite values")
        if np.any(y_new < 0):
            raise ValueError("Area envelope should not produce negative values")

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        """Parse and validate user input parameters"""
        parsed = {}
        for param in cls.params:
            name = param["name"]
            val = user_input.get(name, param["default"])
            try:
                if val == "":
                    parsed[name] = None
                elif param["type"] == "float":
                    parsed[name] = float(val)
                elif param["type"] == "int":
                    parsed[name] = int(val)
                elif param["type"] == "bool":
                    parsed[name] = bool(val)
                else:
                    parsed[name] = val
            except ValueError as e:
                if "could not convert" in str(e) or "invalid literal" in str(e):
                    raise ValueError(f"{name} must be a valid {param['type']}")
                raise e
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        """Apply the processing step to a channel"""
        try:
            x = channel.xdata
            y = channel.ydata
            fs = getattr(channel, 'fs', None)
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            
            # Additional validation for this specific step
            if len(y) < params["window"]:
                raise ValueError(f"Signal length ({len(y)}) must be at least window size ({params['window']})")
            
            # Process the data
            y_final = cls.script(x, y, fs, params)
            
            # Validate output data
            cls._validate_output_data(y, y_final)
            
            return cls.create_new_channel(
                parent=channel,
                xdata=x,
                ydata=y_final,
                params=params,
                suffix="AreaEnv"
            )
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Area envelope processing failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> np.ndarray:
        """Core processing logic"""
        y_abs = np.abs(y)
        window = params["window"]
        overlap = params["overlap"]
        step = max(1, window - overlap)

        envelope = np.zeros_like(y_abs)
        counts = np.zeros_like(y_abs)

        for start in range(0, len(y_abs) - window + 1, step):
            end = start + window
            envelope[start:end] += np.sum(y_abs[start:end])
            counts[start:end] += 1

        # Handle trailing part
        if end < len(y_abs):
            envelope[end:] += np.sum(y_abs[-window:])
            counts[end:] += 1

        counts[counts == 0] = 1
        y_new = envelope / counts
        return y_new