import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class moving_average_step(BaseStep):
    name = "moving_average"
    category = "Filter"
    description = """Apply moving average smoothing using overlapping windows to reduce noise and smooth signal variations."""
    tags = ["time-series", "smoothing", "noise-reduction", "window", "sliding", "average", "moving"]
    params = [
        {'name': 'window', 'type': 'int', 'default': '5', 'help': 'Window size in samples (must be >= 1)'},
        {'name': 'overlap', 'type': 'int', 'default': '0', 'help': 'Overlap between windows in samples (must be < window)'},
    ]

    @classmethod
    def get_info(cls): 
        return f"{cls.name} — Apply moving average smoothing to reduce noise (Category: {cls.category})"
    
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
        """Validate cross-field logic and business rules"""
        window = params.get("window")
        overlap = params.get("overlap")
        
        # Check if window size is valid
        if window < 1:
            raise ValueError("Window size must be at least 1")
        
        # Check if overlap is valid relative to window
        if overlap < 0 or overlap >= window:
            raise ValueError("Overlap must be non-negative and smaller than window size")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output signal data"""
        if np.any(np.isnan(y_new)) and not np.any(np.isnan(y_original)):
            raise ValueError("Moving average produced unexpected NaN values")
        if np.any(np.isinf(y_new)) and not np.any(np.isinf(y_original)):
            raise ValueError("Moving average produced unexpected infinite values")

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        parsed = {}
        
        for param in cls.params:
            name = param["name"]
            val = user_input.get(name, param["default"])
            
            try:
                if val == "": 
                    parsed[name] = None
                elif param["type"] == "int": 
                    parsed_val = int(val)
                    parsed[name] = parsed_val
                else: 
                    parsed[name] = val
            except ValueError as e:
                if "could not convert" in str(e) or "invalid literal" in str(e):
                    raise ValueError(f"{name} must be a valid {param['type']}")
                raise e
                
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        try:
            x = channel.xdata
            y = channel.ydata
            fs = getattr(channel, 'fs', None)
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            
            # Get processed data from script method
            y_new = cls.script(x, y, fs, params)
            
            # Validate output data
            cls._validate_output_data(y, y_new)
            
            return cls.create_new_channel(parent=channel, xdata=x, ydata=y_new, params=params)
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Moving average processing failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> np.ndarray:
        window = params["window"]
        overlap = params["overlap"]
        
        # Check if signal is long enough for the window
        if len(y) < window:
            raise ValueError(f"Signal too short: requires at least {window} samples")

        step = window - overlap
        result = np.zeros_like(y, dtype=float)
        count = np.zeros_like(y, dtype=int)

        for start in range(0, len(y) - window + 1, step):
            end = start + window
            segment = y[start:end]
            avg = np.mean(segment)
            result[start:end] += avg
            count[start:end] += 1

        # Avoid divide by zero
        count[count == 0] = 1
        y_new = result / count  
        return y_new
