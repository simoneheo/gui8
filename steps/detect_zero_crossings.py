import numpy as np
from steps.base_step import BaseStep
from steps.process_registry import register_step
from channel import Channel

@register_step
class detect_zero_crossings_step(BaseStep):
    name = "detect_zero_crossings"
    category = "Features"
    description = """Detects zero-crossing events where signal changes sign (positive to negative or vice versa).
Useful for identifying phase transitions, baseline crossings, and signal polarity changes."""
    tags = ["time-series", "event", "zero-crossing", "detection", "transitions", "polarity"]
    params = [
        {
            'name': 'threshold', 
            'type': 'float', 
            'default': '0.0', 
            'help': 'Crossing threshold value. Events occur when signal crosses this level. Use 0.0 for true zero-crossings, or signal offset for baseline crossings.'
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
            raise ValueError("Signal too short for zero-crossing detection (minimum 2 samples)")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        threshold = params.get("threshold")
        if threshold is not None and np.isnan(threshold):
            raise ValueError("Threshold cannot be NaN")

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
        """Apply zero-crossing detection to the channel data."""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            
            # Process the data
            x_output, y_output = cls.script(x, y, params)
            
            # Validate output data
            cls._validate_output_data(x_output, y_output)
            
            return cls.create_new_channel(
                parent=channel, 
                xdata=x_output, 
                ydata=y_output, 
                params=params,
                suffix="ZeroCrossings"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Zero-crossing detection failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray]:
        """Core processing logic for zero-crossing detection"""
        threshold = params.get("threshold", 0.0)
        indices = np.where(np.diff(np.sign(y - threshold)) != 0)[0]
        x_new = x[indices]
        y_new = y[indices]  
        return x_new, y_new
