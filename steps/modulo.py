import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class modulo_step(BaseStep):
    name = "modulo"
    category = "Arithmetic"
    description = """Apply modulo operation to each sample in the signal.
    
This step computes y % N for each sample, where N is the modulo base.
The result wraps values around the range [0, N-1], useful for:

• **Phase wrapping**: Convert phase angles to [0, 2π] or [0, 360°]
• **Circular data**: Handle periodic signals or circular measurements
• **Data normalization**: Constrain values to a specific range
• **Overflow handling**: Prevent numerical overflow in calculations

The modulo operation preserves the relative relationships between values while constraining them to the specified range."""
    tags = ["time-series", "arithmetic", "normalization", "modulo", "wrap", "periodic"]
    params = [
        {"name": "mod_value", "type": "int", "default": "2", "help": "Modulo base (must be > 0)"},
    ]

    @classmethod
    def get_info(cls): 
        return f"{cls.name} — Apply modulo operation to constrain signal values (Category: {cls.category})"

    @classmethod
    def get_prompt(cls): 
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input signal data"""
        if len(y) == 0:
            raise ValueError("Input signal is empty")
        if np.any(np.isnan(y)):
            raise ValueError("Input signal contains NaN values")
        if np.any(np.isinf(y)):
            raise ValueError("Input signal contains infinite values")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        mod_value = params.get("mod_value")
        
        if mod_value <= 0:
            raise ValueError(f"Modulo base must be positive, got {mod_value}")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output signal data"""
        if len(y_new) != len(y_original):
            raise ValueError("Output signal length differs from input")
        if np.any(np.isnan(y_new)):
            raise ValueError("Modulo operation resulted in NaN values")
        if np.any(np.isinf(y_new)):
            raise ValueError("Modulo operation resulted in infinite values")

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
        """Apply modulo operation to the channel data."""
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
                suffix="Modulo"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Modulo operation failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic for modulo operation"""
        mod_value = params['mod_value']
        y_new = y % mod_value
        return y_new
