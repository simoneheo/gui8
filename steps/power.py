import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class power_step(BaseStep):
    name = "power"
    category = "Arithmetic"
    description = """Raise each signal value to a specified power.
    
This step applies an exponential operation to each sample in the signal,
raising each value to the specified exponent. Useful for:

• **Signal transformation**: Apply power-law transformations
• **Feature extraction**: Create polynomial features for analysis
• **Amplitude modification**: Square signals for power calculations
• **Mathematical operations**: Apply various power transformations

The operation preserves the signal's timing while transforming its amplitude according to the power law."""
    tags = ["time-series", "arithmetic", "transformation", "power", "exponent", "nonlinear"]
    params = [
        {"name": "exponent", "type": "float", "default": "2.0", "help": "Exponent to raise to (must be finite)"},
    ]

    @classmethod
    def get_info(cls): 
        return f"{cls.name} — Raise signal values to specified power (Category: {cls.category})"

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
        exponent = params.get("exponent")
        
        if np.isnan(exponent) or np.isinf(exponent):
            raise ValueError(f"Exponent must be a finite number, got {exponent}")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output signal data"""
        if len(y_new) != len(y_original):
            raise ValueError("Output signal length differs from input")
        if np.any(np.isnan(y_new)):
            raise ValueError("Power operation resulted in NaN values")
        if np.any(np.isinf(y_new)):
            raise ValueError("Power operation resulted in infinite values")

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
        """Apply power operation to the channel data."""
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
                suffix="Power"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Power operation failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic for power operation"""
        exponent = params['exponent']
        
        # Special handling for negative numbers and fractional exponents
        if exponent != int(exponent) and np.any(y < 0):
            raise ValueError("Cannot raise negative numbers to fractional powers")
        
        y_new = np.power(y, exponent)

        return y_new
