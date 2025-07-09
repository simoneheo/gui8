import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class multiply_constant_step(BaseStep):
    name = "multiply_constant"
    category = "Arithmetic"
    description = """Multiply all signal values by a constant factor.
    
This step applies a scalar multiplication to each sample in the signal,
scaling the amplitude by the specified factor. Useful for:

• **Amplitude scaling**: Adjust signal magnitude for analysis or visualization
• **Unit conversion**: Convert between different units of measurement
• **Gain adjustment**: Apply amplification or attenuation
• **Normalization**: Scale signals to specific ranges

The operation preserves the signal's shape and timing while adjusting its magnitude."""
    tags = ["time-series", "arithmetic", "scaling", "multiply", "constant", "gain", "amplify"]
    params = [
        {"name": "factor", "type": "float", "default": "2.0", "help": "Multiplication factor (must be finite)"}
    ]

    @classmethod
    def get_info(cls): 
        return f"{cls.name} — Multiply signal values by constant factor (Category: {cls.category})"

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
        factor = params.get("factor")
        
        if np.isnan(factor) or np.isinf(factor):
            raise ValueError(f"Factor must be a finite number, got {factor}")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output signal data"""
        if len(y_new) != len(y_original):
            raise ValueError("Output signal length differs from input")
        if np.any(np.isnan(y_new)):
            raise ValueError("Multiplication resulted in NaN values")
        if np.any(np.isinf(y_new)):
            raise ValueError("Multiplication resulted in infinite values")

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
        """Apply constant multiplication to the channel data."""
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
                suffix="Multiply"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Constant multiplication failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic for constant multiplication"""
        factor = params['factor']
        y_new = y * factor
        return y_new
