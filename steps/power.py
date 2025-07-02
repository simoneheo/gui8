import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class power_step(BaseStep):
    name = "power"
    category = "Arithmetic"
    description = "Raise each value to a power"
    tags = ["time-series"]
    params = [
        {"name": "exponent", "type": "float", "default": "2.0", "help": "Exponent to raise to"},
    ]

    @classmethod
    def get_info(cls): return f"{cls.name} — {cls.description} (Category: {cls.category})"

    @classmethod
    def get_prompt(cls): return {"info": cls.description, "params": cls.params}

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        try:
            parsed = {}
            for param in cls.params:
                value = user_input.get(param["name"], param["default"])
                parsed[param["name"]] = float(value) if param["type"] == "float" else int(value)
            
            # Parameter validation
            exponent = parsed["exponent"]
            
            if np.isnan(exponent) or np.isinf(exponent):
                raise ValueError(f"Exponent must be a finite number, got {exponent}")
            
            return parsed
        except ValueError as e:
            raise ValueError(f"Parameter validation failed: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to parse input parameters: {str(e)}")

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        # Input validation
        if channel is None:
            raise ValueError("Channel is None")
        if channel.ydata is None or len(channel.ydata) == 0:
            raise ValueError("Channel has no data")
        if channel.xdata is None or len(channel.xdata) == 0:
            raise ValueError("Channel has no x-axis data")
        if len(channel.xdata) != len(channel.ydata):
            raise ValueError("X and Y data lengths don't match")
        
        # Check for NaN values
        if np.any(np.isnan(channel.ydata)):
            raise ValueError("Input signal contains NaN values")
        if np.any(np.isinf(channel.ydata)):
            raise ValueError("Input signal contains infinite values")
        
        y = channel.ydata
        x = channel.xdata
        exponent = params['exponent']
        
        try:
            # Special handling for negative numbers and fractional exponents
            if exponent != int(exponent) and np.any(y < 0):
                raise ValueError("Cannot raise negative numbers to fractional powers")
            
            # Perform power operation
            y_new = np.power(y, exponent)
            
            # Validate result
            if np.any(np.isnan(y_new)):
                raise ValueError("Power operation resulted in NaN values")
            if np.any(np.isinf(y_new)):
                raise ValueError("Power operation resulted in infinite values")
            
            return cls.create_new_channel(parent=channel, xdata=x, ydata=y_new, params=params)
            
        except Exception as e:
            raise ValueError(f"Power operation failed: {str(e)}")
