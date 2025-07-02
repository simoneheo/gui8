import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class quantize_step(BaseStep):
    name = "quantize"
    category = "Arithmetic"
    description = "Round to nearest multiple of step size"
    tags = ["time-series"]
    params = [
        {"name": "step_size", "type": "float", "default": "0.1", "help": "Quantization step size (must be > 0)"},
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
            step_size = parsed["step_size"]
            
            if np.isnan(step_size) or np.isinf(step_size):
                raise ValueError(f"Step size must be a finite number, got {step_size}")
            if step_size <= 0:
                raise ValueError(f"Step size must be positive, got {step_size}")
            
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
        step_size = params['step_size']
        
        try:
            # Perform quantization
            y_new = np.round(y / step_size) * step_size
            
            # Validate result
            if np.any(np.isnan(y_new)) or np.any(np.isinf(y_new)):
                raise ValueError("Quantization resulted in invalid values")
            
            return cls.create_new_channel(parent=channel, xdata=x, ydata=y_new, params=params)
            
        except Exception as e:
            raise ValueError(f"Quantization failed: {str(e)}")
