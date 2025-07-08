import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class standardize_step(BaseStep):
    name = "standardize"
    category = "Transform"
    description = "Scale to custom mean and std"
    tags = ["time-series"]
    params = [
        {"name": "mean", "type": "float", "default": "0.0", "help": "Target mean"},
        {"name": "std", "type": "float", "default": "1.0", "help": "Target std (must be > 0)"},
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
            mean = parsed["mean"]
            std = parsed["std"]
            
            if np.isnan(mean) or np.isinf(mean):
                raise ValueError(f"Target mean must be a finite number, got {mean}")
            if np.isnan(std) or np.isinf(std):
                raise ValueError(f"Target std must be a finite number, got {std}")
            if std <= 0:
                raise ValueError(f"Target std must be positive, got {std}")
            
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
        target_mean = params['mean']
        target_std = params['std']
        
        try:
            # Compute current statistics
            current_mean = np.mean(y)
            current_std = np.std(y)
            
            if current_std == 0:
                raise ValueError("Input signal has zero standard deviation (constant signal)")
            
            if np.isnan(current_mean) or np.isinf(current_mean):
                raise ValueError("Failed to compute signal mean")
            if np.isnan(current_std) or np.isinf(current_std):
                raise ValueError("Failed to compute signal standard deviation")
            
            # Apply standardization
            y_new = ((y - current_mean) / current_std) * target_std + target_mean
            
            # Validate result
            if np.any(np.isnan(y_new)) or np.any(np.isinf(y_new)):
                raise ValueError("Standardization resulted in invalid values")
            
            return cls.create_new_channel(parent=channel, xdata=x, ydata=y_new, params=params)
            
        except Exception as e:
            raise ValueError(f"Custom standardization failed: {str(e)}")
