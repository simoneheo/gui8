import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class add_constant_step(BaseStep):
    name = "add_constant"
    category = "Arithmetic"
    description = """Add a constant value to all signal values, creating a vertical offset.
    
This operation is useful for:
• Removing or adding DC offset to signals
• Shifting baseline levels for visualization
• Normalizing signals to a common baseline
• Adjusting signal ranges for specific analysis requirements

The transformation applies y_new = y + constant element-wise to all signal values."""
    tags = ["time-series", "arithmetic", "offset"]
    params = [
        {
            "name": "constant", 
            "type": "float", 
            "default": "0.0", 
            "help": "Constant value to add to all signal values. Positive values shift up, negative values shift down."
        }
    ]

    @classmethod
    def get_info(cls): 
        return f"{cls.name} — Add constant offset to signal values (y + constant) for baseline adjustment (Category: {cls.category})"

    @classmethod
    def get_prompt(cls): 
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        parsed = {}
        
        try:
            # Parse constant parameter
            constant_val = user_input.get("constant", "0.0")
            
            # Handle both string and numeric inputs
            if isinstance(constant_val, str):
                constant_val = constant_val.strip()
                if not constant_val:
                    raise ValueError("Constant parameter cannot be empty")
                try:
                    constant = float(constant_val)
                except ValueError:
                    raise ValueError(f"Constant must be a valid number, got '{constant_val}'")
            elif isinstance(constant_val, (int, float)):
                constant = float(constant_val)
            else:
                raise ValueError(f"Constant must be a number, got {type(constant_val).__name__}: {constant_val}")
            
            # Validate constant value
            if np.isnan(constant):
                raise ValueError("Constant cannot be NaN")
            if np.isinf(constant):
                raise ValueError("Constant cannot be infinite")
            
            # Sanity check for extremely large values
            if abs(constant) > 1e15:
                raise ValueError(f"Constant value seems too large: {constant}")
            
            parsed["constant"] = constant
            return parsed
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Failed to parse constant parameter: {str(e)}")

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        # Input validation
        if channel is None:
            raise ValueError("Input channel cannot be None")
        
        if not hasattr(channel, 'ydata') or channel.ydata is None:
            raise ValueError("Channel must have valid signal data (ydata)")
        
        if not hasattr(channel, 'xdata') or channel.xdata is None:
            raise ValueError("Channel must have valid time data (xdata)")
        
        y = channel.ydata
        x = channel.xdata
        
        # Data validation
        if len(y) == 0:
            raise ValueError("Signal data cannot be empty")
        
        if len(x) != len(y):
            raise ValueError(f"Time and signal data length mismatch: {len(x)} vs {len(y)}")
        
        # Check for all NaN values
        if np.all(np.isnan(y)):
            raise ValueError("Signal contains only NaN values")
        
        # Extract parameters
        constant = params.get('constant', 0.0)
        
        try:
            # Apply constant addition
            y_new = y + constant
            
            # Validate output
            if np.any(np.isnan(y_new)) and not np.any(np.isnan(y)):
                raise ValueError("Constant addition produced unexpected NaN values")
                
            if np.any(np.isinf(y_new)) and not np.any(np.isinf(y)):
                raise ValueError("Constant addition produced unexpected infinite values")
            
            # Create descriptive suffix
            if constant >= 0:
                suffix = f"Plus{constant:g}"
            else:
                suffix = f"Minus{abs(constant):g}"
            
            # Create output channel with descriptive labeling
            return cls.create_new_channel(
                parent=channel, 
                xdata=x, 
                ydata=y_new, 
                params=params,
                suffix=suffix
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Constant addition failed: {str(e)}")
