import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class abs_transform_step(BaseStep):
    name = "abs_transform"
    category = "Transform"
    description = """Apply absolute value transformation to convert all signal values to their absolute magnitude.
    
This transformation is useful for:
• Converting bipolar signals to unipolar (e.g., EMG rectification)
• Removing sign information while preserving magnitude
• Preprocessing for envelope detection
• Creating energy-like measures from oscillatory signals

The transformation applies y_new = |y| element-wise to all signal values."""
    tags = ["time-series", "transform", "rectification"]
    params = [
        # No parameters needed for absolute value transformation
    ]

    @classmethod
    def get_info(cls): 
        return f"{cls.name} — Apply absolute value transformation (|y|) to remove sign information while preserving magnitude (Category: {cls.category})"

    @classmethod
    def get_prompt(cls): 
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        # No parameters to parse for absolute value transformation
        return {}

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
        
        # Check for all infinite values  
        if np.all(np.isinf(y)):
            raise ValueError("Signal contains only infinite values")
        
        try:
            # Apply absolute value transformation
            y_new = np.abs(y)
            
            # Validate output
            if np.any(np.isnan(y_new)) and not np.any(np.isnan(y)):
                raise ValueError("Absolute value transformation produced unexpected NaN values")
                
            if np.any(np.isinf(y_new)) and not np.any(np.isinf(y)):
                raise ValueError("Absolute value transformation produced unexpected infinite values")
            
            # Create output channel with descriptive labeling
            return cls.create_new_channel(
                parent=channel, 
                xdata=x, 
                ydata=y_new, 
                params=params,
                suffix="AbsVal"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Absolute value transformation failed: {str(e)}")
