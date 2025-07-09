import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class sign_only_step(BaseStep):
    name = "sign_only"
    category = "Arithmetic"
    description = """Extract the sign of each signal value.
    
This step applies the sign function to each sample in the signal, returning
-1 for negative values, 0 for zero values, and +1 for positive values.
This creates a binary-like signal that preserves only the sign information.

Useful for:
• **Sign analysis**: Analyze the sign patterns in signals
• **Binary classification**: Convert continuous signals to sign-based features
• **Zero-crossing detection**: Identify regions where signals change sign
• **Feature extraction**: Create sign-based features for machine learning

The operation preserves the signal's timing while converting amplitude to sign information."""
    tags = ["time-series", "arithmetic", "feature-extraction", "sign", "polarity", "direction"]
    params = []

    @classmethod
    def get_info(cls): 
        return f"{cls.name} — Extract sign of each signal value (Category: {cls.category})"

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
        # No parameters to validate for this step
        pass

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output signal data"""
        if len(y_new) != len(y_original):
            raise ValueError("Output signal length differs from input")
        if np.any(np.isnan(y_new)) or np.any(np.isinf(y_new)):
            raise ValueError("Sign computation resulted in invalid values")

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        """Parse and validate user input parameters"""
        # No parameters to parse for this step
        return {}

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        """Apply sign extraction to the channel data."""
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
                suffix="SignOnly"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Sign computation failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic for sign extraction"""
        y_new = np.sign(y)
        return y_new
