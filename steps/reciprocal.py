import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class reciprocal_step(BaseStep):
    name = "reciprocal"
    category = "Arithmetic"
    description = """Replace each signal value with its reciprocal (1/y).
    
This step computes the reciprocal of each sample in the signal, where
reciprocal(x) = 1/x. Zero values are handled by setting them to zero
in the output to avoid division by zero.

Useful for:
• **Mathematical transformations**: Apply reciprocal transformations
• **Signal inversion**: Invert signal relationships
• **Feature engineering**: Create reciprocal-based features
• **Unit conversions**: Convert between inverse units

The operation preserves the signal's timing while transforming its amplitude."""
    tags = ["time-series", "arithmetic", "transformation", "reciprocal", "inverse", "nonlinear"]
    params = []

    @classmethod
    def get_info(cls): 
        return f"{cls.name} — Replace values with their reciprocals (Category: {cls.category})"

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
        
        # Check for very small values that would create infinities
        very_small_mask = np.abs(y) < 1e-15
        if np.any(very_small_mask & (y != 0)):
            raise ValueError("Signal contains values too small for reciprocal computation")

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
        if np.any(np.isnan(y_new)):
            raise ValueError("Reciprocal computation resulted in NaN values")

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        """Parse and validate user input parameters"""
        # No parameters to parse for this step
        return {}

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        """Apply reciprocal transformation to the channel data."""
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
                suffix="Reciprocal"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Reciprocal computation failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic for reciprocal transformation"""
        # Compute reciprocal, handling zeros safely
        y_new = np.where(y != 0, 1.0 / y, 0.0)
        return y_new
