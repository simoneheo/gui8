import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class rank_transform_step(BaseStep):
    name = "rank_transform"
    category = "Transform"
    description = """Replace each signal value with its rank (0-based indexing).
    
This step transforms the signal by replacing each value with its rank position
when the signal is sorted. The smallest value gets rank 0, the second smallest
gets rank 1, and so on. This creates a monotonic transformation that preserves
the relative ordering of values.

Useful for:
• **Non-parametric analysis**: Remove assumptions about data distribution
• **Outlier robustness**: Reduce the impact of extreme values
• **Feature scaling**: Normalize data to a fixed range [0, n-1]
• **Statistical testing**: Prepare data for rank-based statistical tests

The transformation preserves the relative ordering while removing magnitude information."""
    tags = ["time-series", "transformation", "non-parametric", "rank", "order", "robust"]
    params = []

    @classmethod
    def get_info(cls): 
        return f"{cls.name} — Replace values with their ranks (Category: {cls.category})"

    @classmethod
    def get_prompt(cls): 
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input signal data"""
        if len(y) == 0:
            raise ValueError("Input signal is empty")
        if np.any(np.isnan(y)):
            raise ValueError("Input signal contains NaN values (cannot rank NaN values)")
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
            raise ValueError("Rank computation resulted in invalid values")
        
        # Check that ranks are in expected range
        if len(y_original) > 0:
            if np.min(y_new) < 0 or np.max(y_new) >= len(y_original):
                raise ValueError(f"Computed ranks out of expected range [0, {len(y_original)-1}]")

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        """Parse and validate user input parameters"""
        # No parameters to parse for this step
        return {}

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        """Apply rank transformation to the channel data."""
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
                suffix="RankTransform"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Rank transform failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic for rank transformation"""
        # Compute ranks using argsort of argsort
        # This gives 0-based ranks where the smallest value gets rank 0
        y_new = np.argsort(np.argsort(y)).astype(float)

        return y_new
