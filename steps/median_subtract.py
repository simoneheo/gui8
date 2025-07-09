import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class median_subtract_step(BaseStep):
    name = "median_subtract"
    category = "Arithmetic"
    description = """Subtract the median value from the signal to remove baseline offset.
    
This step computes the median of the entire signal and subtracts it from each sample,
effectively centering the signal around zero. This is useful for:

• **Baseline removal**: Eliminate DC offset or systematic bias
• **Signal centering**: Center the signal around zero for further processing
• **Robust normalization**: More robust than mean subtraction for skewed distributions

The median is more robust to outliers than the mean, making this operation suitable for signals with occasional spikes or artifacts."""
    tags = ["time-series", "baseline-removal", "normalization", "median", "subtract", "robust"]
    params = []

    @classmethod
    def get_info(cls): 
        return f"{cls.name} — Subtract median value for robust baseline removal (Category: {cls.category})"

    @classmethod
    def get_prompt(cls): 
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input signal data"""
        if len(y) == 0:
            raise ValueError("Input signal is empty")
        if np.all(np.isnan(y)):
            raise ValueError("Signal contains only NaN values")

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
        if np.any(np.isnan(y_new)) and not np.any(np.isnan(y_original)):
            raise ValueError("Median subtraction produced unexpected NaN values")

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        """Parse and validate user input parameters"""
        # No parameters to parse for this step
        return {}

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        """Apply median subtraction to the channel data."""
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
                suffix="MedianSubtract"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Median subtraction failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic for median subtraction"""
        median_val = np.median(y)
        if np.isnan(median_val):
            raise ValueError("Cannot compute median - signal contains only NaN values")
        y_new = y - median_val
        return y_new
