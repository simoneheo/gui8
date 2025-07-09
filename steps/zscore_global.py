import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class zscore_global_step(BaseStep):
    name = "zscore_global"
    category = "Normalization"
    description = """Standardize signal to zero mean and unit standard deviation using global statistics across the entire signal."""
    tags = ["time-series", "normalization", "standardization", "zscore", "global", "mean", "std"]
    params = []

    @classmethod
    def get_info(cls): 
        return f"{cls.name} — Standardize signal to zero mean and unit standard deviation (Category: {cls.category})"
    
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
        if np.all(np.isinf(y)):
            raise ValueError("Signal contains only infinite values")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate cross-field logic and business rules"""
        # No parameters to validate for z-score normalization
        pass

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output signal data"""
        if np.any(np.isnan(y_new)) and not np.any(np.isnan(y_original)):
            raise ValueError("Z-score normalization produced unexpected NaN values")
        if np.any(np.isinf(y_new)) and not np.any(np.isinf(y_original)):
            raise ValueError("Z-score normalization produced unexpected infinite values")

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        # No parameters to parse for z-score normalization
        return {}

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        """Apply z-score normalization to the channel data."""
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
                suffix="ZScore"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Z-score normalization failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic for z-score normalization"""
        mean = np.mean(y)
        std = np.std(y)
        
        if std == 0:
            raise ValueError("Standard deviation is zero. Cannot normalize constant signal.")
        
        if np.isnan(std) or np.isinf(std):
            raise ValueError("Failed to compute standard deviation")
            
        y_new = (y - mean) / std
        return y_new
