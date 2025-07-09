import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class abs_val_step(BaseStep):
    name = "abs_val"
    category = "Arithmetic"
    description = """Apply absolute value transformation to convert all signal values to their absolute magnitude"""
    tags = ["time-series", "transform", "rectification", "absolute", "arithmetic", "magnitude"]
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
        # No parameters to validate for absolute value transformation
        pass

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output signal data"""
        if np.any(np.isnan(y_new)) and not np.any(np.isnan(y_original)):
            raise ValueError("Absolute value transformation produced unexpected NaN values")
            
        if np.any(np.isinf(y_new)) and not np.any(np.isinf(y_original)):
            raise ValueError("Absolute value transformation produced unexpected infinite values")

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        # No parameters to parse for absolute value transformation
        return {}

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        try:
            x = channel.xdata
            y = channel.ydata
            fs = getattr(channel, 'fs', None)

            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            
            # Get processed data from script method
            y_final = cls.script(x, y, fs, params)
            
            # Validate output data
            cls._validate_output_data(y, y_final)
            
            return cls.create_new_channel(
                parent=channel, 
                xdata=x, 
                ydata=y_final, 
                params=params,
                suffix="AbsVal"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Absolute value transformation failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> np.ndarray:
        # Apply absolute value transformation
        y_new = np.abs(y)
        
        return y_new
