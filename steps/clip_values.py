import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class clip_values_step(BaseStep):
    name = "clip_values"
    category = "Arithmetic"
    description = """Limit signal values to a specified range by clipping values above max and below min thresholds."""
    tags = ["time-series", "clipping", "bounds", "limit", "threshold", "range"]
    params = [
        {'name': 'min_val', 'type': 'float', 'default': '-3.0', 'help': 'Minimum allowed value'}, 
        {'name': 'max_val', 'type': 'float', 'default': '3.0', 'help': 'Maximum allowed value'}
    ]

    @classmethod
    def get_info(cls): 
        return f"{cls.name} — Clip signal values to specified range (Category: {cls.category})"
    
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
        min_val = params.get("min_val")
        max_val = params.get("max_val")
        
        if min_val is None or max_val is None:
            raise ValueError("Both min_val and max_val parameters are required")
        
        if np.isnan(min_val) or np.isnan(max_val):
            raise ValueError("Min and max values cannot be NaN")
        
        if np.isinf(min_val) or np.isinf(max_val):
            raise ValueError("Min and max values cannot be infinite")
        
        if min_val >= max_val:
            raise ValueError("Min value must be less than max value")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output signal data"""
        if np.any(np.isnan(y_new)) and not np.any(np.isnan(y_original)):
            raise ValueError("Value clipping produced unexpected NaN values")
        if np.any(np.isinf(y_new)) and not np.any(np.isinf(y_original)):
            raise ValueError("Value clipping produced unexpected infinite values")

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        parsed = {}
        for param in cls.params:
            name = param["name"]
            val = user_input.get(name, param["default"])
            try:
                if val == "":
                    parsed[name] = None
                elif param["type"] == "float":
                    parsed_val = float(val)
                    parsed[name] = parsed_val
                else:
                    parsed[name] = val
            except ValueError as e:
                if "could not convert" in str(e) or "invalid literal" in str(e):
                    raise ValueError(f"{name} must be a valid {param['type']}")
                raise e
        return parsed

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
            
            min_val = params.get('min_val', -3.0)
            max_val = params.get('max_val', 3.0)
            suffix = f"Clip{min_val:g}to{max_val:g}"
            
            return cls.create_new_channel(
                parent=channel, 
                xdata=x, 
                ydata=y_final, 
                params=params,
                suffix=suffix
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Value clipping failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> np.ndarray:
        min_val = params["min_val"]
        max_val = params["max_val"]
        
        y_new = np.clip(y, min_val, max_val)
        
        return y_new
